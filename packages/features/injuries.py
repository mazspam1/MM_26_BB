"""
Injury and availability adjustment system.

Tracks player injuries/suspensions and adjusts team strength accordingly.
Uses RAPM-lite player impact values to quantify the effect of missing players.
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

import structlog

from packages.common.database import get_connection

logger = structlog.get_logger()


class InjuryStatus(str, Enum):
    """Player injury status levels."""

    OUT = "out"  # Definitely out
    DOUBTFUL = "doubtful"  # Very unlikely to play (~25% chance)
    QUESTIONABLE = "questionable"  # 50/50 (~50% chance)
    PROBABLE = "probable"  # Likely to play (~75% chance)
    AVAILABLE = "available"  # No injury concerns


# Expected availability by status
STATUS_AVAILABILITY = {
    InjuryStatus.OUT: 0.0,
    InjuryStatus.DOUBTFUL: 0.25,
    InjuryStatus.QUESTIONABLE: 0.50,
    InjuryStatus.PROBABLE: 0.75,
    InjuryStatus.AVAILABLE: 1.0,
}


@dataclass
class PlayerInjury:
    """Player injury record."""

    player_id: str
    team_id: int
    player_name: str
    status: InjuryStatus
    injury_type: Optional[str]  # e.g., "knee", "ankle", "concussion"
    reported_date: date
    expected_return: Optional[date] = None
    # Impact value (from RAPM-lite, negative = value of missing player)
    player_impact_value: float = 0.0
    minutes_per_game: float = 0.0


@dataclass
class TeamInjuryReport:
    """Team-level injury report for a game."""

    team_id: int
    game_date: date
    injuries: list[PlayerInjury]
    # Aggregated impact
    total_impact_loss: float  # Expected value lost
    availability_factor: float  # 0-1, 1 = fully healthy
    key_player_out: bool  # Any starter (>25 min/game) out?
    worst_status: InjuryStatus


def create_injury_tables() -> None:
    """Create injury tracking tables in database."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_injuries (
                id INTEGER PRIMARY KEY,
                player_id VARCHAR NOT NULL,
                team_id INTEGER NOT NULL,
                player_name VARCHAR NOT NULL,
                status VARCHAR(20) NOT NULL,
                injury_type VARCHAR(50),
                reported_date DATE NOT NULL,
                expected_return DATE,
                player_impact_value FLOAT DEFAULT 0.0,
                minutes_per_game FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_injury_team FOREIGN KEY (team_id) REFERENCES teams(team_id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_injuries_team ON player_injuries(team_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_injuries_player ON player_injuries(player_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_injuries_date ON player_injuries(reported_date)
        """)


def get_team_injury_report(
    team_id: int,
    game_date: date,
) -> TeamInjuryReport:
    """
    Get team injury report for a specific game date.

    Args:
        team_id: Team ID
        game_date: Date of the game

    Returns:
        TeamInjuryReport with all injuries and aggregated impact
    """
    create_injury_tables()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT player_id, team_id, player_name, status, injury_type,
                   reported_date, expected_return, player_impact_value, minutes_per_game
            FROM player_injuries
            WHERE team_id = ?
            AND reported_date <= ?
            AND (expected_return IS NULL OR expected_return >= ?)
            ORDER BY player_impact_value ASC
            """,
            (team_id, game_date.isoformat(), game_date.isoformat()),
        ).fetchall()

    injuries = []
    for row in rows:
        try:
            status = InjuryStatus(row[3])
        except ValueError:
            status = InjuryStatus.AVAILABLE

        injury = PlayerInjury(
            player_id=row[0],
            team_id=row[1],
            player_name=row[2],
            status=status,
            injury_type=row[4],
            reported_date=date.fromisoformat(row[5]) if isinstance(row[5], str) else row[5],
            expected_return=date.fromisoformat(row[6])
            if row[6] and isinstance(row[6], str)
            else None,
            player_impact_value=row[7] or 0.0,
            minutes_per_game=row[8] or 0.0,
        )
        injuries.append(injury)

    # Calculate aggregated impact
    total_impact_loss = 0.0
    key_player_out = False
    worst_status = InjuryStatus.AVAILABLE

    for inj in injuries:
        availability = STATUS_AVAILABILITY.get(inj.status, 1.0)
        # Impact loss = player's value * (1 - availability probability)
        loss = abs(inj.player_impact_value) * (1.0 - availability)
        total_impact_loss += loss

        if inj.status in (InjuryStatus.OUT, InjuryStatus.DOUBTFUL):
            if inj.minutes_per_game >= 20:
                key_player_out = True

        # Track worst status
        status_order = [
            InjuryStatus.AVAILABLE,
            InjuryStatus.PROBABLE,
            InjuryStatus.QUESTIONABLE,
            InjuryStatus.DOUBTFUL,
            InjuryStatus.OUT,
        ]
        if status_order.index(inj.status) > status_order.index(worst_status):
            worst_status = inj.status

    # Availability factor based on key players
    availability_factor = max(0.0, 1.0 - total_impact_loss / 10.0)  # Normalize

    return TeamInjuryReport(
        team_id=team_id,
        game_date=game_date,
        injuries=injuries,
        total_impact_loss=round(total_impact_loss, 2),
        availability_factor=round(availability_factor, 3),
        key_player_out=key_player_out,
        worst_status=worst_status,
    )


def get_injury_spread_adjustment(
    home_report: TeamInjuryReport,
    away_report: TeamInjuryReport,
) -> float:
    """
    Calculate spread adjustment from injury differential.

    Positive = favors home team (away team more injured).

    Args:
        home_report: Home team injury report
        away_report: Away team injury report

    Returns:
        Spread adjustment in points
    """
    # Away team injuries hurt away team (help home)
    # Home team injuries hurt home team (help away)
    adjustment = away_report.total_impact_loss - home_report.total_impact_loss

    # Clamp to reasonable range (injuries rarely worth >8 points)
    return max(-8.0, min(8.0, adjustment))


def get_injury_total_adjustment(
    home_report: TeamInjuryReport,
    away_report: TeamInjuryReport,
) -> float:
    """
    Calculate total adjustment from injuries.

    Key player injuries tend to lower scoring pace.

    Args:
        home_report: Home team injury report
        away_report: Away team injury report

    Returns:
        Total adjustment in points (negative = lower total)
    """
    # Combined injury impact lowers total
    total_loss = home_report.total_impact_loss + away_report.total_impact_loss

    # Roughly 1 point lower total per 2 units of combined impact loss
    adjustment = -total_loss * 0.5

    return max(-6.0, min(0.0, adjustment))


def save_injury(
    player_id: str,
    team_id: int,
    player_name: str,
    status: InjuryStatus,
    injury_type: Optional[str] = None,
    expected_return: Optional[date] = None,
    player_impact_value: float = 0.0,
    minutes_per_game: float = 0.0,
) -> None:
    """Save or update a player injury record."""
    create_injury_tables()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO player_injuries (
                player_id, team_id, player_name, status, injury_type,
                reported_date, expected_return, player_impact_value, minutes_per_game,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                player_id,
                team_id,
                player_name,
                status.value,
                injury_type,
                date.today().isoformat(),
                expected_return.isoformat() if expected_return else None,
                player_impact_value,
                minutes_per_game,
            ),
        )
