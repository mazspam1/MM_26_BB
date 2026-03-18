"""
Play-by-play ingestion for NCAA basketball advanced features.

Fetches play-by-play data from SportsDataverse for:
- Garbage time detection and filtering
- Lineup analysis (5-man unit efficiency)
- Pace variation within games
- Clutch time performance
- Foul rate analysis
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import polars as pl
import structlog

from packages.common.database import get_connection

logger = structlog.get_logger()


@dataclass
class PlayByPlayEvent:
    """Single play-by-play event."""

    game_id: int
    play_number: int
    half: int
    game_clock_seconds: float
    home_score: int
    away_score: int
    play_type: str
    description: str
    home_player_ids: list[str] = field(default_factory=list)
    away_player_ids: list[str] = field(default_factory=list)


@dataclass
class GarbageTimeResult:
    """Result of garbage time analysis."""

    game_id: int
    garbage_time_start_seconds: Optional[float]  # When garbage time started
    garbage_time_possessions: int  # Possessions in garbage time
    garbage_time_pct: float  # % of game in garbage time
    non_garbage_home_score: int
    non_garbage_away_score: int
    non_garbage_spread: int


@dataclass
class LineupUnit:
    """5-man lineup unit performance."""

    team_id: int
    player_ids: tuple[str, ...]
    minutes_played: float
    possessions: int
    points_scored: int
    points_allowed: int
    net_rating: float  # Points per 100 possessions differential


GARBAGE_TIME_THRESHOLD = 25  # Point differential for garbage time
GARBAGE_TIME_MIN_REMAINING = 300  # 5 minutes remaining (in seconds)


def fetch_play_by_play(game_id: int) -> list[PlayByPlayEvent]:
    """
    Fetch play-by-play data for a game from ESPN/SportsDataverse.

    Args:
        game_id: ESPN game ID

    Returns:
        List of PlayByPlayEvent objects
    """
    from packages.common.sportsdataverse_mbb import load_mbb

    mbb = load_mbb()

    try:
        pbp_func = getattr(mbb, "espn_mbb_pbp", None)
        if pbp_func is None:
            logger.warning("Play-by-play function not available in sportsdataverse")
            return []

        pbp_df = pbp_func(game_id=game_id, return_as_pandas=False)
    except Exception as e:
        logger.warning("Failed to fetch play-by-play", game_id=game_id, error=str(e))
        return []

    if pbp_df is None or len(pbp_df) == 0:
        return []

    events = []
    for i, row in enumerate(pbp_df.iter_rows(named=True)):
        try:
            clock_str = row.get("clock", "20:00")
            clock_seconds = _parse_clock_to_seconds(str(clock_str))
            half = int(row.get("period", 1))

            event = PlayByPlayEvent(
                game_id=game_id,
                play_number=i,
                half=half,
                game_clock_seconds=clock_seconds,
                home_score=int(row.get("home_score", 0) or 0),
                away_score=int(row.get("away_score", 0) or 0),
                play_type=str(row.get("type_text", "") or ""),
                description=str(row.get("text", "") or ""),
            )
            events.append(event)
        except Exception:
            continue

    return events


def _parse_clock_to_seconds(clock_str: str) -> float:
    """Parse clock string (MM:SS) to seconds remaining in half."""
    try:
        parts = clock_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except (ValueError, IndexError):
        pass
    return 1200.0  # Default 20:00


def detect_garbage_time(
    events: list[PlayByPlayEvent],
    threshold: int = GARBAGE_TIME_THRESHOLD,
    min_remaining: int = GARBAGE_TIME_MIN_REMAINING,
) -> GarbageTimeResult:
    """
    Detect garbage time periods in a game.

    Garbage time = when the score differential exceeds `threshold` points
    with less than `min_remaining` seconds left in the game.

    Args:
        events: List of play-by-play events
        threshold: Point differential threshold (default 25)
        min_remaining: Minimum seconds remaining to count (default 300 = 5 min)

    Returns:
        GarbageTimeResult with garbage time analysis
    """
    if not events:
        return GarbageTimeResult(
            game_id=0,
            garbage_time_start_seconds=None,
            garbage_time_possessions=0,
            garbage_time_pct=0.0,
            non_garbage_home_score=0,
            non_garbage_away_score=0,
            non_garbage_spread=0,
        )

    game_id = events[0].game_id
    total_game_seconds = 2400.0  # 40 minutes

    # Track garbage time onset
    garbage_start = None
    garbage_possessions = 0

    # Scores at garbage time onset
    gt_home_score = 0
    gt_away_score = 0

    for event in events:
        # Calculate total game time remaining
        total_remaining = (event.half - 1) * 1200 + event.game_clock_seconds

        diff = abs(event.home_score - event.away_score)

        if diff >= threshold and total_remaining <= min_remaining:
            if garbage_start is None:
                garbage_start = total_remaining
                gt_home_score = event.home_score
                gt_away_score = event.away_score
            garbage_possessions += 1

    # Calculate garbage time percentage
    if garbage_start is not None:
        gt_pct = garbage_start / total_game_seconds
    else:
        gt_pct = 0.0

    return GarbageTimeResult(
        game_id=game_id,
        garbage_time_start_seconds=garbage_start,
        garbage_time_possessions=garbage_possessions,
        garbage_time_pct=round(gt_pct, 3),
        non_garbage_home_score=gt_home_score,
        non_garbage_away_score=gt_away_score,
        non_garbage_spread=gt_home_score - gt_away_score,
    )


def filter_garbage_time_from_stats(
    box_scores: list[dict],
    pbp_events: dict[int, list[PlayByPlayEvent]],
) -> list[dict]:
    """
    Adjust box score stats to exclude garbage time.

    Uses PBP to identify garbage time periods, then proportionally
    reduces box score stats by the garbage time percentage.

    Args:
        box_scores: List of box score dicts with game_id, team_id, etc.
        pbp_events: Dict of game_id -> list of PlayByPlayEvent

    Returns:
        Adjusted box scores with garbage time filtered
    """
    adjusted = []

    for box in box_scores:
        game_id = box.get("game_id")
        events = pbp_events.get(game_id, [])

        if not events:
            adjusted.append(box)
            continue

        gt_result = detect_garbage_time(events)

        if gt_result.garbage_time_pct > 0.05:  # Only adjust if >5% garbage time
            # Scale factor: keep (1 - gt_pct) of stats
            scale = 1.0 - gt_result.garbage_time_pct

            adjusted_box = box.copy()
            for stat in [
                "field_goals_made",
                "field_goals_attempted",
                "three_pointers_made",
                "three_pointers_attempted",
                "free_throws_made",
                "free_throws_attempted",
                "offensive_rebounds",
                "defensive_rebounds",
                "turnovers",
                "assists",
                "steals",
                "blocks",
                "personal_fouls",
                "points",
            ]:
                if stat in adjusted_box:
                    adjusted_box[stat] = int(round(adjusted_box[stat] * scale))

            adjusted_box["garbage_time_filtered"] = True
            adjusted_box["garbage_time_pct"] = gt_result.garbage_time_pct
            adjusted.append(adjusted_box)
        else:
            box["garbage_time_filtered"] = False
            box["garbage_time_pct"] = 0.0
            adjusted.append(box)

    return adjusted


def save_pbp_to_db(game_id: int, events: list[PlayByPlayEvent]) -> int:
    """Save play-by-play events to database."""
    if not events:
        return 0

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS play_by_play (
                game_id INTEGER NOT NULL,
                play_number INTEGER NOT NULL,
                half INTEGER NOT NULL,
                game_clock_seconds FLOAT NOT NULL,
                home_score INTEGER NOT NULL,
                away_score INTEGER NOT NULL,
                play_type VARCHAR,
                description VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (game_id, play_number),
                CONSTRAINT fk_pbp_game FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)

        for event in events:
            conn.execute(
                """
                INSERT OR REPLACE INTO play_by_play (
                    game_id, play_number, half, game_clock_seconds,
                    home_score, away_score, play_type, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.game_id,
                    event.play_number,
                    event.half,
                    event.game_clock_seconds,
                    event.home_score,
                    event.away_score,
                    event.play_type,
                    event.description,
                ),
            )

    logger.info("Play-by-play saved", game_id=game_id, events=len(events))
    return len(events)
