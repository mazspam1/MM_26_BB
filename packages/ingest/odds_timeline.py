"""
Historical odds timeline storage for proper CLV analysis.

Stores multiple odds snapshots per game (open, +24h, +6h, +1h, close)
to enable proper Closing Line Value (CLV) tracking and analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import structlog

from packages.common.database import get_connection

logger = structlog.get_logger()


@dataclass
class OddsSnapshot:
    """Point-in-time odds snapshot with metadata."""

    game_id: int
    bookmaker: str
    timestamp: datetime
    snapshot_label: str  # "open", "t_minus_24h", "t_minus_6h", "t_minus_1h", "close"
    spread_home: Optional[float]
    spread_away: Optional[float]
    total_line: Optional[float]
    home_ml: Optional[int]
    away_ml: Optional[int]
    # Metadata
    is_closing: bool = False
    is_opening: bool = False


@dataclass
class CLVTimeline:
    """Full CLV timeline for a game."""

    game_id: int
    snapshots: list[OddsSnapshot]
    # Computed CLV metrics
    spread_open: Optional[float] = None
    spread_close: Optional[float] = None
    spread_clv: Optional[float] = None  # Positive = beat the close
    total_open: Optional[float] = None
    total_close: Optional[float] = None
    total_clv: Optional[float] = None
    # Line movement
    spread_movement: Optional[float] = None  # Close - Open
    total_movement: Optional[float] = None
    steam_move: bool = False  # Rapid significant movement


def create_odds_timeline_tables() -> None:
    """Create odds timeline tables."""
    with get_connection() as conn:
        # Enhanced line_snapshots table already exists from schema.sql
        # Add odds_timeline for multiple snapshots per game
        conn.execute("""
            CREATE TABLE IF NOT EXISTS odds_timeline (
                id INTEGER PRIMARY KEY,
                game_id INTEGER NOT NULL,
                bookmaker VARCHAR(50) NOT NULL,
                snapshot_timestamp TIMESTAMP NOT NULL,
                snapshot_label VARCHAR(20) NOT NULL,
                spread_home FLOAT,
                spread_away FLOAT,
                total_line FLOAT,
                home_ml INTEGER,
                away_ml INTEGER,
                is_closing BOOLEAN DEFAULT FALSE,
                is_opening BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_odds_timeline_game FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_timeline_game ON odds_timeline(game_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_timeline_label ON odds_timeline(snapshot_label)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_timeline_ts ON odds_timeline(snapshot_timestamp)
        """)


def save_odds_snapshot(snapshot: OddsSnapshot) -> None:
    """Save an odds snapshot to the timeline."""
    create_odds_timeline_tables()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO odds_timeline (
                game_id, bookmaker, snapshot_timestamp, snapshot_label,
                spread_home, spread_away, total_line,
                home_ml, away_ml, is_closing, is_opening
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.game_id,
                snapshot.bookmaker,
                snapshot.timestamp.isoformat(),
                snapshot.snapshot_label,
                snapshot.spread_home,
                snapshot.spread_away,
                snapshot.total_line,
                snapshot.home_ml,
                snapshot.away_ml,
                snapshot.is_closing,
                snapshot.is_opening,
            ),
        )


def get_clv_timeline(game_id: int, bookmaker: Optional[str] = None) -> CLVTimeline:
    """
    Get full CLV timeline for a game.

    Args:
        game_id: Game ID
        bookmaker: Optional bookmaker filter (default: use sharpest available)

    Returns:
        CLVTimeline with all snapshots and CLV metrics
    """
    create_odds_timeline_tables()

    with get_connection() as conn:
        if bookmaker:
            rows = conn.execute(
                """
                SELECT game_id, bookmaker, snapshot_timestamp, snapshot_label,
                       spread_home, spread_away, total_line,
                       home_ml, away_ml, is_closing, is_opening
                FROM odds_timeline
                WHERE game_id = ? AND bookmaker = ?
                ORDER BY snapshot_timestamp ASC
                """,
                (game_id, bookmaker),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT game_id, bookmaker, snapshot_timestamp, snapshot_label,
                       spread_home, spread_away, total_line,
                       home_ml, away_ml, is_closing, is_opening
                FROM odds_timeline
                WHERE game_id = ?
                ORDER BY snapshot_timestamp ASC
                """,
                (game_id,),
            ).fetchall()

    snapshots = []
    for row in rows:
        snap = OddsSnapshot(
            game_id=row[0],
            bookmaker=row[1],
            timestamp=datetime.fromisoformat(row[2]) if isinstance(row[2], str) else row[2],
            snapshot_label=row[3],
            spread_home=row[4],
            spread_away=row[5],
            total_line=row[6],
            home_ml=row[7],
            away_ml=row[8],
            is_closing=bool(row[9]),
            is_opening=bool(row[10]),
        )
        snapshots.append(snap)

    # Compute CLV metrics
    timeline = CLVTimeline(game_id=game_id, snapshots=snapshots)

    if snapshots:
        # Find opening and closing
        opening = next((s for s in snapshots if s.is_opening), snapshots[0] if snapshots else None)
        closing = next((s for s in snapshots if s.is_closing), snapshots[-1] if snapshots else None)

        if opening:
            timeline.spread_open = opening.spread_home
            timeline.total_open = opening.total_line

        if closing:
            timeline.spread_close = closing.spread_home
            timeline.total_close = closing.total_line

        # Compute CLV (positive = beat the close, i.e., got better number)
        if timeline.spread_open is not None and timeline.spread_close is not None:
            timeline.spread_clv = timeline.spread_close - timeline.spread_open
            timeline.spread_movement = timeline.spread_close - timeline.spread_open

        if timeline.total_open is not None and timeline.total_close is not None:
            timeline.total_clv = timeline.total_close - timeline.total_open
            timeline.total_movement = timeline.total_close - timeline.total_open

        # Detect steam moves (rapid significant movement)
        if len(snapshots) >= 3:
            spreads = [s.spread_home for s in snapshots if s.spread_home is not None]
            if len(spreads) >= 3:
                max_move = max(abs(spreads[i] - spreads[i - 1]) for i in range(1, len(spreads)))
                timeline.steam_move = max_move >= 1.5  # 1.5+ point move = steam

    return timeline


def compute_our_clv(
    our_spread: float,
    market_spread_at_prediction: float,
    closing_spread: float,
    our_side: str,  # "home" or "away"
) -> float:
    """
    Compute CLV for our prediction vs the closing line.

    CLV = closing_line - market_line_at_prediction (adjusted for side)

    Positive CLV means we got a better number than the close.

    Args:
        our_spread: Our predicted spread
        market_spread_at_prediction: Market spread when we made prediction
        closing_spread: Closing spread (sharp reference)
        our_side: Which side we're recommending

    Returns:
        CLV in points (positive = beat the close)
    """
    if our_side == "home":
        # We bet home, so we want the spread to go down (more favorable)
        # CLV = how much better our number is vs close
        return market_spread_at_prediction - closing_spread
    else:
        # We bet away, so we want the spread to go up (more favorable)
        return closing_spread - market_spread_at_prediction


def save_clv_report(
    game_id: int,
    our_spread: float,
    our_total: float,
    market_spread_at_bet: float,
    market_total_at_bet: float,
    closing_spread: float,
    closing_total: float,
    our_spread_side: str,
    our_total_side: str,
) -> None:
    """Save CLV report to database."""
    spread_clv = compute_our_clv(our_spread, market_spread_at_bet, closing_spread, our_spread_side)
    total_clv = (
        closing_total - market_total_at_bet
        if our_total_side == "over"
        else market_total_at_bet - closing_total
    )

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO clv_reports (
                game_id, prediction_timestamp, our_spread, our_total,
                market_spread_at_bet, market_total_at_bet,
                closing_spread, closing_total, spread_clv, total_clv
            ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id,
                our_spread,
                our_total,
                market_spread_at_bet,
                market_total_at_bet,
                closing_spread,
                closing_total,
                spread_clv,
                total_clv,
            ),
        )
