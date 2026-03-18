"""Free ESPN pickcenter odds ingestion for NCAA basketball."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from packages.common.sportsdataverse_mbb import load_mbb
from packages.common.schemas import LineSnapshot
from packages.ingest.odds_api import save_line_snapshots_to_db

logger = structlog.get_logger()
_mbb = load_mbb()
espn_mbb_pbp = _mbb.espn_mbb_pbp


def _parse_optional_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_int(value: object) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fetch_pickcenter_snapshots(game_id: int) -> list[LineSnapshot]:
    """Fetch current and open DraftKings lines exposed by ESPN pickcenter."""
    data = espn_mbb_pbp(game_id=game_id)
    picks = data.get("pickcenter") or []
    if not picks:
        return []

    pick = picks[0]
    provider = ((pick.get("provider") or {}).get("name") or "ESPN").strip()
    bookmaker = f"espn_{provider.lower().replace(' ', '_')}"
    now = datetime.now(timezone.utc)

    point_spread = pick.get("pointSpread") or {}
    total_market = pick.get("total") or {}
    moneyline = pick.get("moneyline") or {}

    current_snapshot = LineSnapshot(
        game_id=game_id,
        bookmaker=bookmaker,
        snapshot_timestamp=now,
        snapshot_type="current",
        spread_home=_parse_optional_float(
            ((point_spread.get("home") or {}).get("close") or {}).get("line")
        ),
        spread_home_price=_parse_optional_int(
            ((point_spread.get("home") or {}).get("close") or {}).get("odds")
        ),
        spread_away=_parse_optional_float(
            ((point_spread.get("away") or {}).get("close") or {}).get("line")
        ),
        spread_away_price=_parse_optional_int(
            ((point_spread.get("away") or {}).get("close") or {}).get("odds")
        ),
        total_line=_parse_optional_float(pick.get("overUnder")),
        over_price=_parse_optional_int(
            ((total_market.get("over") or {}).get("close") or {}).get("odds")
        ),
        under_price=_parse_optional_int(
            ((total_market.get("under") or {}).get("close") or {}).get("odds")
        ),
        home_ml=_parse_optional_int(((moneyline.get("home") or {}).get("close") or {}).get("odds")),
        away_ml=_parse_optional_int(((moneyline.get("away") or {}).get("close") or {}).get("odds")),
    )
    snapshots = [current_snapshot]

    open_home_line = _parse_optional_float(
        ((point_spread.get("home") or {}).get("open") or {}).get("line")
    )
    open_away_line = _parse_optional_float(
        ((point_spread.get("away") or {}).get("open") or {}).get("line")
    )
    open_total_text = ((total_market.get("over") or {}).get("open") or {}).get("line")
    open_total = _parse_optional_float(str(open_total_text)[1:] if open_total_text else None)

    if any(value is not None for value in (open_home_line, open_away_line, open_total)):
        snapshots.append(
            LineSnapshot(
                game_id=game_id,
                bookmaker=bookmaker,
                snapshot_timestamp=now,
                snapshot_type="open",
                spread_home=open_home_line,
                spread_home_price=_parse_optional_int(
                    ((point_spread.get("home") or {}).get("open") or {}).get("odds")
                ),
                spread_away=open_away_line,
                spread_away_price=_parse_optional_int(
                    ((point_spread.get("away") or {}).get("open") or {}).get("odds")
                ),
                total_line=open_total,
                over_price=_parse_optional_int(
                    ((total_market.get("over") or {}).get("open") or {}).get("odds")
                ),
                under_price=_parse_optional_int(
                    ((total_market.get("under") or {}).get("open") or {}).get("odds")
                ),
                home_ml=_parse_optional_int(
                    ((moneyline.get("home") or {}).get("open") or {}).get("odds")
                ),
                away_ml=_parse_optional_int(
                    ((moneyline.get("away") or {}).get("open") or {}).get("odds")
                ),
            )
        )

    return snapshots


def ingest_pickcenter_for_games(game_ids: list[int]) -> int:
    """Fetch and persist ESPN pickcenter lines for the provided game ids."""
    snapshots: list[LineSnapshot] = []
    for game_id in game_ids:
        try:
            snapshots.extend(fetch_pickcenter_snapshots(game_id))
        except Exception as exc:
            logger.warning(
                "Failed to ingest ESPN pickcenter for game", game_id=game_id, error=str(exc)
            )

    if not snapshots:
        logger.info("No ESPN pickcenter lines found", games=len(game_ids))
        return 0

    save_line_snapshots_to_db(snapshots)
    logger.info("ESPN pickcenter lines saved", snapshots=len(snapshots), games=len(game_ids))
    return len(snapshots)
