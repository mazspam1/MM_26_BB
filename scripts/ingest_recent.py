"""
Ingest recent games + boxscores into the database.

This script:
1) Updates games table for a date range (scores + status)
2) Refreshes team_game_stats from ESPN boxscores
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import argparse
import structlog

from packages.common.database import get_connection
from packages.common.season import infer_season_year, season_start_date
from packages.ingest.espn_api import fetch_schedule, save_games_to_db
from packages.ingest.espn_enhanced import ingest_team_stats_for_date_range

logger = structlog.get_logger()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def ingest_range(start_date: date, end_date: date) -> None:
    logger.info(
        "Ingesting recent games",
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )

    # Update games table (scores + status)
    current = start_date
    while current <= end_date:
        games = fetch_schedule(current)
        if games:
            saved = save_games_to_db(games, skip_fk_errors=True)
            logger.info(
                "Games saved",
                date=current.isoformat(),
                fetched=len(games),
                saved=saved,
            )
        current += timedelta(days=1)

    # Update team-game stats from boxscores
    season = infer_season_year(end_date)
    stats_df = ingest_team_stats_for_date_range(start_date, end_date, season=season)
    logger.info("Team stats refresh complete", rows=len(stats_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest recent games + boxscores")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days-back", type=int, default=7, help="Days back from yesterday")
    parser.add_argument("--season-to-date", action="store_true", help="Backfill from season start")
    args = parser.parse_args()

    if args.start and args.end:
        start_date = _parse_date(args.start)
        end_date = _parse_date(args.end)
    else:
        end_date = date.today() - timedelta(days=1)
        season_start = season_start_date(end_date)
        start_date = end_date - timedelta(days=max(args.days_back - 1, 0))

        if args.season_to_date:
            start_date = season_start
        else:
            with get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT MIN(CAST(game_date AS DATE))
                    FROM team_game_stats
                    WHERE CAST(game_date AS DATE) >= ?
                    """,
                    (season_start.isoformat(),),
                ).fetchone()

            min_date = row[0] if row else None
            if isinstance(min_date, str):
                min_date = date.fromisoformat(min_date)

            if min_date is None or min_date > season_start:
                logger.info("Backfilling team stats from season start", season_start=season_start.isoformat())
                start_date = season_start

    if end_date < start_date:
        raise ValueError("End date must be on or after start date")

    ingest_range(start_date, end_date)


if __name__ == "__main__":
    main()
