"""
Script to fetch and populate DraftKings betting splits data.

Run this script to fetch betting splits from DraftKings and save them to the database.
"""

from datetime import date, timedelta

import structlog

from packages.common.database import get_connection, init_database
from packages.ingest.draftkings_splits import (
    DraftKingsSplitsScraper,
    normalize_team_name,
    save_betting_splits_to_db,
)

logger = structlog.get_logger()


def fetch_and_save_splits(target_date: date = None) -> int:
    """
    Fetch betting splits for a date and save to database.

    Args:
        target_date: Date to fetch splits for (default: today)

    Returns:
        Number of games with splits saved
    """
    if target_date is None:
        target_date = date.today()

    init_database()
    scraper = DraftKingsSplitsScraper()
    saved_count = 0

    try:
        # Fetch spread splits
        logger.info("Fetching spread betting splits", date=target_date)
        spread_splits = scraper.fetch_spread_splits(days=7)

        # Fetch total splits
        logger.info("Fetching total betting splits", date=target_date)
        total_splits = scraper.fetch_total_splits(days=7)

        # Get games for the target date
        with get_connection() as conn:
            games = conn.execute(
                """
                SELECT game_id, home_team_name, away_team_name, game_date
                FROM games
                WHERE game_date = ?
                """,
                (target_date.isoformat(),),
            ).fetchall()

        logger.info(f"Found {len(games)} games for {target_date}")

        # Match splits to games and save
        for game_id, home_team, away_team, game_date in games:
            game_date_obj = date.fromisoformat(game_date) if isinstance(game_date, str) else game_date

            # Try to match spread splits
            spread_match = scraper.match_game_to_splits(
                spread_splits, home_team, away_team, game_date_obj
            )

            # Try to match total splits
            total_match = scraper.match_game_to_splits(
                total_splits, home_team, away_team, game_date_obj
            )

            # Combine splits data
            combined_splits = {}
            if spread_match:
                spread_line = spread_match.get("spread_line")
                favored_team = spread_match.get("spread_favored_team")
                combined_splits.update({
                    "spread_line": spread_line,
                    "spread_favored_handle_pct": spread_match.get("spread_favored_handle_pct"),
                    "spread_favored_bets_pct": spread_match.get("spread_favored_bets_pct"),
                    "spread_underdog_handle_pct": spread_match.get("spread_underdog_handle_pct"),
                    "spread_underdog_bets_pct": spread_match.get("spread_underdog_bets_pct"),
                })
                if spread_line is not None and favored_team:
                    # spread_line from DraftKings is always negative (e.g., -33.5 for the favorite)
                    # Convert to POINT DIFFERENTIAL convention (same as our model):
                    #   - Positive = home team wins by X points
                    #   - Negative = away team wins by X points (home loses)
                    # This matches our model's proj_spread = home_score - away_score
                    favored_norm = normalize_team_name(favored_team)
                    home_norm = normalize_team_name(home_team)
                    away_norm = normalize_team_name(away_team)
                    if favored_norm and (favored_norm in home_norm or home_norm in favored_norm):
                        # Home team is favored - flip sign to positive (home wins by X)
                        combined_splits["spread_line_home"] = -spread_line
                    elif favored_norm and (favored_norm in away_norm or away_norm in favored_norm):
                        # Away team is favored - keep negative (home loses by X)
                        combined_splits["spread_line_home"] = spread_line

            if total_match:
                combined_splits.update({
                    "total_line": total_match.get("total_line"),
                    "total_over_handle_pct": total_match.get("total_over_handle_pct"),
                    "total_over_bets_pct": total_match.get("total_over_bets_pct"),
                    "total_under_handle_pct": total_match.get("total_under_handle_pct"),
                    "total_under_bets_pct": total_match.get("total_under_bets_pct"),
                })

            # Log spread_favored_team for debugging
            if spread_match:
                logger.debug(
                    "Spread match details",
                    game_id=game_id,
                    spread_line=spread_match.get("spread_line"),
                    spread_favored_team=spread_match.get("spread_favored_team"),
                    spread_underdog_team=spread_match.get("spread_underdog_team"),
                    home_team=home_team,
                    away_team=away_team,
                )

            # Save if we have any splits data
            if combined_splits and any(v is not None for v in combined_splits.values()):
                try:
                    save_betting_splits_to_db(game_id, combined_splits)
                    saved_count += 1
                    logger.info(
                        "Saved betting splits",
                        game_id=game_id,
                        home=home_team,
                        away=away_team,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to save betting splits",
                        game_id=game_id,
                        error=str(e),
                    )

    finally:
        scraper.close()

    logger.info(f"Saved betting splits for {saved_count} games")
    return saved_count


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_date = date.fromisoformat(sys.argv[1])
    else:
        target_date = date.today()

    count = fetch_and_save_splits(target_date)
    print(f"Saved betting splits for {count} games")

