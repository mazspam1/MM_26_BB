"""
Season helpers for NCAA scheduling.
"""

from datetime import date


def infer_season_year(target_date: date) -> int:
    """
    Infer season year from a game date.

    NCAA seasons run Nov-Apr; we store the year of the season end.
    Example: Nov 2024 games map to season 2025.
    """
    if target_date.month >= 7:
        return target_date.year + 1
    return target_date.year


def season_start_date(target_date: date) -> date:
    """
    Return the season start date for the target date.

    We use July 1 as a safe boundary before preseason games.
    """
    season_year = infer_season_year(target_date)
    return date(season_year - 1, 7, 1)
