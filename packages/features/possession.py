"""
Possession calculations for NCAA basketball.

Based on Dean Oliver's possession formula:
Possessions = FGA - OR + TO + 0.475 * FTA

References:
- KenPom: https://kenpom.com/blog/help-with-team-page/
- Basketball Reference methodology
"""

from typing import Optional

import structlog

from packages.common.schemas import BoxScore

logger = structlog.get_logger()

# Dean Oliver's FTA coefficient
FTA_COEFFICIENT = 0.475


def calculate_possessions_from_stats(
    field_goals_attempted: int,
    offensive_rebounds: int,
    turnovers: int,
    free_throws_attempted: int,
) -> float:
    """
    Calculate possessions using Dean Oliver's formula.

    Formula: Possessions = FGA - OR + TO + 0.475 * FTA

    Args:
        field_goals_attempted: Total field goal attempts
        offensive_rebounds: Offensive rebounds
        turnovers: Total turnovers
        free_throws_attempted: Free throw attempts

    Returns:
        Estimated number of possessions
    """
    possessions = (
        field_goals_attempted
        - offensive_rebounds
        + turnovers
        + FTA_COEFFICIENT * free_throws_attempted
    )
    return max(0.0, possessions)


def calculate_possessions_from_boxscore(box: BoxScore) -> float:
    """
    Calculate possessions from a BoxScore object.

    Args:
        box: BoxScore object with game stats

    Returns:
        Estimated number of possessions
    """
    return calculate_possessions_from_stats(
        field_goals_attempted=box.field_goals_attempted,
        offensive_rebounds=box.offensive_rebounds,
        turnovers=box.turnovers,
        free_throws_attempted=box.free_throws_attempted,
    )


def calculate_game_possessions(
    home_box: BoxScore,
    away_box: BoxScore,
    method: str = "average",
) -> float:
    """
    Calculate total game possessions from both team box scores.

    Args:
        home_box: Home team box score
        away_box: Away team box score
        method: "average" (mean of both), "home", or "away"

    Returns:
        Estimated game possessions
    """
    home_poss = calculate_possessions_from_boxscore(home_box)
    away_poss = calculate_possessions_from_boxscore(away_box)

    if method == "average":
        return (home_poss + away_poss) / 2
    elif method == "home":
        return home_poss
    elif method == "away":
        return away_poss
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_tempo(possessions: float, game_minutes: float = 40.0) -> float:
    """
    Calculate tempo (possessions per 40 minutes).

    Args:
        possessions: Total possessions in the game
        game_minutes: Length of game (40 for regulation)

    Returns:
        Tempo (possessions per 40 minutes)
    """
    if game_minutes <= 0:
        raise ValueError("game_minutes must be positive")

    return possessions * (40.0 / game_minutes)


def expected_game_possessions(
    home_tempo: float,
    away_tempo: float,
    league_avg_tempo: float = 68.0,
    method: str = "harmonic",
) -> float:
    """
    Project expected possessions for a game based on team tempos.

    Args:
        home_tempo: Home team's tempo (poss/40)
        away_tempo: Away team's tempo (poss/40)
        league_avg_tempo: League average tempo for normalization
        method: "harmonic" (harmonic mean), "arithmetic", or "geometric"

    Returns:
        Expected game possessions
    """
    if method == "harmonic":
        # Harmonic mean - tends to be conservative
        if home_tempo <= 0 or away_tempo <= 0:
            return league_avg_tempo
        return 2 * (home_tempo * away_tempo) / (home_tempo + away_tempo)

    elif method == "arithmetic":
        # Simple average
        return (home_tempo + away_tempo) / 2

    elif method == "geometric":
        # Geometric mean
        import math
        if home_tempo <= 0 or away_tempo <= 0:
            return league_avg_tempo
        return math.sqrt(home_tempo * away_tempo)

    else:
        raise ValueError(f"Unknown method: {method}")


def tempo_context_adjustment(
    base_possessions: float,
    is_neutral_site: bool = False,
    home_rest_days: Optional[int] = None,
    away_rest_days: Optional[int] = None,
    altitude_diff: Optional[float] = None,
) -> float:
    """
    Apply context adjustments to expected possessions.

    Args:
        base_possessions: Base expected possessions
        is_neutral_site: Whether game is at neutral site
        home_rest_days: Days of rest for home team
        away_rest_days: Days of rest for away team
        altitude_diff: Altitude difference (away - home) in feet

    Returns:
        Adjusted expected possessions
    """
    adjustment = 1.0

    # Neutral site games tend to be slightly slower
    if is_neutral_site:
        adjustment *= 0.98

    # Back-to-back games can slow tempo
    if home_rest_days is not None and home_rest_days == 0:
        adjustment *= 0.97
    if away_rest_days is not None and away_rest_days == 0:
        adjustment *= 0.97

    # High altitude can affect pace (minimal effect)
    if altitude_diff is not None and altitude_diff > 5000:
        adjustment *= 0.99

    return base_possessions * adjustment


def possessions_per_100(stat_value: float, possessions: float) -> float:
    """
    Convert a stat to per-100-possessions rate.

    Args:
        stat_value: Raw stat value
        possessions: Team possessions in the game

    Returns:
        Stat per 100 possessions
    """
    if possessions <= 0:
        return 0.0
    return (stat_value / possessions) * 100


def points_per_possession(points: int, possessions: float) -> float:
    """
    Calculate points per possession (offensive efficiency).

    Args:
        points: Total points scored
        possessions: Team possessions

    Returns:
        Points per possession (typically 0.85-1.15)
    """
    if possessions <= 0:
        return 0.0
    return points / possessions


def points_per_100_possessions(points: int, possessions: float) -> float:
    """
    Calculate points per 100 possessions (KenPom-style efficiency).

    Args:
        points: Total points scored
        possessions: Team possessions

    Returns:
        Points per 100 possessions (typically 85-115)
    """
    return possessions_per_100(float(points), possessions)
