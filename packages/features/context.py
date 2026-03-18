"""
Contextual features for game prediction.

Includes:
- Rest days calculation
- Travel distance estimation
- Neutral site detection
- Back-to-back detection
- Conference play indicators
- Season phase classification

These factors affect team performance beyond pure efficiency metrics.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import structlog

from packages.common.schemas import Game, SeasonPhase

logger = structlog.get_logger()

# Conference tournament typically starts first week of March
CONFERENCE_TOURNAMENT_START = (3, 1)  # March 1

# NCAA tournament selection Sunday is typically mid-March
SELECTION_SUNDAY = (3, 15)  # Around March 15

# March Madness starts around March 17-18
NCAA_TOURNAMENT_START = (3, 17)


@dataclass
class RestContext:
    """Rest and schedule context for a team."""

    team_id: int
    rest_days: int  # Days since last game
    is_back_to_back: bool  # Played yesterday
    games_last_7_days: int  # Recent workload
    games_last_14_days: int


@dataclass
class GameContext:
    """Full context for a game matchup."""

    game_id: int
    home_rest_days: int
    away_rest_days: int
    home_back_to_back: bool
    away_back_to_back: bool
    rest_advantage: int  # Positive = home has more rest
    is_neutral_site: bool
    is_conference_game: bool
    season_phase: SeasonPhase
    is_rivalry: bool  # Future: detect rivalry games


# Rest day adjustments (points)
# Based on research showing fatigue effects
REST_ADJUSTMENTS = {
    0: -2.5,  # Back-to-back (played yesterday)
    1: -1.0,  # 1 day rest
    2: 0.0,   # Normal rest (reference)
    3: 0.5,   # Extra rest
    4: 0.5,   # Extra rest (diminishing returns)
}

# Default rest adjustment for 5+ days
DEFAULT_LONG_REST = 0.5


def calculate_rest_days(
    team_id: int,
    game_date: date,
    previous_games: list[Game],
) -> int:
    """
    Calculate rest days since team's last game.

    Args:
        team_id: Team to calculate rest for
        game_date: Date of upcoming game
        previous_games: List of team's previous games

    Returns:
        Number of rest days (days since last game)
    """
    # Filter to team's games
    team_games = [
        g for g in previous_games
        if (g.home_team_id == team_id or g.away_team_id == team_id)
        and g.game_date < game_date
    ]

    if not team_games:
        # No previous games found - assume full rest
        return 7

    # Find most recent game
    last_game = max(team_games, key=lambda g: g.game_date)
    rest_days = (game_date - last_game.game_date).days

    return rest_days


def calculate_rest_context(
    team_id: int,
    game_date: date,
    previous_games: list[Game],
) -> RestContext:
    """
    Calculate full rest context for a team.

    Args:
        team_id: Team to analyze
        game_date: Date of upcoming game
        previous_games: Team's previous games

    Returns:
        RestContext dataclass
    """
    # Filter to team's games before this date
    team_games = [
        g for g in previous_games
        if (g.home_team_id == team_id or g.away_team_id == team_id)
        and g.game_date < game_date
    ]

    rest_days = calculate_rest_days(team_id, game_date, previous_games)
    is_back_to_back = rest_days == 1

    # Games in last 7 days
    week_ago = game_date - timedelta(days=7)
    games_last_7 = len([g for g in team_games if g.game_date > week_ago])

    # Games in last 14 days
    two_weeks_ago = game_date - timedelta(days=14)
    games_last_14 = len([g for g in team_games if g.game_date > two_weeks_ago])

    return RestContext(
        team_id=team_id,
        rest_days=rest_days,
        is_back_to_back=is_back_to_back,
        games_last_7_days=games_last_7,
        games_last_14_days=games_last_14,
    )


def get_rest_adjustment(rest_days: int) -> float:
    """
    Get point adjustment for rest days.

    Args:
        rest_days: Days since last game

    Returns:
        Point adjustment (positive = advantage)
    """
    if rest_days in REST_ADJUSTMENTS:
        return REST_ADJUSTMENTS[rest_days]
    elif rest_days >= 5:
        return DEFAULT_LONG_REST
    else:
        return 0.0


def calculate_rest_differential(
    home_rest_days: int,
    away_rest_days: int,
) -> float:
    """
    Calculate rest advantage differential.

    Args:
        home_rest_days: Home team rest days
        away_rest_days: Away team rest days

    Returns:
        Point adjustment for home team (positive = home advantage)
    """
    home_adj = get_rest_adjustment(home_rest_days)
    away_adj = get_rest_adjustment(away_rest_days)
    return home_adj - away_adj


def detect_season_phase(game_date: date, is_conference_game: bool) -> SeasonPhase:
    """
    Detect season phase from date and game context.

    Args:
        game_date: Date of the game
        is_conference_game: Whether it's a conference game

    Returns:
        SeasonPhase enum
    """
    month = game_date.month
    day = game_date.day

    # NCAA Tournament (late March through early April)
    if (month == 3 and day >= NCAA_TOURNAMENT_START[1]) or month == 4:
        return SeasonPhase.TOURNAMENT

    # Conference tournament (early March)
    if month == 3 and day < NCAA_TOURNAMENT_START[1]:
        return SeasonPhase.TOURNAMENT  # Conference tournaments

    # Early season (November through mid-December)
    if month == 11 or (month == 12 and day <= 15):
        return SeasonPhase.EARLY

    # Conference play vs non-conference
    if is_conference_game:
        return SeasonPhase.CONFERENCE
    else:
        return SeasonPhase.NON_CONFERENCE


def build_game_context(
    game: Game,
    home_previous_games: list[Game],
    away_previous_games: list[Game],
) -> GameContext:
    """
    Build full game context from game and history.

    Args:
        game: The game to build context for
        home_previous_games: Home team's previous games
        away_previous_games: Away team's previous games

    Returns:
        GameContext dataclass
    """
    home_rest = calculate_rest_context(
        game.home_team_id, game.game_date, home_previous_games
    )
    away_rest = calculate_rest_context(
        game.away_team_id, game.game_date, away_previous_games
    )

    season_phase = detect_season_phase(game.game_date, game.conference_game)

    return GameContext(
        game_id=game.game_id,
        home_rest_days=home_rest.rest_days,
        away_rest_days=away_rest.rest_days,
        home_back_to_back=home_rest.is_back_to_back,
        away_back_to_back=away_rest.is_back_to_back,
        rest_advantage=home_rest.rest_days - away_rest.rest_days,
        is_neutral_site=game.neutral_site,
        is_conference_game=game.conference_game,
        season_phase=season_phase,
        is_rivalry=False,  # TODO: Add rivalry detection
    )


def calculate_context_adjustment(context: GameContext) -> float:
    """
    Calculate total context-based point adjustment for home team.

    Args:
        context: GameContext object

    Returns:
        Point adjustment for home team prediction
    """
    adjustment = 0.0

    # Rest differential
    rest_diff = calculate_rest_differential(
        context.home_rest_days, context.away_rest_days
    )
    adjustment += rest_diff

    # Back-to-back penalty already in rest adjustment
    # Additional penalty for extreme situations
    if context.home_back_to_back and not context.away_back_to_back:
        adjustment -= 1.0  # Extra penalty for home B2B vs rested opponent
    elif context.away_back_to_back and not context.home_back_to_back:
        adjustment += 1.0  # Bonus for rested home vs tired road team

    # Season phase adjustments
    # Tournament games tend to be closer (better teams, higher stakes)
    if context.season_phase == SeasonPhase.TOURNAMENT:
        # Tournament environments reduce home court effect
        # and increase variance
        pass  # Handled by is_neutral_site typically

    logger.debug(
        "Context adjustment calculated",
        game_id=context.game_id,
        adjustment=adjustment,
        rest_diff=rest_diff,
    )

    return adjustment


# Future: Travel distance estimation
# Would need team location data (lat/long)
# For now, we rely on rest days as proxy

def estimate_travel_fatigue(
    away_team_id: int,
    venue_location: Optional[str],
    team_locations: Optional[dict[int, tuple[float, float]]] = None,
) -> float:
    """
    Estimate travel fatigue adjustment.

    For MVP, returns 0 - would need team location data.

    Args:
        away_team_id: Away team ID
        venue_location: Venue city/state
        team_locations: Dict of team_id -> (lat, lon)

    Returns:
        Point adjustment for travel fatigue
    """
    # TODO: Implement with actual location data
    # Would calculate great circle distance and apply adjustment
    # e.g., cross-country flights = -1.0 points, short trips = 0
    return 0.0
