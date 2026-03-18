"""
ESPN data connector via sportsdataverse-py.

Primary data source for:
- Game schedules
- Team information
- Box scores
- Play-by-play data
"""

from datetime import date, datetime
from typing import Optional

import polars as pl
import structlog

from packages.common.sportsdataverse_mbb import load_mbb

from packages.common.database import get_connection, insert_dataframe
from packages.common.schemas import BoxScore, Game, Team, VenueType, SeasonPhase

logger = structlog.get_logger()
_mbb = load_mbb()
load_mbb_team_boxscore = _mbb.load_mbb_team_boxscore
espn_mbb_schedule = _mbb.espn_mbb_schedule
espn_mbb_teams = _mbb.espn_mbb_teams
espn_mbb_calendar = getattr(_mbb, "espn_mbb_calendar", None)

# ESPN Groups: 50 = Division I, 51 = Division II/III
DIVISION_I_GROUP = 50

# Season types: 2 = regular season, 3 = post-season
REGULAR_SEASON = 2
POST_SEASON = 3


def fetch_schedule(
    target_date: date,
    groups: int = DIVISION_I_GROUP,
    season_type: int = REGULAR_SEASON,
) -> list[Game]:
    """
    Fetch games scheduled for a specific date.

    Args:
        target_date: The date to fetch games for
        groups: ESPN group ID (50 = D1)
        season_type: 2 = regular, 3 = postseason

    Returns:
        List of Game objects
    """
    # Format date as YYYYMMDD integer for sportsdataverse
    date_int = int(target_date.strftime("%Y%m%d"))

    logger.info(
        "Fetching schedule from ESPN",
        date=target_date.isoformat(),
        date_int=date_int,
        groups=groups,
    )

    try:
        schedule_df = espn_mbb_schedule(
            dates=date_int,
            groups=groups,
            season_type=season_type,
            return_as_pandas=False,  # Return Polars
        )
    except Exception as e:
        logger.error("Failed to fetch schedule", error=str(e))
        raise

    if schedule_df is None or len(schedule_df) == 0:
        logger.warning("No games found for date", date=target_date.isoformat())
        return []

    games = _parse_schedule_dataframe(schedule_df, target_date)
    logger.info("Schedule fetched successfully", games_count=len(games))

    return games


def _parse_schedule_dataframe(df: pl.DataFrame, target_date: date) -> list[Game]:
    """Parse sportsdataverse schedule DataFrame into Game objects."""
    games: list[Game] = []

    # Get column names for debugging
    columns = df.columns
    logger.debug("Schedule DataFrame columns", columns=columns)

    for row in df.iter_rows(named=True):
        try:
            # Extract game ID
            game_id = int(row.get("game_id") or row.get("id") or 0)
            if game_id == 0:
                continue

            # Extract team IDs
            home_team_id = int(row.get("home_id") or row.get("home_team_id") or 0)
            away_team_id = int(row.get("away_id") or row.get("away_team_id") or 0)

            if home_team_id == 0 or away_team_id == 0:
                logger.warning("Missing team IDs", game_id=game_id)
                continue

            # Extract team names
            home_name = row.get("home_name") or row.get("home_team_name") or ""
            away_name = row.get("away_name") or row.get("away_team_name") or ""

            # Extract scores (None if game not played)
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            if home_score is not None:
                home_score = int(home_score) if home_score != "" else None
            if away_score is not None:
                away_score = int(away_score) if away_score != "" else None

            # Extract game datetime
            game_datetime = None
            date_str = row.get("game_date") or row.get("date")
            if date_str:
                try:
                    if isinstance(date_str, datetime):
                        game_datetime = date_str
                    elif isinstance(date_str, str):
                        game_datetime = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            # Determine venue type
            neutral_site = bool(row.get("neutral_site") or row.get("neutral") or False)
            venue_type = VenueType.NEUTRAL if neutral_site else VenueType.HOME

            # Game status
            status = row.get("status") or row.get("status_type_name") or "scheduled"
            if isinstance(status, str):
                status = status.lower()
                if "final" in status or "complete" in status:
                    status = "final"
                elif "progress" in status or "live" in status:
                    status = "in_progress"
                else:
                    status = "scheduled"

            # Avoid persisting placeholder scores for non-final games.
            if status != "final":
                home_score = None
                away_score = None

            # Conference game detection
            conference_game = bool(row.get("conference_game") or row.get("conference_competition") or False)

            # Season extraction
            season = row.get("season") or row.get("season_year") or target_date.year
            if isinstance(season, str):
                season = int(season)

            # Determine season phase
            month = target_date.month
            if month in [11, 12] and target_date.day <= 15:
                season_phase = SeasonPhase.EARLY
            elif month in [3, 4]:
                season_phase = SeasonPhase.TOURNAMENT
            elif conference_game:
                season_phase = SeasonPhase.CONFERENCE
            else:
                season_phase = SeasonPhase.NON_CONFERENCE

            game = Game(
                game_id=game_id,
                season=season,
                game_date=target_date,
                game_datetime=game_datetime,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_team_name=home_name,
                away_team_name=away_name,
                venue_type=venue_type,
                neutral_site=neutral_site,
                venue_name=row.get("venue_name") or row.get("venue"),
                home_score=home_score,
                away_score=away_score,
                status=status,
                conference_game=conference_game,
                season_phase=season_phase,
            )
            games.append(game)

        except Exception as e:
            logger.warning("Failed to parse game row", error=str(e), row=row)
            continue

    return games


def fetch_teams(groups: int = DIVISION_I_GROUP) -> list[Team]:
    """
    Fetch all Division I teams.

    Args:
        groups: ESPN group ID (50 = D1)

    Returns:
        List of Team objects
    """
    logger.info("Fetching teams from ESPN", groups=groups)

    try:
        teams_df = espn_mbb_teams(groups=groups, return_as_pandas=False)
    except Exception as e:
        logger.error("Failed to fetch teams", error=str(e))
        raise

    if teams_df is None or len(teams_df) == 0:
        logger.warning("No teams found")
        return []

    teams = _parse_teams_dataframe(teams_df)
    logger.info("Teams fetched successfully", teams_count=len(teams))

    return teams


def _parse_teams_dataframe(df: pl.DataFrame) -> list[Team]:
    """Parse sportsdataverse teams DataFrame into Team objects."""
    teams: list[Team] = []

    for row in df.iter_rows(named=True):
        try:
            team_id = int(row.get("team_id") or row.get("id") or 0)
            if team_id == 0:
                continue

            name = row.get("team_name") or row.get("display_name") or row.get("name") or ""
            abbreviation = row.get("team_abbreviation") or row.get("abbreviation") or name[:4].upper()
            conference = row.get("conference_name") or row.get("conference") or "Unknown"

            team = Team(
                team_id=team_id,
                name=name,
                abbreviation=abbreviation,
                conference=conference,
                logo_url=row.get("logo") or row.get("team_logo"),
                color=row.get("color") or row.get("team_color"),
                alternate_color=row.get("alternate_color") or row.get("alt_color"),
            )
            teams.append(team)

        except Exception as e:
            logger.warning("Failed to parse team row", error=str(e), row=row)
            continue

    return teams


def fetch_boxscores(season: int, limit: Optional[int] = None) -> list[BoxScore]:
    """
    Fetch team box scores for a season.

    Args:
        season: Season year (e.g., 2024 for 2024-25 season)
        limit: Optional limit on number of games

    Returns:
        List of BoxScore objects
    """
    logger.info("Fetching box scores", season=season, limit=limit)

    try:
        boxscore_df = load_mbb_team_boxscore(seasons=[season], return_as_pandas=False)
    except Exception as e:
        logger.error("Failed to fetch box scores", error=str(e))
        raise

    if boxscore_df is None or len(boxscore_df) == 0:
        logger.warning("No box scores found", season=season)
        return []

    boxscores = _parse_boxscore_dataframe(boxscore_df, limit)
    logger.info("Box scores fetched successfully", boxscores_count=len(boxscores))

    return boxscores


def _parse_boxscore_dataframe(df: pl.DataFrame, limit: Optional[int] = None) -> list[BoxScore]:
    """Parse sportsdataverse box score DataFrame into BoxScore objects."""
    boxscores: list[BoxScore] = []

    if limit:
        df = df.head(limit * 2)  # Each game has 2 box scores

    for row in df.iter_rows(named=True):
        try:
            game_id = int(row.get("game_id") or 0)
            team_id = int(row.get("team_id") or 0)

            if game_id == 0 or team_id == 0:
                continue

            # Determine if home team
            is_home = bool(row.get("home_away") == "home" or row.get("is_home") or False)

            # Extract stats with safe defaults
            def safe_int(val, default=0):
                if val is None:
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default

            boxscore = BoxScore(
                game_id=game_id,
                team_id=team_id,
                is_home=is_home,
                field_goals_made=safe_int(row.get("field_goals_made") or row.get("fg")),
                field_goals_attempted=safe_int(row.get("field_goals_attempted") or row.get("fga")),
                three_pointers_made=safe_int(row.get("three_point_field_goals_made") or row.get("fg3")),
                three_pointers_attempted=safe_int(row.get("three_point_field_goals_attempted") or row.get("fga3")),
                free_throws_made=safe_int(row.get("free_throws_made") or row.get("ft")),
                free_throws_attempted=safe_int(row.get("free_throws_attempted") or row.get("fta")),
                offensive_rebounds=safe_int(row.get("offensive_rebounds") or row.get("oreb")),
                defensive_rebounds=safe_int(row.get("defensive_rebounds") or row.get("dreb")),
                turnovers=safe_int(row.get("turnovers") or row.get("to")),
                assists=safe_int(row.get("assists") or row.get("ast")),
                steals=safe_int(row.get("steals") or row.get("stl")),
                blocks=safe_int(row.get("blocks") or row.get("blk")),
                personal_fouls=safe_int(row.get("fouls") or row.get("pf")),
                points=safe_int(row.get("points") or row.get("pts")),
            )
            boxscores.append(boxscore)

        except Exception as e:
            logger.warning("Failed to parse box score row", error=str(e))
            continue

    return boxscores


def save_games_to_db(games: list[Game]) -> int:
    """
    Save games to database (upsert).

    Args:
        games: List of Game objects

    Returns:
        Number of games saved
    """
    if not games:
        return 0

    with get_connection() as conn:
        for game in games:
            conn.execute(
                """
                INSERT OR REPLACE INTO games (
                    game_id, season, game_date, game_datetime,
                    home_team_id, away_team_id, home_team_name, away_team_name,
                    venue_type, neutral_site, venue_name,
                    home_score, away_score, status,
                    conference_game, season_phase, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    game.game_id,
                    game.season,
                    game.game_date.isoformat(),
                    game.game_datetime.isoformat() if game.game_datetime else None,
                    game.home_team_id,
                    game.away_team_id,
                    game.home_team_name,
                    game.away_team_name,
                    game.venue_type.value,
                    game.neutral_site,
                    game.venue_name,
                    game.home_score,
                    game.away_score,
                    game.status,
                    game.conference_game,
                    game.season_phase.value,
                ),
            )

    logger.info("Games saved to database", count=len(games))
    return len(games)


def save_teams_to_db(teams: list[Team]) -> int:
    """
    Save teams to database (upsert).

    Args:
        teams: List of Team objects

    Returns:
        Number of teams saved
    """
    if not teams:
        return 0

    with get_connection() as conn:
        for team in teams:
            conn.execute(
                """
                INSERT OR REPLACE INTO teams (
                    team_id, name, abbreviation, conference,
                    logo_url, color, alternate_color, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    team.team_id,
                    team.name,
                    team.abbreviation,
                    team.conference,
                    team.logo_url,
                    team.color,
                    team.alternate_color,
                ),
            )

    logger.info("Teams saved to database", count=len(teams))
    return len(teams)


def save_boxscores_to_db(boxscores: list[BoxScore]) -> int:
    """
    Save box scores to database (upsert).

    Args:
        boxscores: List of BoxScore objects

    Returns:
        Number of box scores saved
    """
    if not boxscores:
        return 0

    with get_connection() as conn:
        for box in boxscores:
            conn.execute(
                """
                INSERT OR REPLACE INTO box_scores (
                    game_id, team_id, is_home,
                    field_goals_made, field_goals_attempted,
                    three_pointers_made, three_pointers_attempted,
                    free_throws_made, free_throws_attempted,
                    offensive_rebounds, defensive_rebounds,
                    turnovers, assists, steals, blocks, personal_fouls,
                    points
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    box.game_id,
                    box.team_id,
                    box.is_home,
                    box.field_goals_made,
                    box.field_goals_attempted,
                    box.three_pointers_made,
                    box.three_pointers_attempted,
                    box.free_throws_made,
                    box.free_throws_attempted,
                    box.offensive_rebounds,
                    box.defensive_rebounds,
                    box.turnovers,
                    box.assists,
                    box.steals,
                    box.blocks,
                    box.personal_fouls,
                    box.points,
                ),
            )

    logger.info("Box scores saved to database", count=len(boxscores))
    return len(boxscores)
