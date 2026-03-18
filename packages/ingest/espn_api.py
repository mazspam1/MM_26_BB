"""
Direct ESPN API connector for NCAA Men's Basketball.

Bypasses sportsdataverse to avoid xgboost compatibility issues.
Uses the same ESPN endpoints directly via HTTP.
"""

from datetime import date, datetime
from typing import Optional

import httpx
import structlog

from packages.common.database import get_connection
from packages.common.schemas import BoxScore, Game, Team, VenueType, SeasonPhase
from packages.features.conference_hca import get_conference_name

logger = structlog.get_logger()

# ESPN API endpoints
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_CORE_URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"

# Division I group ID
DIVISION_I_GROUP = 50


async def fetch_teams_async(groups: int = DIVISION_I_GROUP) -> list[Team]:
    """
    Fetch all Division I teams from ESPN API.

    Args:
        groups: ESPN group ID (50 = D1)

    Returns:
        List of Team objects
    """
    url = f"{ESPN_BASE_URL}/teams"
    params = {"groups": groups, "limit": 500}

    logger.info("Fetching teams from ESPN", url=url)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    teams = []
    for team_data in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team_info = team_data.get("team", {})
        try:
            team = Team(
                team_id=int(team_info.get("id", 0)),
                name=team_info.get("displayName", ""),
                abbreviation=team_info.get("abbreviation", "")[:10],
                conference=team_info.get("conferenceId", "Unknown"),
                logo_url=team_info.get("logos", [{}])[0].get("href") if team_info.get("logos") else None,
                color=team_info.get("color"),
                alternate_color=team_info.get("alternateColor"),
            )
            teams.append(team)
        except Exception as e:
            logger.warning("Failed to parse team", error=str(e), team=team_info)

    logger.info("Teams fetched", count=len(teams))
    return teams


def fetch_teams(groups: int = DIVISION_I_GROUP) -> list[Team]:
    """
    Fetch all Division I teams (sync wrapper).

    Args:
        groups: ESPN group ID (50 = D1)

    Returns:
        List of Team objects
    """
    import asyncio
    return asyncio.run(fetch_teams_async(groups))


async def fetch_schedule_async(
    target_date: date,
    groups: int = DIVISION_I_GROUP,
) -> list[Game]:
    """
    Fetch games scheduled for a specific date.

    Args:
        target_date: The date to fetch games for
        groups: ESPN group ID (50 = D1)

    Returns:
        List of Game objects
    """
    date_str = target_date.strftime("%Y%m%d")
    url = f"{ESPN_BASE_URL}/scoreboard"
    params = {"dates": date_str, "groups": groups, "limit": 500}

    logger.info("Fetching schedule from ESPN", date=date_str, url=url)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    games = []
    for event in data.get("events", []):
        try:
            game = _parse_event(event, target_date)
            if game:
                games.append(game)
        except Exception as e:
            logger.warning("Failed to parse event", error=str(e), event_id=event.get("id"))

    logger.info("Schedule fetched", games_count=len(games))
    return games


def fetch_schedule(
    target_date: date,
    groups: int = DIVISION_I_GROUP,
) -> list[Game]:
    """
    Fetch games for a specific date (sync wrapper).

    Args:
        target_date: The date to fetch games for
        groups: ESPN group ID (50 = D1)

    Returns:
        List of Game objects
    """
    import asyncio
    return asyncio.run(fetch_schedule_async(target_date, groups))


async def fetch_team_conferences_async(
    target_date: date,
    groups: int = DIVISION_I_GROUP,
) -> dict[int, tuple[int, Optional[str]]]:
    """
    Fetch team conference IDs for a specific date from ESPN scoreboard.

    Returns:
        Dict of team_id -> (conference_id, conference_name)
    """
    date_str = target_date.strftime("%Y%m%d")
    url = f"{ESPN_BASE_URL}/scoreboard"
    params = {"dates": date_str, "groups": groups, "limit": 500}

    logger.info("Fetching team conferences from ESPN", date=date_str, url=url)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    team_confs: dict[int, tuple[int, Optional[str]]] = {}
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        for competitor in competition.get("competitors", []):
            team = competitor.get("team", {})
            team_id = int(team.get("id", 0)) if team.get("id") else 0
            if team_id <= 0:
                continue

            conf_id_raw = team.get("conferenceId")
            if conf_id_raw is None:
                continue

            try:
                conf_id = int(conf_id_raw)
            except (TypeError, ValueError):
                continue

            conf_name = get_conference_name(conf_id)
            team_confs[team_id] = (conf_id, conf_name)

    logger.info("Team conferences fetched", count=len(team_confs))
    return team_confs


def fetch_team_conferences(
    target_date: date,
    groups: int = DIVISION_I_GROUP,
) -> dict[int, tuple[int, Optional[str]]]:
    """
    Fetch team conference IDs (sync wrapper).

    Returns:
        Dict of team_id -> (conference_id, conference_name)
    """
    import asyncio
    return asyncio.run(fetch_team_conferences_async(target_date, groups))


def _parse_event(event: dict, target_date: date) -> Optional[Game]:
    """Parse ESPN event data into Game object."""
    game_id = int(event.get("id", 0))
    if game_id == 0:
        return None

    competition = event.get("competitions", [{}])[0]
    competitors = competition.get("competitors", [])

    if len(competitors) != 2:
        return None

    # Find home and away teams
    home_team = None
    away_team = None
    for comp in competitors:
        if comp.get("homeAway") == "home":
            home_team = comp
        else:
            away_team = comp

    if not home_team or not away_team:
        return None

    home_team_id = int(home_team.get("id", 0))
    away_team_id = int(away_team.get("id", 0))

    if home_team_id == 0 or away_team_id == 0:
        return None

    # Get scores
    home_score = None
    away_score = None
    if home_team.get("score"):
        try:
            home_score = int(home_team["score"])
        except (ValueError, TypeError):
            pass
    if away_team.get("score"):
        try:
            away_score = int(away_team["score"])
        except (ValueError, TypeError):
            pass

    # Parse game datetime
    game_datetime = None
    date_str = event.get("date")
    if date_str:
        try:
            game_datetime = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    # Determine status
    status_type = event.get("status", {}).get("type", {})
    status_name = status_type.get("name", "STATUS_SCHEDULED").lower()
    if "final" in status_name or "complete" in status_name:
        status = "final"
    elif "progress" in status_name or "live" in status_name:
        status = "in_progress"
    else:
        status = "scheduled"

    # Avoid persisting placeholder scores for non-final games.
    if status != "final":
        home_score = None
        away_score = None

    # Neutral site
    neutral_site = competition.get("neutralSite", False)
    venue_type = VenueType.NEUTRAL if neutral_site else VenueType.HOME

    # Conference game
    conference_game = competition.get("conferenceCompetition", False)

    # Season
    season = event.get("season", {}).get("year", target_date.year)

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

    return Game(
        game_id=game_id,
        season=season,
        game_date=target_date,
        game_datetime=game_datetime,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_team.get("team", {}).get("displayName"),
        away_team_name=away_team.get("team", {}).get("displayName"),
        venue_type=venue_type,
        neutral_site=neutral_site,
        venue_name=competition.get("venue", {}).get("fullName"),
        home_score=home_score,
        away_score=away_score,
        status=status,
        conference_game=conference_game,
        season_phase=season_phase,
    )


def save_games_to_db(games: list[Game], skip_fk_errors: bool = True) -> int:
    """
    Save games to database (upsert).

    Args:
        games: List of Game objects
        skip_fk_errors: Skip games with FK violations (non-D1 opponents)

    Returns:
        Number of games saved
    """
    if not games:
        return 0

    saved_count = 0
    with get_connection() as conn:
        for game in games:
            try:
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
                saved_count += 1
            except Exception as e:
                if skip_fk_errors and "foreign key" in str(e).lower():
                    logger.debug("Skipping game with non-D1 opponent", game_id=game.game_id)
                else:
                    raise

    logger.info("Games saved to database", count=saved_count, skipped=len(games) - saved_count)
    return saved_count


def save_team_conferences_to_db(team_confs: dict[int, tuple[int, Optional[str]]]) -> int:
    """
    Save team conference IDs to database (upsert).

    Args:
        team_confs: Dict of team_id -> (conference_id, conference_name)

    Returns:
        Number of teams saved
    """
    if not team_confs:
        return 0

    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS team_conference_ids (
                team_id INTEGER PRIMARY KEY,
                conference_id INTEGER,
                conference_name VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_team_conf_team FOREIGN KEY (team_id) REFERENCES teams(team_id)
            )
            """
        )

        for team_id, (conf_id, conf_name) in team_confs.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO team_conference_ids (
                    team_id, conference_id, conference_name, updated_at
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (team_id, conf_id, conf_name),
            )

    logger.info("Team conferences saved to database", count=len(team_confs))
    return len(team_confs)


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
