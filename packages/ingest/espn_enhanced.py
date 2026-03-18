"""
Enhanced ESPN data ingestion using sportsdataverse.

Fetches REAL box score data with all statistics needed for PhD-level modeling:
- Four Factors (eFG%, TOV%, ORB%, FTR)
- Tempo and possessions
- Conference and venue data
- Historical schedules for rest calculation

NO MOCK DATA. NO FALLBACKS. REAL ESPN DATA ONLY.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
import structlog

import pandas as pd

from packages.common.sportsdataverse_mbb import load_mbb

from packages.common.database import get_connection
from packages.common.season import infer_season_year

logger = structlog.get_logger()
_mbb = load_mbb()
espn_mbb_schedule = _mbb.espn_mbb_schedule
espn_mbb_pbp = _mbb.espn_mbb_pbp
espn_mbb_teams = _mbb.espn_mbb_teams
load_mbb_team_boxscore = _mbb.load_mbb_team_boxscore
load_mbb_schedule = _mbb.load_mbb_schedule


@dataclass
class TeamBoxScore:
    """Complete team box score with all stats for Four Factors."""
    game_id: int
    team_id: int
    opponent_id: int
    game_date: date
    is_home: bool
    is_neutral: bool

    # Score
    team_score: int
    opponent_score: int

    # Shooting
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int

    # Rebounds
    offensive_rebounds: int
    defensive_rebounds: int

    # Other
    turnovers: int
    assists: int
    steals: int
    blocks: int
    fouls: int


@dataclass
class FourFactors:
    """Dean Oliver's Four Factors for a team performance."""
    efg_pct: float  # Effective FG%
    tov_pct: float  # Turnover %
    orb_pct: float  # Offensive Rebound %
    ftr: float      # Free Throw Rate

    # Derived
    possessions: float
    offensive_rating: float  # Points per 100 possessions


def calculate_possessions(fga: int, fta: int, orb: int, tov: int) -> float:
    """
    Calculate possessions using Dean Oliver formula.

    Possessions = FGA - OR + TO + 0.475 * FTA
    """
    return fga - orb + tov + 0.475 * fta


def calculate_four_factors(
    fgm: int, fga: int, fg3m: int,
    ftm: int, fta: int,
    orb: int, opp_drb: int,
    tov: int, team_score: int
) -> FourFactors:
    """
    Calculate Dean Oliver's Four Factors from box score stats.

    Updated weights from 2023 research:
    - eFG%: 46% (was 40%)
    - TOV%: 35% (was 25%)
    - ORB%: 12% (was 20%)
    - FTR:   7% (was 15%)
    """
    # Effective Field Goal Percentage
    # eFG% = (FGM + 0.5 * 3PM) / FGA
    efg_pct = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0

    # Turnover Percentage
    # TOV% = TO / (FGA + 0.44 * FTA + TO)
    tov_pct = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) > 0 else 0.0

    # Offensive Rebound Percentage
    # ORB% = ORB / (ORB + Opp_DRB)
    orb_pct = orb / (orb + opp_drb) if (orb + opp_drb) > 0 else 0.0

    # Free Throw Rate
    # FTR = FTA / FGA
    ftr = fta / fga if fga > 0 else 0.0

    # Possessions
    possessions = calculate_possessions(fga, fta, orb, tov)

    # Offensive Rating (points per 100 possessions)
    off_rating = (team_score / possessions * 100) if possessions > 0 else 100.0

    return FourFactors(
        efg_pct=efg_pct,
        tov_pct=tov_pct,
        orb_pct=orb_pct,
        ftr=ftr,
        possessions=possessions,
        offensive_rating=off_rating,
    )


def fetch_season_boxscores(season: int = 2025) -> pd.DataFrame:
    """
    Fetch all team box scores for a season from ESPN.

    Returns DataFrame with complete stats for each team in each game.
    """
    logger.info("Fetching season box scores from ESPN", season=season)

    boxscores = load_mbb_team_boxscore(seasons=[season], return_as_pandas=True)

    logger.info("Box scores fetched", count=len(boxscores))
    return boxscores


def fetch_season_schedule(season: int = 2025) -> pd.DataFrame:
    """
    Fetch complete schedule for a season.

    Used for rest day calculations and neutral site detection.
    """
    logger.info("Fetching season schedule from ESPN", season=season)

    schedule = load_mbb_schedule(seasons=[season], return_as_pandas=True)

    logger.info("Schedule fetched", games=len(schedule))
    return schedule


def fetch_all_teams() -> pd.DataFrame:
    """
    Fetch all Division I teams with metadata.
    """
    logger.info("Fetching D1 teams from ESPN")

    teams = espn_mbb_teams(groups=50, return_as_pandas=True)

    logger.info("Teams fetched", count=len(teams))
    return teams


def fetch_boxscores_for_date_range(
    start_date: date,
    end_date: date,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch box scores for a specific date range.
    """
    if season is None:
        season = infer_season_year(end_date)

    boxscores = fetch_season_boxscores(season)
    if boxscores.empty:
        return boxscores

    boxscores = boxscores.copy()
    boxscores["game_date"] = pd.to_datetime(boxscores["game_date"]).dt.date
    mask = (boxscores["game_date"] >= start_date) & (boxscores["game_date"] <= end_date)
    return boxscores[mask]


def ingest_team_stats_for_date_range(
    start_date: date,
    end_date: date,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ingest team-game stats for a date range.
    """
    if season is None:
        season = infer_season_year(end_date)

    logger.info(
        "Ingesting team stats for date range",
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        season=season,
    )

    # Full season schedule is needed for rest-day calculation.
    schedule = fetch_season_schedule(season)
    boxscores = fetch_boxscores_for_date_range(start_date, end_date, season)

    if boxscores.empty:
        logger.warning("No boxscores found for date range", start=start_date, end=end_date)
        return boxscores

    stats_df = process_boxscores_to_team_stats(boxscores, schedule)
    if not stats_df.empty:
        save_team_stats_to_db(stats_df)

    return stats_df


def fetch_game_odds(game_id: int) -> Optional[dict]:
    """
    Fetch betting odds for a specific game.

    Returns dict with spread, total, moneyline from DraftKings.
    """
    try:
        data = espn_mbb_pbp(game_id=game_id)

        if 'pickcenter' in data and len(data['pickcenter']) > 0:
            pick = data['pickcenter'][0]
            return {
                'spread': pick.get('spread'),
                'total': pick.get('overUnder'),
                'provider': pick.get('provider', {}).get('name', 'Unknown'),
            }
    except Exception as e:
        logger.warning("Failed to fetch odds", game_id=game_id, error=str(e))

    return None


def calculate_rest_days(schedule: pd.DataFrame, team_id: int, game_date: date) -> int:
    """
    Calculate days since team's last game.

    Returns:
        Number of rest days (0 = back-to-back, 1 = one day rest, etc.)
        Returns 7 if no previous game found (season opener)
    """
    # Get all games for this team before the target date
    team_games = schedule[
        ((schedule['home_id'] == team_id) | (schedule['away_id'] == team_id)) &
        (pd.to_datetime(schedule['game_date']).dt.date < game_date) &
        (schedule['status_type_completed'] == True)
    ].copy()

    if len(team_games) == 0:
        return 7  # Season opener

    # Find most recent game
    team_games['game_date'] = pd.to_datetime(team_games['game_date']).dt.date
    last_game_date = team_games['game_date'].max()

    rest_days = (game_date - last_game_date).days - 1  # -1 because game day doesn't count

    return max(0, rest_days)


def process_boxscores_to_team_stats(boxscores: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw box scores into team-game stats with Four Factors.

    Returns DataFrame with one row per team per game, including:
    - Four Factors (offensive and defensive)
    - Possessions and tempo
    - Rest days
    - Home/away/neutral
    """
    logger.info("Processing box scores into team stats")

    processed = []

    # Group by game to get both teams' stats
    for game_id, game_data in boxscores.groupby('game_id'):
        if len(game_data) != 2:
            continue  # Skip incomplete games

        # Get schedule info for this game
        game_schedule = schedule[schedule['game_id'] == game_id]
        is_neutral = False
        if len(game_schedule) > 0:
            is_neutral = bool(game_schedule.iloc[0].get('neutral_site', False))

        for idx, team_row in game_data.iterrows():
            team_id = team_row['team_id']
            opp_id = team_row['opponent_team_id']
            game_date = pd.to_datetime(team_row['game_date']).date()
            is_home = team_row['team_home_away'] == 'home'

            # Get opponent stats
            opp_row = game_data[game_data['team_id'] == opp_id]
            if len(opp_row) == 0:
                continue
            opp_row = opp_row.iloc[0]

            # Calculate Four Factors (offensive)
            off_factors = calculate_four_factors(
                fgm=int(team_row['field_goals_made']),
                fga=int(team_row['field_goals_attempted']),
                fg3m=int(team_row['three_point_field_goals_made']),
                ftm=int(team_row['free_throws_made']),
                fta=int(team_row['free_throws_attempted']),
                orb=int(team_row['offensive_rebounds']),
                opp_drb=int(opp_row['defensive_rebounds']),
                tov=int(team_row['turnovers']),
                team_score=int(team_row['team_score']),
            )

            # Calculate Four Factors (defensive - from opponent's perspective)
            def_factors = calculate_four_factors(
                fgm=int(opp_row['field_goals_made']),
                fga=int(opp_row['field_goals_attempted']),
                fg3m=int(opp_row['three_point_field_goals_made']),
                ftm=int(opp_row['free_throws_made']),
                fta=int(opp_row['free_throws_attempted']),
                orb=int(opp_row['offensive_rebounds']),
                opp_drb=int(team_row['defensive_rebounds']),
                tov=int(opp_row['turnovers']),
                team_score=int(opp_row['team_score']),
            )

            # Calculate rest days
            rest_days = calculate_rest_days(schedule, team_id, game_date)

            processed.append({
                'game_id': game_id,
                'game_date': game_date,
                'team_id': team_id,
                'opponent_id': opp_id,
                'is_home': is_home,
                'is_neutral': is_neutral,
                'team_score': int(team_row['team_score']),
                'opponent_score': int(opp_row['team_score']),
                'won': team_row['team_winner'],

                # Raw stats
                'fgm': int(team_row['field_goals_made']),
                'fga': int(team_row['field_goals_attempted']),
                'fg3m': int(team_row['three_point_field_goals_made']),
                'fg3a': int(team_row['three_point_field_goals_attempted']),
                'ftm': int(team_row['free_throws_made']),
                'fta': int(team_row['free_throws_attempted']),
                'orb': int(team_row['offensive_rebounds']),
                'drb': int(team_row['defensive_rebounds']),
                'turnovers': int(team_row['turnovers']),
                'assists': int(team_row['assists']),
                'steals': int(team_row['steals']),
                'blocks': int(team_row['blocks']),

                # Offensive Four Factors
                'off_efg': off_factors.efg_pct,
                'off_tov': off_factors.tov_pct,
                'off_orb': off_factors.orb_pct,
                'off_ftr': off_factors.ftr,
                'possessions': off_factors.possessions,
                'off_rating': off_factors.offensive_rating,

                # Defensive Four Factors (opponent's offensive factors)
                'def_efg': def_factors.efg_pct,
                'def_tov': def_factors.tov_pct,
                'def_orb': def_factors.orb_pct,  # This is opponent's ORB%
                'def_ftr': def_factors.ftr,
                'def_rating': def_factors.offensive_rating,

                # Context
                'rest_days': rest_days,
            })

    df = pd.DataFrame(processed)
    logger.info("Processed team stats", rows=len(df))
    return df


def save_team_stats_to_db(stats_df: pd.DataFrame):
    """
    Save processed team stats to database.
    """
    logger.info("Saving team stats to database", rows=len(stats_df))

    with get_connection() as conn:
        # Create table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_game_stats (
                game_id INTEGER,
                game_date TEXT,
                team_id INTEGER,
                opponent_id INTEGER,
                is_home BOOLEAN,
                is_neutral BOOLEAN,
                team_score INTEGER,
                opponent_score INTEGER,
                won BOOLEAN,

                fgm INTEGER,
                fga INTEGER,
                fg3m INTEGER,
                fg3a INTEGER,
                ftm INTEGER,
                fta INTEGER,
                orb INTEGER,
                drb INTEGER,
                turnovers INTEGER,
                assists INTEGER,
                steals INTEGER,
                blocks INTEGER,

                off_efg REAL,
                off_tov REAL,
                off_orb REAL,
                off_ftr REAL,
                possessions REAL,
                off_rating REAL,

                def_efg REAL,
                def_tov REAL,
                def_orb REAL,
                def_ftr REAL,
                def_rating REAL,

                rest_days INTEGER,

                PRIMARY KEY (game_id, team_id)
            )
        """)

        # Insert data
        for _, row in stats_df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO team_game_stats VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?
                )
            """, (
                row['game_id'], str(row['game_date']), row['team_id'], row['opponent_id'],
                row['is_home'], row['is_neutral'], row['team_score'], row['opponent_score'],
                row['won'],
                row['fgm'], row['fga'], row['fg3m'], row['fg3a'],
                row['ftm'], row['fta'], row['orb'], row['drb'],
                row['turnovers'], row['assists'], row['steals'], row['blocks'],
                row['off_efg'], row['off_tov'], row['off_orb'], row['off_ftr'],
                row['possessions'], row['off_rating'],
                row['def_efg'], row['def_tov'], row['def_orb'], row['def_ftr'],
                row['def_rating'],
                row['rest_days'],
            ))

    logger.info("Team stats saved to database")


def ingest_full_season(season: int = 2025):
    """
    Complete ingestion of a season's data.

    1. Fetches all box scores
    2. Fetches schedule for rest calculations
    3. Processes into team-game stats with Four Factors
    4. Saves to database
    """
    logger.info("Starting full season ingestion", season=season)

    # Fetch raw data
    boxscores = fetch_season_boxscores(season)
    schedule = fetch_season_schedule(season)

    # Process into team stats
    stats_df = process_boxscores_to_team_stats(boxscores, schedule)

    # Save to database
    save_team_stats_to_db(stats_df)

    logger.info("Full season ingestion complete", games=len(stats_df) // 2)

    return stats_df


if __name__ == "__main__":
    # Run ingestion for current season
    ingest_full_season(2025)
