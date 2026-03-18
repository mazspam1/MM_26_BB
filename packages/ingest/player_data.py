"""
Player data ingestion from ESPN free API.

Fetches player rosters, game stats, and season aggregates for all D1 teams.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import duckdb
import requests
import structlog

logger = structlog.get_logger()

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_STATS_BASE = (
    "https://site.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball"
)


@dataclass
class PlayerInfo:
    player_id: str
    team_id: int
    team_name: str
    first_name: str
    last_name: str
    display_name: str
    position: str
    jersey: str
    height_inches: Optional[int]
    weight_lbs: Optional[int]
    class_year: Optional[str]


@dataclass
class PlayerSeasonStats:
    player_id: str
    season: int
    team_id: int
    games_played: int
    games_started: int
    minutes_per_game: float
    points_per_game: float
    rebounds_per_game: float
    assists_per_game: float
    steals_per_game: float
    blocks_per_game: float
    turnovers_per_game: float
    field_goal_pct: float
    three_point_pct: float
    free_throw_pct: float
    usage_rate: float
    effective_fg_pct: float
    true_shooting_pct: float


def fetch_team_roster(team_espn_id: int, team_name: str) -> list[PlayerInfo]:
    """Fetch roster for a team from ESPN API."""
    url = f"{ESPN_BASE}/teams/{team_espn_id}/roster"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch roster", team=team_name, error=str(e))
        return []

    players = []
    for athlete in data.get("athletes", []):
        # Parse height
        height = athlete.get("height")
        height_inches = None
        if height:
            # Format is typically "6-10" for feet-inches
            try:
                parts = str(height).split("-")
                if len(parts) == 2:
                    height_inches = int(parts[0]) * 12 + int(parts[1])
            except (ValueError, IndexError):
                pass

        # Parse weight
        weight = athlete.get("weight")
        weight_lbs = None
        if weight:
            try:
                weight_lbs = int(str(weight).replace(" lbs", "").replace("lbs", ""))
            except (ValueError, TypeError):
                pass

        player = PlayerInfo(
            player_id=str(athlete.get("id", "")),
            team_id=team_espn_id,
            team_name=team_name,
            first_name=athlete.get("firstName", ""),
            last_name=athlete.get("lastName", ""),
            display_name=athlete.get("displayName", ""),
            position=athlete.get("position", {}).get("abbreviation", ""),
            jersey=str(athlete.get("jersey", "")),
            height_inches=height_inches,
            weight_lbs=weight_lbs,
            class_year=athlete.get("position", {}).get("name", ""),
        )
        players.append(player)

    return players


def fetch_team_stats(team_espn_id: int) -> dict:
    """Fetch team statistics from ESPN API."""
    url = f"{ESPN_BASE}/teams/{team_espn_id}/statistics"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch team stats", team_id=team_espn_id, error=str(e))
        return {}


def fetch_player_season_stats(player_id: str, season: int = 2026) -> Optional[PlayerSeasonStats]:
    """Fetch individual player season statistics."""
    url = f"{ESPN_STATS_BASE}/athletes/{player_id}/stats"
    params = {"season": season}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch player stats", player_id=player_id, error=str(e))
        return None

    # Parse stats from response
    categories = data.get("categories", [])

    stats_dict = {}
    for cat in categories:
        cat_name = cat.get("name", "")
        for stat in cat.get("stats", []):
            stat_name = stat.get("name", "")
            stat_value = stat.get("value", 0)
            stats_dict[f"{cat_name}_{stat_name}"] = stat_value

    # Extract key stats
    try:
        games_played = int(stats_dict.get("general_gamesPlayed", 0))
        games_started = int(stats_dict.get("general_gamesStarted", 0))

        if games_played == 0:
            return None

        minutes_per_game = (
            stats_dict.get("general_avgMinutes", 0) / games_played if games_played > 0 else 0
        )
        points_per_game = stats_dict.get("offensive_avgPoints", 0)
        rebounds_per_game = stats_dict.get("general_avgRebounds", 0)
        assists_per_game = stats_dict.get("offensive_avgAssists", 0)
        steals_per_game = stats_dict.get("defensive_avgSteals", 0)
        blocks_per_game = stats_dict.get("defensive_avgBlocks", 0)
        turnovers_per_game = stats_dict.get("offensive_avgTurnovers", 0)

        fga = stats_dict.get("offensive_fieldGoalsAttempted", 0)
        fgm = stats_dict.get("offensive_fieldGoalsMade", 0)
        tpa = stats_dict.get("offensive_threePointFieldGoalsAttempted", 0)
        tpm = stats_dict.get("offensive_threePointFieldGoalsMade", 0)
        fta = stats_dict.get("offensive_freeThrowsAttempted", 0)
        ftm = stats_dict.get("offensive_freeThrowsMade", 0)

        field_goal_pct = (fgm / fga * 100) if fga > 0 else 0
        three_point_pct = (tpm / tpa * 100) if tpa > 0 else 0
        free_throw_pct = (ftm / fta * 100) if fta > 0 else 0

        # Usage rate approximation
        team_fga = 0  # Would need team totals
        team_fta = 0
        team_tov = 0
        team_minutes = 0

        # Effective FG%
        efg_pct = ((fgm + 0.5 * tpm) / fga * 100) if fga > 0 else 0

        # True Shooting %
        ts_pct = (points_per_game / (2 * (fga + 0.44 * fta)) * 100) if (fga + fta) > 0 else 0

        return PlayerSeasonStats(
            player_id=player_id,
            season=season,
            team_id=0,  # Will be filled from roster
            games_played=games_played,
            games_started=games_started,
            minutes_per_game=round(minutes_per_game, 1),
            points_per_game=round(points_per_game, 1),
            rebounds_per_game=round(rebounds_per_game, 1),
            assists_per_game=round(assists_per_game, 1),
            steals_per_game=round(steals_per_game, 1),
            blocks_per_game=round(blocks_per_game, 1),
            turnovers_per_game=round(turnovers_per_game, 1),
            field_goal_pct=round(field_goal_pct, 1),
            three_point_pct=round(three_point_pct, 1),
            free_throw_pct=round(free_throw_pct, 1),
            usage_rate=0.0,  # Complex calculation
            effective_fg_pct=round(efg_pct, 1),
            true_shooting_pct=round(ts_pct, 1),
        )
    except (KeyError, TypeError, ZeroDivisionError) as e:
        logger.warning("Failed to parse player stats", player_id=player_id, error=str(e))
        return None


def calculate_team_player_summary(
    team_id: int,
    team_name: str,
    players: list[PlayerInfo],
    player_stats: list[PlayerSeasonStats],
) -> dict:
    """Calculate team-level player composition metrics."""
    if not players or not player_stats:
        return {}

    # Height stats
    heights = [p.height_inches for p in players if p.height_inches]
    avg_height = sum(heights) / len(heights) if heights else 0

    # Weight stats
    weights = [p.weight_lbs for p in players if p.weight_lbs]
    avg_weight = sum(weights) / len(weights) if weights else 0

    # Top scorers
    sorted_by_pts = sorted(player_stats, key=lambda x: x.points_per_game, reverse=True)
    top_scorer = sorted_by_pts[0] if sorted_by_pts else None
    top_rebounder = max(player_stats, key=lambda x: x.rebounds_per_game) if player_stats else None
    top_assists = max(player_stats, key=lambda x: x.assists_per_game) if player_stats else None

    # Three-point heavy?
    total_3pa = sum(
        s.three_point_pct * s.games_played for s in player_stats if s.three_point_pct > 0
    )
    three_point_heavy = any(s.three_point_pct > 35 and s.games_played > 10 for s in player_stats)

    # Interior presence
    interior_presence = any(
        s.blocks_per_game > 1.5 and s.rebounds_per_game > 6 for s in player_stats
    )

    # Depth rating (number of players with >15 min/game)
    deep_rotation = sum(1 for s in player_stats if s.minutes_per_game > 15)
    depth_rating = min(1.0, deep_rotation / 8)  # Normalize to 0-1

    return {
        "team_id": team_id,
        "team_name": team_name,
        "season": 2026,
        "n_players": len(players),
        "avg_height_inches": round(avg_height, 1),
        "avg_weight_lbs": round(avg_weight, 1),
        "top_scorer_id": top_scorer.player_id if top_scorer else None,
        "top_scorer_ppg": round(top_scorer.points_per_game, 1) if top_scorer else 0,
        "top_rebounder_id": top_rebounder.player_id if top_rebounder else None,
        "top_rebounder_rpg": round(top_rebounder.rebounds_per_game, 1) if top_rebounder else 0,
        "top_assists_id": top_assists.player_id if top_assists else None,
        "top_assists_apg": round(top_assists.assists_per_game, 1) if top_assists else 0,
        "three_point_heavy": three_point_heavy,
        "interior_presence": interior_presence,
        "depth_rating": round(depth_rating, 2),
    }


def ingest_player_data(
    db_path: str,
    team_mapping: dict[str, int],
    season: int = 2026,
    delay: float = 0.1,
) -> dict:
    """
    Ingest player data for all teams.

    Args:
        db_path: Path to DuckDB database
        team_mapping: Dict of team_name -> ESPN team_id
        season: Season year
        delay: Delay between API calls to avoid rate limiting

    Returns:
        Summary of ingestion
    """
    logger.info("Starting player data ingestion", teams=len(team_mapping))

    conn = duckdb.connect(db_path)

    # Create tables
    with open("data/player_schema.sql", "r") as f:
        schema_sql = f.read()
        conn.execute(schema_sql)

    summary = {
        "teams_processed": 0,
        "players_found": 0,
        "stats_fetched": 0,
        "errors": 0,
    }

    all_players = []
    all_stats = []

    for team_name, espn_id in team_mapping.items():
        try:
            logger.info("Processing team", team=team_name)

            # Fetch roster
            players = fetch_team_roster(espn_id, team_name)
            time.sleep(delay)

            if not players:
                summary["errors"] += 1
                continue

            summary["players_found"] += len(players)
            all_players.extend(players)

            # Fetch stats for key players (top 10 by minutes)
            for player in players[:10]:  # Limit to top players to reduce API calls
                stats = fetch_player_season_stats(player.player_id, season)
                time.sleep(delay)

                if stats:
                    stats.team_id = espn_id
                    all_stats.append(stats)
                    summary["stats_fetched"] += 1

            summary["teams_processed"] += 1

        except Exception as e:
            logger.error("Failed to process team", team=team_name, error=str(e))
            summary["errors"] += 1

    # Save to database
    if all_players:
        for p in all_players:
            conn.execute(
                """
                INSERT OR REPLACE INTO players 
                (player_id, team_id, team_name, first_name, last_name, display_name, position, jersey, height_inches, weight_lbs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    p.player_id,
                    p.team_id,
                    p.team_name,
                    p.first_name,
                    p.last_name,
                    p.display_name,
                    p.position,
                    p.jersey,
                    p.height_inches,
                    p.weight_lbs,
                ],
            )

    if all_stats:
        for s in all_stats:
            conn.execute(
                """
                INSERT OR REPLACE INTO player_season_stats
                (player_id, season, team_id, games_played, games_started, minutes_per_game, points_per_game, 
                 rebounds_per_game, assists_per_game, steals_per_game, blocks_per_game, turnovers_per_game,
                 field_goal_pct, three_point_pct, free_throw_pct, effective_fg_pct, true_shooting_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    s.player_id,
                    s.season,
                    s.team_id,
                    s.games_played,
                    s.games_started,
                    s.minutes_per_game,
                    s.points_per_game,
                    s.rebounds_per_game,
                    s.assists_per_game,
                    s.steals_per_game,
                    s.blocks_per_game,
                    s.turnovers_per_game,
                    s.field_goal_pct,
                    s.three_point_pct,
                    s.free_throw_pct,
                    s.effective_fg_pct,
                    s.true_shooting_pct,
                ],
            )

    conn.close()

    logger.info("Player data ingestion complete", **summary)
    return summary


# ESPN team IDs for tournament teams (will be populated from team data)
TOURNAMENT_TEAM_IDS = {
    "Duke Blue Devils": 150,
    "UConn Huskies": 41,
    "Michigan State Spartans": 127,
    "Kansas Jayhawks": 2305,
    "St. John's Red Storm": 2599,
    "Louisville Cardinals": 97,
    "UCLA Bruins": 26,
    "Ohio State Buckeyes": 194,
    "TCU Horned Frogs": 2628,
    "UCF Knights": 2116,
    "South Florida Bulls": 58,
    "Northern Iowa Panthers": 2460,
    "North Dakota State Bison": 2449,
    "Furman Paladins": 231,
    "Siena Saints": 2561,
    "Arizona Wildcats": 12,
    "Purdue Boilermakers": 2509,
    "Gonzaga Bulldogs": 2250,
    "Arkansas Razorbacks": 8,
    "Wisconsin Badgers": 275,
    "BYU Cougars": 252,
    "Miami Hurricanes": 2390,
    "Villanova Wildcats": 222,
    "Utah State Aggies": 328,
    "Missouri Tigers": 142,
    "Texas Longhorns": 251,
    "High Point Panthers": 2272,
    "Hawai'i Rainbow Warriors": 62,
    "Kennesaw State Owls": 338,
    "Queens University Royals": 2511,
    "Long Island University Sharks": 112358,
    "Florida Gators": 57,
    "Houston Cougars": 248,
    "Illinois Fighting Illini": 356,
    "Nebraska Cornhuskers": 158,
    "Vanderbilt Commodores": 238,
    "North Carolina Tar Heels": 153,
    "Saint Mary's Gaels": 2608,
    "Clemson Tigers": 228,
    "Iowa Hawkeyes": 2294,
    "Texas A&M Aggies": 245,
    "VCU Rams": 2670,
    "McNeese Cowboys": 2377,
    "Troy Trojans": 2653,
    "Pennsylvania Quakers": 219,
    "Idaho Vandals": 70,
    "Lehigh Mountain Hawks": 2329,
    "Michigan Wolverines": 130,
    "Iowa State Cyclones": 66,
    "Virginia Cavaliers": 258,
    "Alabama Crimson Tide": 333,
    "Texas Tech Red Raiders": 2641,
    "Tennessee Volunteers": 2633,
    "Kentucky Wildcats": 96,
    "Georgia Bulldogs": 61,
    "Saint Louis Billikens": 139,
    "Santa Clara Broncos": 2541,
    "SMU Mustangs": 2567,
    "Akron Zips": 2006,
    "Hofstra Pride": 2275,
    "Wright State Raiders": 2750,
    "Tennessee State Tigers": 2634,
    "UMBC Retrievers": 2378,
    "California Baptist Lancers": 2856,
    "Howard Bison": 47,
    "Miami (OH) RedHawks": 193,
    "NC State Wolfpack": 152,
    "Prairie View A&M Panthers": 2504,
}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    db_path = str(project_root / "data" / "cbb_lines.duckdb")

    print("Starting player data ingestion...")
    summary = ingest_player_data(db_path, TOURNAMENT_TEAM_IDS, delay=0.15)
    print(f"\nSummary: {summary}")
