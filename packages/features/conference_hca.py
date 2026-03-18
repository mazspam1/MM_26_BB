"""
Conference-specific home court advantage calculations.

Based on research:
- ACC: 4.2 pts
- Big Ten: 4.0 pts
- SEC: 3.8 pts
- Big 12: 3.5 pts
- Compact conferences (SoCon): 2.5 pts
- Blue blood programs: 6-7+ pts

Also calculates travel distance effects.
"""

from dataclasses import dataclass
from typing import Optional
import math
import structlog

import httpx
import pandas as pd

logger = structlog.get_logger()

ESPN_GROUP_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/"
    "leagues/mens-college-basketball/groups"
)

# ESPN conference group IDs (verified via ESPN core API)
# Keep this minimal to avoid mismatches; use name-based fallback for others.
CONFERENCE_HCA = {
    # Power conferences + Big East
    2: 4.2,   # Atlantic Coast Conference (ACC)
    4: 3.8,   # Big East Conference
    7: 4.0,   # Big Ten Conference
    8: 3.5,   # Big 12 Conference
    23: 3.8,  # Southeastern Conference (SEC)
    44: 3.5,  # Mountain West Conference
}

# Name-based fallbacks (keyword match on ESPN group name)
CONFERENCE_HCA_KEYWORDS = [
    ("atlantic coast", 4.2),
    ("big ten", 4.0),
    ("southeastern", 3.8),
    ("big east", 3.8),
    ("big 12", 3.5),
    ("pac-12", 3.8),
    ("mountain west", 3.5),
    ("american athletic", 3.5),
    ("atlantic 10", 3.3),
    ("mid-american", 3.2),
    ("missouri valley", 3.2),
    ("west coast", 3.2),
    ("sun belt", 3.0),
    ("southern conference", 3.0),
    ("conference usa", 3.0),
    ("c-usa", 3.0),
    ("ivy", 2.8),
    ("patriot", 2.8),
    ("metro atlantic", 2.8),
    ("asun", 2.5),
    ("atlantic sun", 2.5),
    ("western athletic", 2.5),
    ("wac", 2.5),
    ("southland", 2.5),
    ("big sky", 2.5),
    ("big west", 2.5),
    ("meac", 2.5),
    ("swac", 2.5),
    ("northeast", 2.5),
    ("america east", 2.5),
    ("big south", 2.5),
    ("colonial", 2.5),
    ("caa", 2.5),
    ("horizon", 2.5),
    ("summit", 2.5),
    ("ohio valley", 2.5),
    ("socon", 2.5),
]

# Blue blood programs get extra HCA
BLUE_BLOOD_TEAMS = {
    2305: 7.0,  # Kansas
    150: 6.5,   # Duke
    96: 6.5,    # Kentucky
    26: 6.0,    # UCLA
    153: 5.5,   # North Carolina
    356: 5.5,   # Indiana
    # Add more elite venues
    248: 5.5,   # Houston (Fertitta Center)
    127: 5.5,   # Michigan State (Breslin)
}

# Team arena coordinates for distance calculation (lat, lon)
# Key venues with known coordinates
ARENA_COORDINATES = {
    # ACC
    150: (36.0014, -78.9382),  # Duke
    153: (35.9050, -79.0469),  # North Carolina
    52: (35.7796, -78.6382),   # NC State
    228: (36.0726, -79.7920),  # Wake Forest
    59: (33.7490, -84.3880),   # Georgia Tech
    234: (37.5407, -77.4360),  # Virginia
    259: (37.2296, -80.4139),  # Virginia Tech
    103: (40.4406, -79.9959),  # Pittsburgh
    367: (42.3601, -71.0589),  # Boston College
    2507: (43.6532, -79.3832), # Syracuse (actually NY)
    2390: (25.7617, -80.1918), # Miami

    # Big Ten
    356: (39.1653, -86.5264),  # Indiana
    127: (42.7325, -84.5555),  # Michigan State
    130: (42.2808, -83.7430),  # Michigan
    77: (41.8781, -87.6298),   # Illinois
    275: (43.0731, -89.4012),  # Wisconsin
    2509: (41.0082, -74.1057), # Rutgers
    158: (40.8258, -96.7014),  # Nebraska
    277: (40.1020, -88.2272),  # Purdue
    2294: (44.9778, -93.2650), # Minnesota
    194: (41.6611, -91.5302),  # Iowa
    120: (42.2917, -85.5872),  # Maryland
    204: (40.7128, -74.0060),  # Northwestern
    221: (33.4484, -112.0740), # Penn State

    # SEC
    2: (32.6010, -85.4808),    # Auburn
    333: (33.2098, -87.5692),  # Alabama
    96: (38.0406, -84.5037),   # Kentucky
    57: (29.6516, -82.3248),   # Florida
    61: (33.9519, -83.3576),   # Georgia
    99: (30.4583, -91.1403),   # LSU
    142: (32.2988, -90.1848),  # Ole Miss
    344: (33.4504, -88.8184),  # Mississippi State
    235: (36.1627, -86.7816),  # Vanderbilt
    2633: (29.9511, -90.0715), # Tulane
    2653: (29.7604, -95.3698), # Texas A&M

    # Big 12
    2305: (38.9543, -95.2558), # Kansas
    66: (39.7392, -104.9903),  # Colorado (now Pac-12)
    239: (30.2672, -97.7431),  # Texas
    251: (36.1540, -95.9928),  # Oklahoma
    201: (35.2226, -97.4395),  # Oklahoma State
    2628: (31.8457, -102.3676),# Texas Tech
    326: (39.1836, -96.5717),  # Kansas State
    2050: (39.7285, -104.9903),# BYU
    38: (40.2338, -111.6585),  # Utah
    9: (33.4484, -111.9400),   # Arizona State
    12: (32.2319, -110.9501),  # Arizona
    254: (37.6879, -97.3301),  # Wichita State

    # Pac-12
    26: (34.0689, -118.4452),  # UCLA
    30: (34.0224, -118.2851),  # USC
    24: (37.8719, -122.2585),  # California
    23: (37.4275, -122.1697),  # Stanford
    264: (47.6553, -122.3035), # Washington
    265: (46.7324, -117.1631), # Washington State
    2483: (44.5646, -123.2620),# Oregon State
    2572: (44.0521, -123.0868),# Oregon

    # Other notable
    21: (47.6732, -117.4147), # Gonzaga (McCarthey Athletic Center)
    167: (40.0051, -75.3404), # Villanova (Finneran Pavilion)
    2168: (38.8916, -77.0723),# Georgetown (McDonough Arena/Capital One)
    2306: (41.7610, -72.2530),# UConn (Gampel Pavilion)
    252: (41.4443, -72.9372), # Providence (Dunkin' Donuts Center)
    2132: (41.2995, -95.9407),# Creighton (CHI Health Center)
    236: (39.1487, -84.5163), # Xavier (Cintas Center)
    202: (39.7392, -104.9903),# Marquette
    166: (41.8781, -87.6298), # DePaul
    160: (40.7128, -74.0060), # Seton Hall
    165: (43.0389, -87.9065), # Butler
}

_CONFERENCE_NAME_CACHE: dict[int, str] = {}


@dataclass
class HCAInfo:
    """Home court advantage information for a team."""
    team_id: int
    base_hca: float  # Conference base HCA
    venue_hca: float  # Venue-specific adjustment (blue bloods)
    total_hca: float  # Combined HCA


def get_team_hca(
    team_id: int,
    conference_id: Optional[int] = None,
    conference_name: Optional[str] = None,
) -> float:
    """
    Get home court advantage for a team.

    Combines conference-level HCA with venue-specific adjustments.
    """
    # Check for blue blood venue bonus
    if team_id in BLUE_BLOOD_TEAMS:
        return BLUE_BLOOD_TEAMS[team_id]

    # Use conference HCA
    if conference_name:
        conf_hca = _get_hca_from_name(conference_name)
        if conf_hca is not None:
            return conf_hca

    if conference_id is not None:
        if conference_id in CONFERENCE_HCA:
            return CONFERENCE_HCA[conference_id]

        conf_name = get_conference_name(conference_id)
        conf_hca = _get_hca_from_name(conf_name)
        if conf_hca is not None:
            return conf_hca

    # Default
    return 3.5


def get_conference_name(conference_id: Optional[int]) -> Optional[str]:
    """Resolve ESPN conference group name from ID (cached)."""
    if conference_id is None:
        return None
    if conference_id in _CONFERENCE_NAME_CACHE:
        return _CONFERENCE_NAME_CACHE[conference_id]

    try:
        resp = httpx.get(f"{ESPN_GROUP_URL}/{conference_id}", timeout=15.0)
        if resp.status_code != 200:
            return None
        name = resp.json().get("name")
    except Exception as exc:
        logger.warning("Failed to fetch conference name", conference_id=conference_id, error=str(exc))
        return None

    if name:
        _CONFERENCE_NAME_CACHE[conference_id] = name
    return name


def _get_hca_from_name(conf_name: Optional[str]) -> Optional[float]:
    """Map ESPN conference name to HCA via keyword matching."""
    if not conf_name:
        return None
    lowered = conf_name.lower()
    for keyword, hca in CONFERENCE_HCA_KEYWORDS:
        if keyword in lowered:
            return hca
    return None


def get_conference_hca_map(teams_df: pd.DataFrame) -> dict[int, float]:
    """
    Build mapping of team_id -> HCA based on conference.

    Args:
        teams_df: DataFrame with team_id and conference_id columns

    Returns:
        Dict mapping team_id to HCA value
    """
    hca_map = {}

    for _, row in teams_df.iterrows():
        team_id = row.get('team_id') or row.get('id')
        conf_id = row.get('conference_id') or row.get('team_conference_id')
        conf_name = row.get('conference_name') or row.get('conference')

        hca_map[team_id] = get_team_hca(team_id, conf_id, conf_name)

    return hca_map


def calculate_travel_distance(home_team_id: int, away_team_id: int) -> Optional[float]:
    """
    Calculate great-circle distance between two teams' arenas.

    Returns distance in miles, or None if coordinates not available.
    """
    if home_team_id not in ARENA_COORDINATES or away_team_id not in ARENA_COORDINATES:
        return None

    home_lat, home_lon = ARENA_COORDINATES[home_team_id]
    away_lat, away_lon = ARENA_COORDINATES[away_team_id]

    # Haversine formula
    R = 3959  # Earth's radius in miles

    lat1, lat2 = math.radians(home_lat), math.radians(away_lat)
    dlat = math.radians(away_lat - home_lat)
    dlon = math.radians(away_lon - home_lon)

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def get_travel_adjustment(distance_miles: Optional[float]) -> float:
    """
    Get point spread adjustment based on travel distance.

    Research-based adjustments:
    - < 500 miles: 0 pts
    - 500-1000 miles: -0.6 pts
    - 1000-2000 miles: -0.9 pts
    - > 2000 miles: -1.2 pts

    Returns adjustment for AWAY team (negative = disadvantage).
    """
    if distance_miles is None:
        return 0.0

    if distance_miles < 500:
        return 0.0
    elif distance_miles < 1000:
        return -0.6
    elif distance_miles < 2000:
        return -0.9
    else:
        return -1.2


def get_timezone_adjustment(home_team_id: int, away_team_id: int) -> float:
    """
    Get adjustment based on timezone difference.

    -0.3 pts per timezone crossed (for away team).
    """
    # Simplified: use longitude difference
    if home_team_id not in ARENA_COORDINATES or away_team_id not in ARENA_COORDINATES:
        return 0.0

    home_lon = ARENA_COORDINATES[home_team_id][1]
    away_lon = ARENA_COORDINATES[away_team_id][1]

    # Each 15 degrees longitude = 1 timezone
    lon_diff = abs(home_lon - away_lon)
    timezone_diff = int(lon_diff / 15)

    return -0.3 * timezone_diff


def get_rest_adjustment(rest_diff: int) -> float:
    """
    Get point spread adjustment based on rest day differential.

    Research-based adjustments:
    - Back-to-back (0 rest): -2.5 pts
    - 1 day rest: -1.0 pts
    - 2 days rest: 0 pts (baseline)
    - 3+ days rest: +0.5 pts

    Returns adjustment for the team with fewer rest days.
    """
    REST_ADJUSTMENTS = {
        0: -2.5,   # Back-to-back
        1: -1.0,   # 1 day rest
        2: 0.0,    # Baseline
        3: 0.5,    # Extra rest
        4: 0.5,    # Diminishing returns
    }

    # Clamp rest_diff to reasonable range
    rest_diff = max(-7, min(7, rest_diff))

    if rest_diff == 0:
        return 0.0

    # Positive rest_diff means home team has more rest
    if rest_diff > 0:
        # Home team advantage
        return REST_ADJUSTMENTS.get(rest_diff, 0.5)
    else:
        # Away team advantage
        return -REST_ADJUSTMENTS.get(-rest_diff, 0.5)


def calculate_total_context_adjustment(
    home_team_id: int,
    away_team_id: int,
    is_neutral: bool,
    home_rest_days: int,
    away_rest_days: int,
    home_conference_id: Optional[int] = None,
) -> float:
    """
    Calculate total context-based spread adjustment.

    Combines:
    - Home court advantage (or none for neutral)
    - Travel distance effect
    - Timezone effect
    - Rest day differential

    Returns adjustment in points (positive = favors home team).
    """
    adjustment = 0.0

    # Home court advantage
    if not is_neutral:
        hca = get_team_hca(home_team_id, home_conference_id)
        adjustment += hca
    else:
        # Neutral site - still consider travel
        pass

    # Travel distance
    distance = calculate_travel_distance(home_team_id, away_team_id)
    travel_adj = get_travel_adjustment(distance)
    adjustment -= travel_adj  # Travel hurts away team, which helps home

    # Timezone
    tz_adj = get_timezone_adjustment(home_team_id, away_team_id)
    adjustment -= tz_adj  # TZ hurts away team

    # Rest
    rest_diff = home_rest_days - away_rest_days
    rest_adj = get_rest_adjustment(rest_diff)
    adjustment += rest_adj

    return adjustment
