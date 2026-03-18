"""
Tournament bracket loader for March Madness.

Loads the live NCAA tournament bracket from free NCAA data and persists a
normalized bracket graph into DuckDB.
"""

from __future__ import annotations

from datetime import date, datetime
import html
import json
import re
from typing import Any, Optional

import httpx
import structlog
from pydantic import BaseModel

from packages.common.database import get_connection, init_database

logger = structlog.get_logger()

NCAA_APP_CONFIG_URL = "https://mmldata.ncaa.com/mml/{year}/configs/prod/v2/live/appConfig_web.json"
NCAA_DATA_HOST_HEADER = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

NCAA_GQL_OPERATIONS = {
    "official_bracket": "official_bracket_web",
    "scores_bracket": "scores_bracket_web",
}

# NCAA's live web app currently uses these persisted-query hashes in the JS bundle.
# The hashes in appConfig can lag behind the deployed bundle, so we prefer the live
# web values and fall back to appConfig only if needed.
NCAA_PERSISTED_QUERY_HASHES = {
    "official_bracket": "58cd1e8be6f2902dd6d7fed23392b885c7349ea6ff04b740f95cfe8f8c226595",
    "scores_bracket": "e5746c1f7317fbbb07928dee293eb92e7fa30cc349e5ed0c20e45fa94aacc22e",
}

# Round constants
ROUND_FIRST_FOUR = 0
ROUND_OF_64 = 1
ROUND_OF_32 = 2
ROUND_SWEET_16 = 3
ROUND_ELITE_8 = 4
ROUND_FINAL_FOUR = 5
ROUND_CHAMPIONSHIP = 6

ROUND_NAMES = {
    ROUND_FIRST_FOUR: "First Four",
    ROUND_OF_64: "Round of 64",
    ROUND_OF_32: "Round of 32",
    ROUND_SWEET_16: "Sweet 16",
    ROUND_ELITE_8: "Elite 8",
    ROUND_FINAL_FOUR: "Final Four",
    ROUND_CHAMPIONSHIP: "Championship",
}

NCAA_ROUND_MAP = {
    1: ROUND_FIRST_FOUR,
    2: ROUND_OF_64,
    3: ROUND_OF_32,
    4: ROUND_SWEET_16,
    5: ROUND_ELITE_8,
    6: ROUND_FINAL_FOUR,
    7: ROUND_CHAMPIONSHIP,
}

# Region IDs used by the local schema
REGION_EAST = 1
REGION_WEST = 2
REGION_SOUTH = 3
REGION_MIDWEST = 4
REGION_FIRST_FOUR = 5

REGION_NAMES = {
    REGION_EAST: "East",
    REGION_WEST: "West",
    REGION_SOUTH: "South",
    REGION_MIDWEST: "Midwest",
    REGION_FIRST_FOUR: "First Four",
}

NCAA_SECTION_TO_REGION = {
    1: REGION_FIRST_FOUR,
    2: REGION_EAST,
    3: REGION_WEST,
    4: REGION_SOUTH,
    5: REGION_MIDWEST,
}

MANUAL_TEAM_ALIASES = {
    "ohio st": "ohio state",
    "iowa st": "iowa state",
    "tennessee st": "tennessee state",
    "michigan st": "michigan state",
    "north dakota st": "north dakota state",
    "utah st": "utah state",
    "kennesaw st": "kennesaw state",
    "wright st": "wright state",
    "st johns": "st johns",
    "st marys": "saint marys",
    "st louis": "saint louis",
    "penn": "pennsylvania",
    "miami fla": "miami hurricanes",
    "miami fl": "miami hurricanes",
    "miami florida": "miami hurricanes",
    "miami ohio": "miami oh",
    "miami oh": "miami oh",
    "cal baptist": "california baptist",
    "hawaii": "hawaii",
    "queens nc": "queens university",
    "queens n c": "queens university",
    "queens north carolina": "queens university",
    "long island": "long island university",
    "long island university": "long island university",
    "uni": "northern iowa",
    "saint louis": "saint louis",
    "nc state": "north carolina state",
    "prairie view am": "prairie view and m",
}


class BracketSlot(BaseModel):
    """A single slot in the tournament bracket."""

    slot_id: int
    year: int
    region_id: Optional[int]
    round: int
    game_in_round: int
    seed_a: Optional[int]
    seed_b: Optional[int]
    team_a_id: Optional[int]
    team_b_id: Optional[int]
    winner_team_id: Optional[int]
    game_id: Optional[int]
    game_date: Optional[date]
    venue: Optional[str]
    is_first_four: bool
    next_slot_id: Optional[int]
    victor_game_position: Optional[str] = None


class TournamentTeam(BaseModel):
    """A team in the tournament."""

    team_id: int
    name: str
    seed: int
    region_id: int
    region_name: str
    is_first_four: bool = False


# Kept for manual fallback/debugging only; live workflow uses NCAA data.
BRACKET_2026_TEMPLATE = {
    "year": 2026,
    "start_date": "2026-03-17",
    "championship_date": "2026-04-06",
    "regions": {
        REGION_EAST: {
            "name": "East",
            "teams": [
                (1, "Duke"),
                (16, "Siena"),
                (8, "Ohio State"),
                (9, "TCU"),
                (5, "St. John's"),
                (12, "Northern Iowa"),
                (4, "Kansas"),
                (13, "California Baptist"),
                (6, "Louisville"),
                (11, "South Florida"),
                (3, "Michigan State"),
                (14, "North Dakota State"),
                (7, "UCLA"),
                (10, "UCF"),
                (2, "UConn"),
                (15, "Furman"),
            ],
        },
        REGION_WEST: {
            "name": "West",
            "teams": [
                (1, "Arizona"),
                (16, "Long Island University"),
                (8, "Villanova"),
                (9, "Utah State"),
                (5, "Wisconsin"),
                (12, "High Point"),
                (4, "Arkansas"),
                (13, "Hawai'i"),
                (6, "BYU"),
                (11, "TBD_WEST_11"),
                (3, "Gonzaga"),
                (14, "Kennesaw State"),
                (7, "Miami (Fla.)"),
                (10, "Missouri"),
                (2, "Purdue"),
                (15, "Queens (N.C.)"),
            ],
        },
        REGION_SOUTH: {
            "name": "South",
            "teams": [
                (1, "Florida"),
                (16, "TBD_SOUTH_16"),
                (8, "Clemson"),
                (9, "Iowa"),
                (5, "Vanderbilt"),
                (12, "McNeese"),
                (4, "Nebraska"),
                (13, "Troy"),
                (6, "North Carolina"),
                (11, "VCU"),
                (3, "Illinois"),
                (14, "Penn"),
                (7, "Saint Mary's"),
                (10, "Texas A&M"),
                (2, "Houston"),
                (15, "Idaho"),
            ],
        },
        REGION_MIDWEST: {
            "name": "Midwest",
            "teams": [
                (1, "Michigan"),
                (16, "TBD_MIDWEST_16"),
                (8, "Georgia"),
                (9, "Saint Louis"),
                (5, "Texas Tech"),
                (12, "Akron"),
                (4, "Alabama"),
                (13, "Hofstra"),
                (6, "Tennessee"),
                (11, "TBD_MIDWEST_11"),
                (3, "Virginia"),
                (14, "Wright State"),
                (7, "Kentucky"),
                (10, "Santa Clara"),
                (2, "Iowa State"),
                (15, "Tennessee State"),
            ],
        },
    },
    "first_four": [
        {"region": REGION_MIDWEST, "seed": 16, "team1": "UMBC", "team2": "Howard"},
        {"region": REGION_WEST, "seed": 11, "team1": "Texas", "team2": "NC State"},
        {"region": REGION_SOUTH, "seed": 16, "team1": "Prairie View A&M", "team2": "Lehigh"},
        {"region": REGION_MIDWEST, "seed": 11, "team1": "Miami (Ohio)", "team2": "SMU"},
    ],
}


def _clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return html.unescape(str(value)).strip()


def _normalize_team_name(value: str) -> str:
    normalized = _clean_text(value).lower()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("'", "")
    normalized = normalized.replace(".", " ")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("(", " ")
    normalized = normalized.replace(")", " ")
    normalized = re.sub(r"\buniv\b", "university", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    previous = ""
    while previous != normalized:
        previous = normalized
        normalized = MANUAL_TEAM_ALIASES.get(normalized, normalized)
    return normalized


def _team_aliases(name: str) -> set[str]:
    normalized = _normalize_team_name(name)
    aliases = {normalized}

    words = normalized.split()
    if len(words) > 1:
        aliases.add(" ".join(words[:-1]))

    aliases.update(
        {
            normalized.replace("saint", "st"),
            normalized.replace("california", "cal"),
            normalized.replace("hawaii", "hawaii"),
        }
    )

    return {alias.strip() for alias in aliases if alias.strip()}


def _build_team_lookup() -> dict[str, int]:
    with get_connection() as conn:
        rows = conn.execute("SELECT team_id, name FROM teams").fetchall()

    team_lookup: dict[str, int] = {}
    for team_id, name in rows:
        for alias in _team_aliases(str(name)):
            team_lookup.setdefault(alias, int(team_id))

    return team_lookup


def _resolve_team_id(team_fields: dict, team_lookup: dict[str, int]) -> Optional[int]:
    candidates = [
        team_fields.get("textOverride"),
        team_fields.get("nameShort"),
        team_fields.get("seoname"),
    ]

    nickname = _clean_text(team_fields.get("nickname"))
    name_short = _clean_text(team_fields.get("nameShort"))
    if name_short and nickname:
        candidates.append(f"{name_short} {nickname}")

    for candidate in candidates:
        if not candidate:
            continue
        normalized = _normalize_team_name(str(candidate).replace("-", " "))
        if normalized in team_lookup:
            return team_lookup[normalized]

    return None


def _parse_start_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%m/%d/%Y").date()


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round_from_ncaa(round_info: dict) -> int:
    round_number = _safe_int(round_info.get("roundNumber"))
    if round_number is None or round_number not in NCAA_ROUND_MAP:
        raise ValueError(f"Unsupported NCAA round payload: {round_info}")
    return NCAA_ROUND_MAP[round_number]


def _map_region_id(region_info: Optional[dict], round_value: int) -> Optional[int]:
    if round_value >= ROUND_FINAL_FOUR:
        return None
    if round_value == ROUND_FIRST_FOUR:
        return REGION_FIRST_FOUR
    section_id = _safe_int((region_info or {}).get("sectionId"))
    return NCAA_SECTION_TO_REGION.get(section_id) if section_id is not None else None


def _winner_team_id(
    team_a: dict | None, team_b: dict | None, team_lookup: dict[str, int]
) -> Optional[int]:
    for team in (team_a, team_b):
        if team and bool(team.get("isWinner")):
            resolved = _resolve_team_id(team, team_lookup)
            if resolved is not None:
                return resolved
    return None


def fetch_bracket_app_config(year: int = 2026) -> dict:
    """Fetch NCAA app config used by the live bracket site."""
    url = NCAA_APP_CONFIG_URL.format(year=year)
    with httpx.Client(timeout=30.0, headers={"User-Agent": NCAA_DATA_HOST_HEADER}) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()

    gql = payload.get("gql", {})
    if not gql.get("host") or not gql.get("links"):
        raise ValueError("NCAA app config missing GraphQL host/links")

    return payload


def fetch_ncaa_persisted_query(
    host: str,
    operation_name: str,
    query_hash: str,
    season_year: int,
    static_test_env: Optional[str],
) -> dict:
    """Fetch a live NCAA persisted GraphQL query via GET."""
    params = {
        "operationName": operation_name,
        "variables": json.dumps(
            {"seasonYear": season_year, "staticTestEnv": static_test_env},
            separators=(",", ":"),
        ),
        "extensions": json.dumps(
            {"persistedQuery": {"version": 1, "sha256Hash": query_hash}},
            separators=(",", ":"),
        ),
    }

    with httpx.Client(timeout=30.0, headers={"User-Agent": NCAA_DATA_HOST_HEADER}) as client:
        response = client.get(host, params=params)
        response.raise_for_status()
        payload = response.json()

    if payload.get("errors"):
        raise ValueError(f"NCAA query {operation_name} returned errors: {payload['errors']}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"NCAA query {operation_name} returned no data")

    return data


def fetch_live_bracket_payload(year: int = 2026) -> dict:
    """Fetch live official and score-aware bracket payloads from NCAA."""
    app_config = fetch_bracket_app_config(year)
    gql = app_config["gql"]
    season_year = int(gql["season_year"])
    host = gql["host"]
    static_test_env = gql.get("static_test_env")
    links = gql["links"]

    official_hash = NCAA_PERSISTED_QUERY_HASHES["official_bracket"]
    scores_hash = NCAA_PERSISTED_QUERY_HASHES["scores_bracket"]

    official = fetch_ncaa_persisted_query(
        host=host,
        operation_name=NCAA_GQL_OPERATIONS["official_bracket"],
        query_hash=official_hash,
        season_year=season_year,
        static_test_env=static_test_env,
    )
    scores = fetch_ncaa_persisted_query(
        host=host,
        operation_name=NCAA_GQL_OPERATIONS["scores_bracket"],
        query_hash=scores_hash,
        season_year=season_year,
        static_test_env=static_test_env,
    )

    official_contests = official.get("mmlContests", [])
    score_contests = scores.get("mmlContests", [])
    if not official_contests or not score_contests:
        raise ValueError("Live NCAA bracket payload returned no contests")

    logger.info(
        "Fetched live NCAA bracket payload",
        tournament_year=year,
        season_year=season_year,
        official_games=len(official_contests),
        scored_games=len(score_contests),
    )

    return {
        "year": year,
        "season_year": season_year,
        "official_contests": official_contests,
        "score_contests": score_contests,
    }


def build_live_bracket_slots(
    year: int,
    official_contests: list[dict],
    score_contests: list[dict],
    team_lookup: Optional[dict[str, int]] = None,
) -> list[BracketSlot]:
    """Build normalized bracket slots from NCAA live payloads."""
    score_by_slot = {int(contest["bracketId"]): contest for contest in score_contests}
    resolved_lookup = team_lookup or _build_team_lookup()

    pending_slots: list[BracketSlot] = []
    for official in sorted(official_contests, key=lambda item: int(item["bracketId"])):
        slot_id = int(official["bracketId"])
        scored = score_by_slot.get(slot_id, {})
        round_value = _round_from_ncaa(official.get("round", {}))
        region_id = _map_region_id(official.get("region"), round_value)
        teams = {bool(team.get("isTop")): team for team in official.get("teams", [])}
        top_team = teams.get(True)
        bottom_team = teams.get(False)
        winner_of = scored.get("winnerOf", []) or []

        seed_a = _safe_int(top_team.get("seed")) if top_team else None
        seed_b = _safe_int(bottom_team.get("seed")) if bottom_team else None

        for previous in winner_of:
            previous_seed = _safe_int(previous.get("homeSeed")) or _safe_int(
                previous.get("visitSeed")
            )
            if bool(previous.get("isTop")) and seed_a is None:
                seed_a = previous_seed
            if not bool(previous.get("isTop")) and seed_b is None:
                seed_b = previous_seed

        pending_slots.append(
            BracketSlot(
                slot_id=slot_id,
                year=year,
                region_id=region_id,
                round=round_value,
                game_in_round=0,
                seed_a=seed_a,
                seed_b=seed_b,
                team_a_id=_resolve_team_id(top_team, resolved_lookup) if top_team else None,
                team_b_id=_resolve_team_id(bottom_team, resolved_lookup) if bottom_team else None,
                winner_team_id=_winner_team_id(top_team, bottom_team, resolved_lookup),
                game_id=_safe_int(scored.get("contestId")),
                game_date=_parse_start_date(scored.get("startDate") or official.get("startDate")),
                venue=_clean_text((scored.get("location") or {}).get("venue")) or None,
                is_first_four=round_value == ROUND_FIRST_FOUR,
                next_slot_id=_safe_int(
                    scored.get("victorBracketPositionId") or official.get("victorBracketPositionId")
                ),
                victor_game_position=_clean_text(
                    scored.get("victorGamePosition") or official.get("victorGamePosition")
                )
                or None,
            )
        )

    slots: list[BracketSlot] = []
    game_numbers: dict[int, int] = {}
    for slot in sorted(pending_slots, key=lambda item: (item.round, item.slot_id)):
        game_numbers[slot.round] = game_numbers.get(slot.round, 0) + 1
        slot.game_in_round = game_numbers[slot.round]
        slots.append(slot)

    return slots


def resolve_team_ids(bracket_data: dict) -> dict:
    """Resolve template bracket team names using the current teams table."""
    team_lookup = _build_team_lookup()

    for region in bracket_data.get("regions", {}).values():
        resolved_teams = []
        for seed, team_name in region["teams"]:
            team_id = _resolve_team_id({"nameShort": team_name, "seoname": team_name}, team_lookup)
            resolved_teams.append((seed, team_name, team_id))
        region["teams"] = resolved_teams

    for first_four in bracket_data.get("first_four", []):
        first_four["team1_id"] = _resolve_team_id(
            {"nameShort": first_four["team1"], "seoname": first_four["team1"]},
            team_lookup,
        )
        first_four["team2_id"] = _resolve_team_id(
            {"nameShort": first_four["team2"], "seoname": first_four["team2"]},
            team_lookup,
        )

    return bracket_data


def generate_bracket_slots(year: int, bracket_data: dict) -> list[BracketSlot]:
    """Generate the standard 67-slot bracket from the local template structure."""
    bracket_data = resolve_team_ids(bracket_data)
    slots: list[BracketSlot] = []
    slot_id = 1

    ff_targets = {
        (REGION_MIDWEST, 16): (29, "Bottom"),
        (REGION_WEST, 11): (17, "Bottom"),
        (REGION_SOUTH, 16): (21, "Bottom"),
        (REGION_MIDWEST, 11): (33, "Bottom"),
    }

    ff_games = bracket_data.get("first_four", [])
    for i, first_four in enumerate(ff_games):
        target_slot, target_position = ff_targets.get(
            (first_four["region"], first_four["seed"]),
            (None, None),
        )
        slots.append(
            BracketSlot(
                slot_id=slot_id,
                year=year,
                region_id=REGION_FIRST_FOUR,
                round=ROUND_FIRST_FOUR,
                game_in_round=i + 1,
                seed_a=first_four["seed"],
                seed_b=first_four["seed"],
                team_a_id=first_four.get("team1_id"),
                team_b_id=first_four.get("team2_id"),
                winner_team_id=None,
                game_id=None,
                game_date=date(year, 3, 17 + (i // 2)),
                venue="UD Arena",
                is_first_four=True,
                next_slot_id=target_slot,
                victor_game_position=target_position,
            )
        )
        slot_id += 1

    r64_start_slot = slot_id
    matchups = [
        (0, 15),
        (7, 8),
        (4, 11),
        (3, 12),
        (5, 10),
        (2, 13),
        (6, 9),
        (1, 14),
    ]

    for region_id in [REGION_EAST, REGION_WEST, REGION_SOUTH, REGION_MIDWEST]:
        teams = bracket_data["regions"][region_id]["teams"]
        for matchup_index, (idx_a, idx_b) in enumerate(matchups):
            seed_a, _name_a, id_a = teams[idx_a]
            seed_b, _name_b, id_b = teams[idx_b]
            next_slot = r64_start_slot + 32 + (region_id - 1) * 4 + (matchup_index // 2)
            slots.append(
                BracketSlot(
                    slot_id=slot_id,
                    year=year,
                    region_id=region_id,
                    round=ROUND_OF_64,
                    game_in_round=matchup_index + 1,
                    seed_a=seed_a,
                    seed_b=seed_b,
                    team_a_id=id_a,
                    team_b_id=id_b,
                    winner_team_id=None,
                    game_id=None,
                    game_date=date(year, 3, 19 + ((region_id - 1) // 2)),
                    venue=None,
                    is_first_four=False,
                    next_slot_id=next_slot,
                    victor_game_position="Top" if matchup_index % 2 == 0 else "Bottom",
                )
            )
            slot_id += 1

    r32_start_slot = slot_id
    for region_id in [REGION_EAST, REGION_WEST, REGION_SOUTH, REGION_MIDWEST]:
        for game_index in range(4):
            slots.append(
                BracketSlot(
                    slot_id=slot_id,
                    year=year,
                    region_id=region_id,
                    round=ROUND_OF_32,
                    game_in_round=game_index + 1,
                    seed_a=None,
                    seed_b=None,
                    team_a_id=None,
                    team_b_id=None,
                    winner_team_id=None,
                    game_id=None,
                    game_date=date(year, 3, 21 + (game_index // 2)),
                    venue=None,
                    is_first_four=False,
                    next_slot_id=r32_start_slot + 16 + (region_id - 1) * 2 + (game_index // 2),
                    victor_game_position="Top" if game_index % 2 == 0 else "Bottom",
                )
            )
            slot_id += 1

    s16_start_slot = slot_id
    for region_id in [REGION_EAST, REGION_WEST, REGION_SOUTH, REGION_MIDWEST]:
        for game_index in range(2):
            slots.append(
                BracketSlot(
                    slot_id=slot_id,
                    year=year,
                    region_id=region_id,
                    round=ROUND_SWEET_16,
                    game_in_round=game_index + 1,
                    seed_a=None,
                    seed_b=None,
                    team_a_id=None,
                    team_b_id=None,
                    winner_team_id=None,
                    game_id=None,
                    game_date=date(year, 3, 26 + (region_id % 2)),
                    venue=None,
                    is_first_four=False,
                    next_slot_id=s16_start_slot + 8 + (region_id - 1),
                    victor_game_position="Top" if game_index == 0 else "Bottom",
                )
            )
            slot_id += 1

    e8_start_slot = slot_id
    for region_id in [REGION_EAST, REGION_WEST, REGION_SOUTH, REGION_MIDWEST]:
        slots.append(
            BracketSlot(
                slot_id=slot_id,
                year=year,
                region_id=region_id,
                round=ROUND_ELITE_8,
                game_in_round=1,
                seed_a=None,
                seed_b=None,
                team_a_id=None,
                team_b_id=None,
                winner_team_id=None,
                game_id=None,
                game_date=date(year, 3, 28 + (region_id % 2)),
                venue=None,
                is_first_four=False,
                next_slot_id=e8_start_slot + 4 + ((region_id - 1) // 2),
                victor_game_position="Top"
                if region_id in (REGION_EAST, REGION_SOUTH)
                else "Bottom",
            )
        )
        slot_id += 1

    championship_slot_id = slot_id + 2
    slots.append(
        BracketSlot(
            slot_id=slot_id,
            year=year,
            region_id=None,
            round=ROUND_FINAL_FOUR,
            game_in_round=1,
            seed_a=None,
            seed_b=None,
            team_a_id=None,
            team_b_id=None,
            winner_team_id=None,
            game_id=None,
            game_date=date(year, 4, 4),
            venue="Lucas Oil Stadium",
            is_first_four=False,
            next_slot_id=championship_slot_id,
            victor_game_position="Top",
        )
    )
    slot_id += 1
    slots.append(
        BracketSlot(
            slot_id=slot_id,
            year=year,
            region_id=None,
            round=ROUND_FINAL_FOUR,
            game_in_round=2,
            seed_a=None,
            seed_b=None,
            team_a_id=None,
            team_b_id=None,
            winner_team_id=None,
            game_id=None,
            game_date=date(year, 4, 4),
            venue="Lucas Oil Stadium",
            is_first_four=False,
            next_slot_id=championship_slot_id,
            victor_game_position="Bottom",
        )
    )
    slot_id += 1
    slots.append(
        BracketSlot(
            slot_id=slot_id,
            year=year,
            region_id=None,
            round=ROUND_CHAMPIONSHIP,
            game_in_round=1,
            seed_a=None,
            seed_b=None,
            team_a_id=None,
            team_b_id=None,
            winner_team_id=None,
            game_id=None,
            game_date=date(year, 4, 6),
            venue="Lucas Oil Stadium",
            is_first_four=False,
            next_slot_id=None,
            victor_game_position=None,
        )
    )

    return slots


def save_bracket_to_db(year: int, bracket_payload: dict | list[BracketSlot]) -> int:
    """Save a normalized bracket payload to DuckDB."""
    if isinstance(bracket_payload, list):
        slots = bracket_payload
        start_date = min(
            (slot.game_date for slot in slots if slot.game_date is not None),
            default=date(year, 3, 17),
        )
        championship_date = max(
            (slot.game_date for slot in slots if slot.game_date is not None),
            default=date(year, 4, 6),
        )
    elif isinstance(bracket_payload, dict) and "slots" in bracket_payload:
        slots = bracket_payload["slots"]
        start_date = date.fromisoformat(bracket_payload.get("start_date", f"{year}-03-17"))
        championship_date = date.fromisoformat(
            bracket_payload.get("championship_date", f"{year}-04-06")
        )
    else:
        resolved = resolve_team_ids(bracket_payload)
        slots = generate_bracket_slots(year, resolved)
        start_date = date.fromisoformat(resolved.get("start_date", f"{year}-03-17"))
        championship_date = date.fromisoformat(resolved.get("championship_date", f"{year}-04-06"))

    with get_connection() as conn:
        try:
            conn.execute(
                "ALTER TABLE tournament_bracket ADD COLUMN victor_game_position VARCHAR(10)"
            )
        except Exception:
            pass

        conn.execute("DELETE FROM tournament_bracket WHERE year = ?", (year,))
        conn.execute("DELETE FROM tournament_predictions WHERE year = ?", (year,))
        conn.execute("DELETE FROM tournament_regions WHERE year = ?", (year,))

        conn.execute(
            """
            INSERT OR REPLACE INTO tournament_years (year, start_date, championship_date, num_teams, status)
            VALUES (?, ?, ?, 68, 'upcoming')
            """,
            (year, start_date.isoformat(), championship_date.isoformat()),
        )

        for region_id, region_name in REGION_NAMES.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO tournament_regions (region_id, year, region_name, display_order)
                VALUES (?, ?, ?, ?)
                """,
                (region_id, year, region_name, region_id),
            )

        for slot in slots:
            conn.execute(
                """
                INSERT OR REPLACE INTO tournament_bracket
                (slot_id, year, region_id, round, game_in_round, seed_a, seed_b,
                 team_a_id, team_b_id, winner_team_id, game_id, game_date, venue,
                 is_first_four, next_slot_id, victor_game_position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slot.slot_id,
                    slot.year,
                    slot.region_id,
                    slot.round,
                    slot.game_in_round,
                    slot.seed_a,
                    slot.seed_b,
                    slot.team_a_id,
                    slot.team_b_id,
                    slot.winner_team_id,
                    slot.game_id,
                    slot.game_date.isoformat() if slot.game_date else None,
                    slot.venue,
                    slot.is_first_four,
                    slot.next_slot_id,
                    slot.victor_game_position,
                ),
            )

    logger.info("Bracket saved to database", year=year, slots=len(slots))
    return len(slots)


def load_bracket_from_db(year: int) -> list[dict]:
    """Load tournament bracket slots from DuckDB."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT
                tb.*,
                ta.name AS team_a_name,
                tb2.name AS team_b_name,
                tr.region_name
            FROM tournament_bracket tb
            LEFT JOIN teams ta ON tb.team_a_id = ta.team_id
            LEFT JOIN teams tb2 ON tb.team_b_id = tb2.team_id
            LEFT JOIN tournament_regions tr
                ON tb.region_id = tr.region_id AND tr.year = tb.year
            WHERE tb.year = ?
            ORDER BY tb.round, tb.game_in_round, tb.slot_id
            """,
            (year,),
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

    return [dict(zip(columns, row)) for row in rows]


def _maybe_refresh_teams_for_bracket() -> None:
    try:
        from packages.ingest.espn_api import fetch_teams, save_teams_to_db

        teams = fetch_teams()
        if teams:
            save_teams_to_db(teams)
            logger.info("Refreshed teams table before tournament team resolution", count=len(teams))
    except Exception as exc:
        logger.warning("Could not refresh teams for tournament bracket", error=str(exc))


def _derive_abbreviation(name: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", _clean_text(name))
    initials = "".join(part[0] for part in parts if part).upper()
    if 2 <= len(initials) <= 10:
        return initials
    compact = "".join(parts).upper()
    return compact[:10] if compact else "TEAM"


def _maybe_seed_teams_from_tournament_schedule(year: int) -> None:
    try:
        from packages.common.schemas import Team
        from packages.ingest.espn_api import fetch_schedule, save_teams_to_db

        with get_connection() as conn:
            existing_ids = {
                int(row[0]) for row in conn.execute("SELECT team_id FROM teams").fetchall()
            }

        inferred_teams: list[Team] = []
        current = date(year, 3, 17)
        end_date = date(year, 3, 20)
        while current <= end_date:
            for game in fetch_schedule(current):
                for team_id, team_name in (
                    (game.home_team_id, game.home_team_name),
                    (game.away_team_id, game.away_team_name),
                ):
                    if team_id <= 0 or team_id in existing_ids or not team_name:
                        continue
                    inferred_teams.append(
                        Team(
                            team_id=team_id,
                            name=team_name,
                            abbreviation=_derive_abbreviation(team_name),
                            conference="Unknown",
                        )
                    )
                    existing_ids.add(team_id)
            current = date.fromordinal(current.toordinal() + 1)

        if inferred_teams:
            save_teams_to_db(inferred_teams)
            logger.info("Seeded missing teams from tournament schedule", count=len(inferred_teams))
    except Exception as exc:
        logger.warning("Could not seed missing tournament teams from schedule", error=str(exc))


def load_2026_bracket(year: int = 2026, use_template: bool = False) -> int:
    """Load the live 2026 March Madness bracket into the database."""
    init_database()
    logger.info("Loading tournament bracket", year=year, use_template=use_template)

    if use_template:
        payload = BRACKET_2026_TEMPLATE.copy()
        payload["start_date"] = f"{year}-03-17"
        payload["championship_date"] = f"{year}-04-06"
        return save_bracket_to_db(year, payload)

    live_payload = fetch_live_bracket_payload(year)
    _maybe_seed_teams_from_tournament_schedule(year)
    team_lookup = _build_team_lookup()
    slots = build_live_bracket_slots(
        year=year,
        official_contests=live_payload["official_contests"],
        score_contests=live_payload["score_contests"],
        team_lookup=team_lookup,
    )

    unresolved = [
        slot.slot_id
        for slot in slots
        if (slot.round == ROUND_FIRST_FOUR and slot.seed_a is not None and slot.team_a_id is None)
        or (slot.round == ROUND_FIRST_FOUR and slot.seed_b is not None and slot.team_b_id is None)
        or (slot.round == ROUND_OF_64 and slot.seed_a is not None and slot.team_a_id is None)
    ]
    if unresolved:
        logger.warning(
            "Unresolved tournament teams found, refreshing team master", slots=unresolved
        )
        _maybe_refresh_teams_for_bracket()
        _maybe_seed_teams_from_tournament_schedule(year)
        slots = build_live_bracket_slots(
            year=year,
            official_contests=live_payload["official_contests"],
            score_contests=live_payload["score_contests"],
            team_lookup=_build_team_lookup(),
        )

    saved = save_bracket_to_db(year, slots)
    logger.info("Live tournament bracket loaded", year=year, slots=saved)
    return saved


if __name__ == "__main__":
    load_2026_bracket()
