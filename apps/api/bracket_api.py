"""
FastAPI server for interactive round-by-round NCAA tournament bracket simulation.

Loads team data from DuckDB on startup, maintains bracket state in memory,
and provides endpoints to simulate rounds and retrieve bracket state.

Run: uvicorn apps.api.bracket_api:app --host 0.0.0.0 --port 2502
"""

import copy
import time
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
HAS_GPU = False
try:
    import cupy as cp

    if cp.cuda.is_available():
        HAS_GPU = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "cbb_lines.duckdb"
BRACKET_HTML = PROJECT_ROOT / "apps" / "dashboard" / "bracket.html"

# ---------------------------------------------------------------------------
# Constants from run_march_madness_v2.py
# ---------------------------------------------------------------------------
REGIONS = {
    "East": {
        1: "Duke", 2: "UConn", 3: "Michigan State", 4: "Kansas",
        5: "St. John's", 6: "Louisville", 7: "UCLA", 8: "Ohio State",
        9: "TCU", 10: "UCF", 11: "South Florida", 12: "Northern Iowa",
        13: "Cal Baptist", 14: "North Dakota State", 15: "Furman", 16: "Siena",
    },
    "West": {
        1: "Arizona", 2: "Purdue", 3: "Gonzaga", 4: "Arkansas",
        5: "Wisconsin", 6: "BYU", 7: "Miami", 8: "Villanova",
        9: "Utah State", 10: "Missouri", 11: "Texas", 12: "High Point",
        13: "Hawaii", 14: "Kennesaw State", 15: "Queens", 16: "Long Island",
    },
    "South": {
        1: "Florida", 2: "Houston", 3: "Illinois", 4: "Nebraska",
        5: "Vanderbilt", 6: "North Carolina", 7: "Saint Mary's", 8: "Clemson",
        9: "Iowa", 10: "Texas A&M", 11: "VCU", 12: "McNeese",
        13: "Troy", 14: "Penn", 15: "Idaho", 16: "Lehigh",
    },
    "Midwest": {
        1: "Michigan", 2: "Iowa State", 3: "Virginia", 4: "Alabama",
        5: "Texas Tech", 6: "Tennessee", 7: "Kentucky", 8: "Georgia",
        9: "Saint Louis", 10: "Santa Clara", 11: "SMU", 12: "Akron",
        13: "Hofstra", 14: "Wright State", 15: "Tennessee State", 16: "UMBC",
    },
}

R64_MATCHUPS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

BRACKET_TEAM_MAP = {
    "Duke": "Duke Blue Devils",
    "UConn": "UConn Huskies",
    "Michigan State": "Michigan State Spartans",
    "Kansas": "Kansas Jayhawks",
    "St. John's": "St. John's Red Storm",
    "Louisville": "Louisville Cardinals",
    "UCLA": "UCLA Bruins",
    "Ohio State": "Ohio State Buckeyes",
    "TCU": "TCU Horned Frogs",
    "UCF": "UCF Knights",
    "South Florida": "South Florida Bulls",
    "Northern Iowa": "Northern Iowa Panthers",
    "Cal Baptist": "California Baptist Lancers",
    "North Dakota State": "North Dakota State Bison",
    "Furman": "Furman Paladins",
    "Siena": "Siena Saints",
    "Arizona": "Arizona Wildcats",
    "Purdue": "Purdue Boilermakers",
    "Gonzaga": "Gonzaga Bulldogs",
    "Arkansas": "Arkansas Razorbacks",
    "Wisconsin": "Wisconsin Badgers",
    "BYU": "BYU Cougars",
    "Miami": "Miami Hurricanes",
    "Villanova": "Villanova Wildcats",
    "Utah State": "Utah State Aggies",
    "Missouri": "Missouri Tigers",
    "Texas": "Texas Longhorns",
    "High Point": "High Point Panthers",
    "Hawaii": "Hawai'i Rainbow Warriors",
    "Hawai'i": "Hawai'i Rainbow Warriors",
    "Kennesaw State": "Kennesaw State Owls",
    "Queens": "Queens University Royals",
    "Long Island": "Long Island University Sharks",
    "Florida": "Florida Gators",
    "Houston": "Houston Cougars",
    "Illinois": "Illinois Fighting Illini",
    "Nebraska": "Nebraska Cornhuskers",
    "Vanderbilt": "Vanderbilt Commodores",
    "North Carolina": "North Carolina Tar Heels",
    "Saint Mary's": "Saint Mary's Gaels",
    "Clemson": "Clemson Tigers",
    "Iowa": "Iowa Hawkeyes",
    "Texas A&M": "Texas A&M Aggies",
    "VCU": "VCU Rams",
    "McNeese": "McNeese Cowboys",
    "Troy": "Troy Trojans",
    "Penn": "Pennsylvania Quakers",
    "Idaho": "Idaho Vandals",
    "Lehigh": "Lehigh Mountain Hawks",
    "Michigan": "Michigan Wolverines",
    "Iowa State": "Iowa State Cyclones",
    "Virginia": "Virginia Cavaliers",
    "Alabama": "Alabama Crimson Tide",
    "Texas Tech": "Texas Tech Red Raiders",
    "Tennessee": "Tennessee Volunteers",
    "Kentucky": "Kentucky Wildcats",
    "Georgia": "Georgia Bulldogs",
    "Saint Louis": "Saint Louis Billikens",
    "Santa Clara": "Santa Clara Broncos",
    "SMU": "SMU Mustangs",
    "Akron": "Akron Zips",
    "Hofstra": "Hofstra Pride",
    "Wright State": "Wright State Raiders",
    "Tennessee State": "Tennessee State Tigers",
    "UMBC": "UMBC Retrievers",
    # First Four extra teams
    "Howard": "Howard Bison",
    "NC State": "NC State Wolfpack",
    "Prairie View": "Prairie View A&M Panthers",
    "Miami (OH)": "Miami (OH) RedHawks",
}

# First Four games:
#   Midwest 16: UMBC vs Howard  -> winner takes Midwest 16 slot
#   West 11:    Texas vs NC State -> winner takes West 11 slot
#   South 16:   Prairie View vs Lehigh -> winner takes South 16 slot
#   Midwest 11: Miami (OH) vs SMU -> winner takes Midwest 11 slot
FIRST_FOUR = [
    {
        "id": 101, "region": "Midwest", "slot_seed": 16,
        "team_a": "UMBC", "seed_a": 16, "team_b": "Howard", "seed_b": 16,
    },
    {
        "id": 102, "region": "West", "slot_seed": 11,
        "team_a": "Texas", "seed_a": 11, "team_b": "NC State", "seed_b": 11,
    },
    {
        "id": 103, "region": "South", "slot_seed": 16,
        "team_a": "Prairie View", "seed_a": 16, "team_b": "Lehigh", "seed_b": 16,
    },
    {
        "id": 104, "region": "Midwest", "slot_seed": 11,
        "team_a": "Miami (OH)", "seed_a": 11, "team_b": "SMU", "seed_b": 11,
    },
]

ROUND_NAMES = {
    0: "First Four",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}

# ---------------------------------------------------------------------------
# TeamData class (copied from run_march_madness_v2.py)
# ---------------------------------------------------------------------------
class TeamData:
    """Complete team data for matchup modeling."""

    def __init__(
        self,
        name: str,
        adj_off: float,
        adj_def: float,
        adj_em: float,
        tempo: float,
        efg_pct: float,
        tov_pct: float,
        orb_pct: float,
        ftr: float,
        def_efg: float,
        def_tov: float,
        def_drb: float,
        def_ftr: float,
        off_std: float = 10.0,
        def_std: float = 10.0,
    ):
        self.name = name
        self.adj_off = adj_off
        self.adj_def = adj_def
        self.adj_em = adj_em
        self.tempo = tempo
        self.efg_pct = efg_pct
        self.tov_pct = tov_pct
        self.orb_pct = orb_pct
        self.ftr = ftr
        self.def_efg = def_efg
        self.def_tov = def_tov
        self.def_drb = def_drb
        self.def_ftr = def_ftr
        self.off_std = off_std
        self.def_std = def_std


# ---------------------------------------------------------------------------
# Data loading helpers (adapted from run_march_madness_v2.py)
# ---------------------------------------------------------------------------
def load_teams() -> dict[str, TeamData]:
    """Load team data from DuckDB database."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    rows = conn.execute("""
        SELECT
            t.name,
            ts.adj_offensive_efficiency,
            ts.adj_defensive_efficiency,
            ts.adj_em,
            ts.adj_tempo,
            ts.off_efg, ts.off_tov, ts.off_orb, ts.off_ftr,
            ts.def_efg, ts.def_tov, ts.def_drb, ts.def_ftr,
            ts.off_rating_std, ts.def_rating_std
        FROM team_strengths ts
        JOIN teams t ON ts.team_id = t.team_id
        WHERE ts.as_of_date = (SELECT MAX(as_of_date) FROM team_strengths)
    """).fetchall()
    conn.close()

    teams: dict[str, TeamData] = {}
    for row in rows:
        name = row[0]
        t = TeamData(
            name=name,
            adj_off=row[1],
            adj_def=row[2],
            adj_em=row[3],
            tempo=row[4],
            efg_pct=row[5],
            tov_pct=row[6],
            orb_pct=row[7],
            ftr=row[8],
            def_efg=row[9],
            def_tov=row[10],
            def_drb=row[11],
            def_ftr=row[12],
            off_std=row[13] or 10.0,
            def_std=row[14] or 10.0,
        )
        teams[name] = t
        teams[name.lower()] = t
        parts = name.lower().split()
        if len(parts) > 1:
            teams[parts[-1].lower()] = t
            teams[parts[0].lower()] = t
    return teams


def find_team(name: str, teams: dict[str, TeamData]) -> Optional[TeamData]:
    """Find team by name using explicit mapping, with fallbacks."""
    if not name:
        return None

    apostrophes = ["\u2018", "\u2019", "\u00b4", "\u2018"]
    for apos in apostrophes:
        name = name.replace(apos, "'")

    if name in BRACKET_TEAM_MAP:
        db_name = BRACKET_TEAM_MAP[name]
        if db_name is None:
            return None
        db_name_lower = db_name.lower()
        if db_name_lower in teams:
            return teams[db_name_lower]
        return None

    name_lower = name.lower()
    if name_lower in teams:
        return teams[name_lower]

    for key, t in teams.items():
        if key.lower() == name_lower:
            return t

    return None


def make_fallback_team(name: str, seed: int) -> TeamData:
    """Create a fallback team with seed-based estimates when DB lookup fails."""
    base_eff = 100 - (seed - 8) * 3
    return TeamData(
        name=name,
        adj_off=base_eff + 5,
        adj_def=base_eff - 5,
        adj_em=10,
        tempo=68,
        efg_pct=0.50,
        tov_pct=0.18,
        orb_pct=0.28,
        ftr=0.35,
        def_efg=0.50,
        def_tov=0.18,
        def_drb=0.72,
        def_ftr=0.35,
        off_std=10.0,
        def_std=10.0,
    )


# ---------------------------------------------------------------------------
# calc_game (copied from run_march_madness_v2.py)
# ---------------------------------------------------------------------------
def calc_game(a: TeamData, b: TeamData, rng_values):
    """
    Calculate game outcome using efficiency matchup + Four Factors + matchup features.

    Returns: (score_a, score_b, winner_code, poss, ppp_a, ppp_b)
    Handles ties with overtime.
    """
    tempo_a = a.tempo
    tempo_b = b.tempo

    tempo_diff = abs(tempo_a - tempo_b)
    if tempo_a > tempo_b:
        pace_control_a = 0.45 + (tempo_diff / 100) * 0.1
    else:
        pace_control_a = 0.55 - (tempo_diff / 100) * 0.1
    pace_control_a = max(0.3, min(0.7, pace_control_a))
    pace_control_b = 1.0 - pace_control_a

    base_poss = tempo_a * pace_control_a + tempo_b * pace_control_b
    base_poss *= 0.97  # Neutral site adjustment

    # Points per possession
    ppp_a_base = a.adj_off / 100.0
    ppp_b_base = b.adj_off / 100.0

    ppp_a_base *= b.adj_def / 100.0
    ppp_b_base *= a.adj_def / 100.0

    # Four Factors matchup adjustment
    efg_edge_a = (a.efg_pct - b.def_efg) * 1.2
    efg_edge_b = (b.efg_pct - a.def_efg) * 1.2

    tov_edge_a = (b.def_tov - a.tov_pct) * 0.8
    tov_edge_b = (a.def_tov - b.tov_pct) * 0.8

    orb_edge_a = (a.orb_pct - (1 - b.def_drb)) * 0.6
    orb_edge_b = (b.orb_pct - (1 - a.def_drb)) * 0.6

    ftr_edge_a = (a.ftr - b.def_ftr) * 0.4
    ftr_edge_b = (b.ftr - a.def_ftr) * 0.4

    # Matchup-specific features
    three_pt_factor_a = 1.0
    three_pt_factor_b = 1.0

    if a.efg_pct > 0.55 and b.def_efg < 0.48:
        three_pt_factor_a = 1.03
    if b.efg_pct > 0.55 and a.def_efg < 0.48:
        three_pt_factor_b = 1.03

    interior_edge_a = (a.orb_pct - 0.28) * 0.5 - (b.def_drb - 0.72) * 0.3
    interior_edge_b = (b.orb_pct - 0.28) * 0.5 - (a.def_drb - 0.72) * 0.3

    turnover_edge_a = (b.def_tov - 0.18) * 0.4 + (a.tov_pct - 0.18) * (-0.4)
    turnover_edge_b = (a.def_tov - 0.18) * 0.4 + (b.tov_pct - 0.18) * (-0.4)

    matchup_adj_a = (interior_edge_a + turnover_edge_a) * 0.02
    matchup_adj_b = (interior_edge_b + turnover_edge_b) * 0.02

    ff_adj_a = (efg_edge_a + tov_edge_a + orb_edge_a + ftr_edge_a) * 0.15
    ff_adj_b = (efg_edge_b + tov_edge_b + orb_edge_b + ftr_edge_b) * 0.15

    ppp_a = (ppp_a_base + ff_adj_a + matchup_adj_a) * three_pt_factor_a
    ppp_b = (ppp_b_base + ff_adj_b + matchup_adj_b) * three_pt_factor_b

    # Variance
    pace_clash_variance = 0.02 * min(tempo_diff / 10, 1.0)
    ppp_var_a = 0.095 * (1 + 0.15 * (a.off_std / 10)) + pace_clash_variance
    ppp_var_b = 0.095 * (1 + 0.15 * (b.off_std / 10)) + pace_clash_variance

    ppp_a_random = ppp_a + rng_values[0] * ppp_var_a
    ppp_b_random = ppp_b + rng_values[1] * ppp_var_b

    poss_var = base_poss * 0.035
    poss = base_poss + rng_values[2] * poss_var

    score_a = poss * ppp_a_random
    score_b = poss * ppp_b_random

    score_a_int = int(round(score_a))
    score_b_int = int(round(score_b))

    # Overtime
    if score_a_int == score_b_int:
        ot_poss = 4 + rng_values[3] * 1.5
        ot_ppp_a = ppp_a_random * 1.05
        ot_ppp_b = ppp_b_random * 1.05
        score_a_int += int(round(ot_poss * ot_ppp_a))
        score_b_int += int(round(ot_poss * ot_ppp_b))
        if score_a_int == score_b_int:
            score_a_int += 1 if rng_values[4] > 0.5 else 0

    winner = "A" if score_a_int > score_b_int else "B"
    return score_a_int, score_b_int, winner, poss, ppp_a_random, ppp_b_random


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class GameResult(BaseModel):
    id: int
    round: int
    round_name: str = ""
    region: str = ""
    team_a: Optional[str] = None
    seed_a: Optional[int] = None
    team_b: Optional[str] = None
    seed_b: Optional[int] = None
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    win_prob_a: Optional[float] = None
    spread: Optional[float] = None
    total: Optional[float] = None
    score_a_p10: Optional[float] = None
    score_a_p50: Optional[float] = None
    score_a_p90: Optional[float] = None
    score_b_p10: Optional[float] = None
    score_b_p50: Optional[float] = None
    score_b_p90: Optional[float] = None
    winner: Optional[str] = None
    winner_seed: Optional[int] = None
    simulated: bool = False


class BracketStateResponse(BaseModel):
    current_round: int
    games: list[GameResult]
    advancement: dict[str, dict[str, float]]


class SimulateRoundRequest(BaseModel):
    round: int = Field(..., ge=0, le=6)
    num_sims: int = Field(default=100_000, ge=1000, le=2_000_000)


class SimulateRoundResponse(BaseModel):
    round: int
    round_name: str
    num_sims: int
    runtime_ms: float
    gpu_used: bool
    games: list[GameResult]
    advancement: dict[str, dict[str, float]]


class ResetResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Bracket state manager
# ---------------------------------------------------------------------------
class BracketState:
    """Manages the tournament bracket state across rounds."""

    def __init__(self):
        self.teams_db: dict[str, TeamData] = {}  # raw db index
        self.team_data: dict[str, TeamData] = {}  # canonical bracket name -> TeamData
        self.seed_map: dict[str, int] = {}         # canonical name -> seed
        self.region_map: dict[str, str] = {}        # canonical name -> region
        self.games: dict[int, dict] = {}            # game_id -> game dict
        self.current_round: int = 0
        self.advancement: dict[str, dict[str, float]] = {}  # team -> round_key -> probability

    def load(self):
        """Load teams from DuckDB and initialize bracket."""
        self.teams_db = load_teams()
        self._resolve_bracket_teams()
        self._build_initial_bracket()

    def _resolve_bracket_teams(self):
        """Resolve all bracket team names to TeamData objects."""
        # Resolve region teams
        for region, seed_teams in REGIONS.items():
            for seed, name in seed_teams.items():
                t = find_team(name, self.teams_db)
                if not t:
                    print(f"  FALLBACK: {name} (seed {seed})")
                    t = make_fallback_team(name, seed)
                self.team_data[name] = t
                self.seed_map[name] = seed
                self.region_map[name] = region

        # Resolve First Four extra teams (not already in REGIONS)
        ff_extras = {
            "Howard": ("Midwest", 16),
            "NC State": ("West", 11),
            "Prairie View": ("South", 16),
            "Miami (OH)": ("Midwest", 11),
        }
        for name, (region, seed) in ff_extras.items():
            if name not in self.team_data:
                t = find_team(name, self.teams_db)
                if not t:
                    print(f"  FALLBACK (FF): {name} (seed {seed})")
                    t = make_fallback_team(name, seed)
                self.team_data[name] = t
                self.seed_map[name] = seed
                self.region_map[name] = region

    def _build_initial_bracket(self):
        """Build the full bracket structure with game slots."""
        self.games = {}
        self.current_round = 0
        self.advancement = {}

        # Initialize advancement tracking for all teams
        round_keys = ["ff", "r64", "r32", "s16", "e8", "f4", "champ"]
        for name in self.team_data:
            self.advancement[name] = {k: 0.0 for k in round_keys}

        # --- First Four (round 0) ---
        for ff in FIRST_FOUR:
            self.games[ff["id"]] = {
                "id": ff["id"],
                "round": 0,
                "round_name": "First Four",
                "region": ff["region"],
                "slot_seed": ff["slot_seed"],
                "team_a": ff["team_a"],
                "seed_a": ff["seed_a"],
                "team_b": ff["team_b"],
                "seed_b": ff["seed_b"],
                "score_a": None,
                "score_b": None,
                "win_prob_a": None,
                "spread": None,
                "total": None,
                "score_a_p10": None, "score_a_p50": None, "score_a_p90": None,
                "score_b_p10": None, "score_b_p50": None, "score_b_p90": None,
                "winner": None,
                "winner_seed": None,
                "simulated": False,
            }

        # --- Round of 64 (round 1): 32 games, ids 201-232 ---
        # 8 games per region, ordered East, West, South, Midwest
        game_id = 201
        region_order = ["East", "West", "South", "Midwest"]
        for region in region_order:
            seed_teams = REGIONS[region]
            for high_seed, low_seed in R64_MATCHUPS:
                high_name = seed_teams[high_seed]
                low_name = seed_teams[low_seed]

                # Check if either team is a First Four placeholder
                # First Four winners replace specific slots
                ff_placeholder_a = self._is_first_four_slot(region, high_seed)
                ff_placeholder_b = self._is_first_four_slot(region, low_seed)

                # If a seed is contested in First Four, mark as TBD until FF resolves
                team_a = None if ff_placeholder_a else high_name
                team_b = None if ff_placeholder_b else low_name
                seed_a = high_seed
                seed_b = low_seed

                self.games[game_id] = {
                    "id": game_id,
                    "round": 1,
                    "round_name": "Round of 64",
                    "region": region,
                    "team_a": team_a,
                    "seed_a": seed_a,
                    "team_b": team_b,
                    "seed_b": seed_b,
                    "score_a": None,
                    "score_b": None,
                    "win_prob_a": None,
                    "spread": None,
                    "total": None,
                    "score_a_p10": None, "score_a_p50": None, "score_a_p90": None,
                    "score_b_p10": None, "score_b_p50": None, "score_b_p90": None,
                    "winner": None,
                    "winner_seed": None,
                    "simulated": False,
                    # Track which First Four game feeds this slot
                    "ff_feed_a": self._ff_game_for_slot(region, high_seed),
                    "ff_feed_b": self._ff_game_for_slot(region, low_seed),
                }
                game_id += 1

        # --- Round of 32 (round 2): 16 games, ids 301-316 ---
        # Pair R64 games: 201+202, 203+204, ..., 231+232
        game_id = 301
        r64_ids = list(range(201, 233))
        for i in range(0, 32, 2):
            feeder_a = r64_ids[i]
            feeder_b = r64_ids[i + 1]
            region = self.games[feeder_a]["region"]
            self.games[game_id] = {
                "id": game_id,
                "round": 2,
                "round_name": "Round of 32",
                "region": region,
                "team_a": None, "seed_a": None,
                "team_b": None, "seed_b": None,
                "score_a": None, "score_b": None,
                "win_prob_a": None, "spread": None, "total": None,
                "score_a_p10": None, "score_a_p50": None, "score_a_p90": None,
                "score_b_p10": None, "score_b_p50": None, "score_b_p90": None,
                "winner": None, "winner_seed": None, "simulated": False,
                "feeder_a": feeder_a, "feeder_b": feeder_b,
            }
            game_id += 1

        # --- Sweet 16 (round 3): 8 games, ids 401-408 ---
        game_id = 401
        r32_ids = list(range(301, 317))
        for i in range(0, 16, 2):
            feeder_a = r32_ids[i]
            feeder_b = r32_ids[i + 1]
            region = self.games[feeder_a]["region"]
            self.games[game_id] = {
                "id": game_id,
                "round": 3,
                "round_name": "Sweet 16",
                "region": region,
                "team_a": None, "seed_a": None,
                "team_b": None, "seed_b": None,
                "score_a": None, "score_b": None,
                "win_prob_a": None, "spread": None, "total": None,
                "score_a_p10": None, "score_a_p50": None, "score_a_p90": None,
                "score_b_p10": None, "score_b_p50": None, "score_b_p90": None,
                "winner": None, "winner_seed": None, "simulated": False,
                "feeder_a": feeder_a, "feeder_b": feeder_b,
            }
            game_id += 1

        # --- Elite 8 (round 4): 4 games, ids 501-504 ---
        game_id = 501
        s16_ids = list(range(401, 409))
        for i in range(0, 8, 2):
            feeder_a = s16_ids[i]
            feeder_b = s16_ids[i + 1]
            region = self.games[feeder_a]["region"]
            self.games[game_id] = {
                "id": game_id,
                "round": 4,
                "round_name": "Elite 8",
                "region": region,
                "team_a": None, "seed_a": None,
                "team_b": None, "seed_b": None,
                "score_a": None, "score_b": None,
                "win_prob_a": None, "spread": None, "total": None,
                "score_a_p10": None, "score_a_p50": None, "score_a_p90": None,
                "score_b_p10": None, "score_b_p50": None, "score_b_p90": None,
                "winner": None, "winner_seed": None, "simulated": False,
                "feeder_a": feeder_a, "feeder_b": feeder_b,
            }
            game_id += 1

        # --- Final Four (round 5): 2 games, ids 601-602 ---
        # Semi 1: East champ vs West champ (501 vs 502)
        # Semi 2: South champ vs Midwest champ (503 vs 504)
        self.games[601] = {
            "id": 601,
            "round": 5,
            "round_name": "Final Four",
            "region": "Final Four",
            "team_a": None, "seed_a": None,
            "team_b": None, "seed_b": None,
            "score_a": None, "score_b": None,
            "win_prob_a": None, "spread": None, "total": None,
            "score_a_p10": None, "score_a_p50": None, "score_a_p90": None,
            "score_b_p10": None, "score_b_p50": None, "score_b_p90": None,
            "winner": None, "winner_seed": None, "simulated": False,
            "feeder_a": 501, "feeder_b": 502,
        }
        self.games[602] = {
            "id": 602,
            "round": 5,
            "round_name": "Final Four",
            "region": "Final Four",
            "team_a": None, "seed_a": None,
            "team_b": None, "seed_b": None,
            "score_a": None, "score_b": None,
            "win_prob_a": None, "spread": None, "total": None,
            "score_a_p10": None, "score_a_p50": None, "score_a_p90": None,
            "score_b_p10": None, "score_b_p50": None, "score_b_p90": None,
            "winner": None, "winner_seed": None, "simulated": False,
            "feeder_a": 503, "feeder_b": 504,
        }

        # --- Championship (round 6): 1 game, id 701 ---
        self.games[701] = {
            "id": 701,
            "round": 6,
            "round_name": "Championship",
            "region": "Championship",
            "team_a": None, "seed_a": None,
            "team_b": None, "seed_b": None,
            "score_a": None, "score_b": None,
            "win_prob_a": None, "spread": None, "total": None,
            "score_a_p10": None, "score_a_p50": None, "score_a_p90": None,
            "score_b_p10": None, "score_b_p50": None, "score_b_p90": None,
            "winner": None, "winner_seed": None, "simulated": False,
            "feeder_a": 601, "feeder_b": 602,
        }

    def _is_first_four_slot(self, region: str, seed: int) -> bool:
        """Check if a region+seed is contested in the First Four."""
        for ff in FIRST_FOUR:
            if ff["region"] == region and ff["slot_seed"] == seed:
                return True
        return False

    def _ff_game_for_slot(self, region: str, seed: int) -> Optional[int]:
        """Return the First Four game ID that feeds into a specific region+seed slot."""
        for ff in FIRST_FOUR:
            if ff["region"] == region and ff["slot_seed"] == seed:
                return ff["id"]
        return None

    def get_round_games(self, round_num: int) -> list[dict]:
        """Get all games for a specific round."""
        return [g for g in self.games.values() if g["round"] == round_num]

    def simulate_round(self, round_num: int, num_sims: int) -> list[dict]:
        """
        Simulate all games in the specified round.

        Returns list of game results with simulation statistics.
        """
        round_games = self.get_round_games(round_num)
        if not round_games:
            raise ValueError(f"No games found for round {round_num}")

        # Validate previous round is complete (except for round 0)
        if round_num > 0:
            prev_games = self.get_round_games(round_num - 1)
            for g in prev_games:
                if not g["simulated"]:
                    raise ValueError(
                        f"Previous round ({ROUND_NAMES.get(round_num - 1, round_num - 1)}) "
                        f"is not complete. Game {g['id']} has not been simulated."
                    )

        # For rounds > 0, populate teams from feeder games
        if round_num >= 1:
            self._populate_round_teams(round_num)

        # Validate all games have both teams assigned
        for g in round_games:
            if g["team_a"] is None or g["team_b"] is None:
                raise ValueError(
                    f"Game {g['id']} is missing teams: "
                    f"team_a={g['team_a']}, team_b={g['team_b']}"
                )

        # Generate random values
        n_games = len(round_games)
        values_per_game = 5

        if HAS_GPU:
            rng = cp.random.default_rng(42 + round_num)
            all_rand = cp.asnumpy(
                rng.standard_normal(
                    (num_sims, n_games, values_per_game), dtype=cp.float32
                )
            )
        else:
            rng = np.random.default_rng(42 + round_num)
            all_rand = rng.standard_normal(
                (num_sims, n_games, values_per_game)
            ).astype(np.float32)

        # Simulate each game
        results = []
        for game_idx, g in enumerate(round_games):
            ta = self.team_data.get(g["team_a"])
            tb = self.team_data.get(g["team_b"])

            if not ta or not tb:
                raise ValueError(
                    f"Game {g['id']}: TeamData not found for "
                    f"{g['team_a']} or {g['team_b']}"
                )

            scores_a = np.zeros(num_sims, dtype=np.float32)
            scores_b = np.zeros(num_sims, dtype=np.float32)
            wins_a = 0

            for sim in range(num_sims):
                rv = all_rand[sim, game_idx]
                sa, sb, winner_code, _, _, _ = calc_game(ta, tb, rv)
                scores_a[sim] = sa
                scores_b[sim] = sb
                if winner_code == "A":
                    wins_a += 1

            # Aggregate statistics
            mean_a = float(np.mean(scores_a))
            mean_b = float(np.mean(scores_b))
            win_prob_a = wins_a / num_sims
            spread = mean_a - mean_b
            total = mean_a + mean_b

            p10_a = float(np.percentile(scores_a, 10))
            p50_a = float(np.percentile(scores_a, 50))
            p90_a = float(np.percentile(scores_a, 90))
            p10_b = float(np.percentile(scores_b, 10))
            p50_b = float(np.percentile(scores_b, 50))
            p90_b = float(np.percentile(scores_b, 90))

            # Pick winner (team with >50% win probability)
            if win_prob_a >= 0.5:
                winner_name = g["team_a"]
                winner_seed = g["seed_a"]
            else:
                winner_name = g["team_b"]
                winner_seed = g["seed_b"]

            # Update game state
            g["score_a"] = round(mean_a, 1)
            g["score_b"] = round(mean_b, 1)
            g["win_prob_a"] = round(win_prob_a, 4)
            g["spread"] = round(spread, 1)
            g["total"] = round(total, 1)
            g["score_a_p10"] = round(p10_a, 0)
            g["score_a_p50"] = round(p50_a, 0)
            g["score_a_p90"] = round(p90_a, 0)
            g["score_b_p10"] = round(p10_b, 0)
            g["score_b_p50"] = round(p50_b, 0)
            g["score_b_p90"] = round(p90_b, 0)
            g["winner"] = winner_name
            g["winner_seed"] = winner_seed
            g["simulated"] = True

            results.append(copy.deepcopy(g))

        # Update advancement probabilities with a full forward simulation
        self._update_advancement(num_sims)

        return results

    def _populate_round_teams(self, round_num: int):
        """Populate teams for a round from the winners of feeder games."""
        round_games = self.get_round_games(round_num)

        for g in round_games:
            if round_num == 1:
                # Round of 64: check for First Four feeders
                ff_a = g.get("ff_feed_a")
                ff_b = g.get("ff_feed_b")

                if ff_a is not None:
                    ff_game = self.games[ff_a]
                    if ff_game["simulated"] and ff_game["winner"]:
                        g["team_a"] = ff_game["winner"]
                        g["seed_a"] = ff_game["winner_seed"]

                if ff_b is not None:
                    ff_game = self.games[ff_b]
                    if ff_game["simulated"] and ff_game["winner"]:
                        g["team_b"] = ff_game["winner"]
                        g["seed_b"] = ff_game["winner_seed"]
            else:
                # Rounds 2-6: get winners from feeder games
                feeder_a = g.get("feeder_a")
                feeder_b = g.get("feeder_b")

                if feeder_a is not None and feeder_a in self.games:
                    fa = self.games[feeder_a]
                    if fa["simulated"] and fa["winner"]:
                        g["team_a"] = fa["winner"]
                        g["seed_a"] = fa["winner_seed"]

                if feeder_b is not None and feeder_b in self.games:
                    fb = self.games[feeder_b]
                    if fb["simulated"] and fb["winner"]:
                        g["team_b"] = fb["winner"]
                        g["seed_b"] = fb["winner_seed"]

    def _update_advancement(self, num_sims: int):
        """
        Run a forward simulation from the current bracket state to compute
        advancement probabilities for all remaining teams.

        For already-eliminated teams, advancement stays at 0 for future rounds.
        For teams that have already won a round, that round is set to 1.0.
        """
        round_keys = ["ff", "r64", "r32", "s16", "e8", "f4", "champ"]

        # Reset all advancement
        for name in self.team_data:
            self.advancement[name] = {k: 0.0 for k in round_keys}

        # Mark teams that are in the bracket (everyone starts with some probability)
        # First, mark completed rounds
        for g in self.games.values():
            if not g["simulated"]:
                continue
            rnd = g["round"]
            winner = g["winner"]
            if winner and winner in self.advancement:
                # The winner advanced past this round
                round_key = round_keys[rnd]
                self.advancement[winner][round_key] = 1.0

        # For future rounds, run forward Monte Carlo from current state
        # Find the next unplayed round
        max_simulated_round = -1
        for g in self.games.values():
            if g["simulated"] and g["round"] > max_simulated_round:
                max_simulated_round = g["round"]

        if max_simulated_round >= 6:
            # Tournament is complete
            return

        # Forward simulate from the current winners
        # Use a smaller number of sims for advancement (speed)
        adv_sims = min(num_sims, 50_000)
        adv_counts: dict[str, dict[str, int]] = {
            name: {k: 0 for k in round_keys} for name in self.team_data
        }

        # Seed for reproducibility
        rng = np.random.default_rng(123)

        for sim in range(adv_sims):
            # Start with current winners for all simulated games
            sim_winners: dict[int, tuple[str, int]] = {}

            for g in self.games.values():
                if g["simulated"] and g["winner"]:
                    sim_winners[g["id"]] = (g["winner"], g["winner_seed"])

            # Simulate remaining rounds
            for rnd in range(max_simulated_round + 1, 7):
                round_games = self.get_round_games(rnd)
                for g in round_games:
                    if g["simulated"]:
                        continue

                    # Determine teams for this game
                    team_a, seed_a = None, None
                    team_b, seed_b = None, None

                    if rnd == 0:
                        team_a = g["team_a"]
                        seed_a = g["seed_a"]
                        team_b = g["team_b"]
                        seed_b = g["seed_b"]
                    elif rnd == 1:
                        # R64: check First Four feeders
                        ff_a = g.get("ff_feed_a")
                        ff_b = g.get("ff_feed_b")

                        if ff_a and ff_a in sim_winners:
                            team_a, seed_a = sim_winners[ff_a]
                        else:
                            team_a = g["team_a"]
                            seed_a = g["seed_a"]

                        if ff_b and ff_b in sim_winners:
                            team_b, seed_b = sim_winners[ff_b]
                        else:
                            team_b = g["team_b"]
                            seed_b = g["seed_b"]
                    else:
                        feeder_a = g.get("feeder_a")
                        feeder_b = g.get("feeder_b")

                        if feeder_a and feeder_a in sim_winners:
                            team_a, seed_a = sim_winners[feeder_a]
                        if feeder_b and feeder_b in sim_winners:
                            team_b, seed_b = sim_winners[feeder_b]

                    if not team_a or not team_b:
                        continue

                    ta = self.team_data.get(team_a)
                    tb = self.team_data.get(team_b)
                    if not ta or not tb:
                        continue

                    rv = rng.standard_normal(5).astype(np.float32)
                    _, _, winner_code, _, _, _ = calc_game(ta, tb, rv)

                    if winner_code == "A":
                        w_name, w_seed = team_a, seed_a
                    else:
                        w_name, w_seed = team_b, seed_b

                    sim_winners[g["id"]] = (w_name, w_seed)

                    # Count advancement
                    rk = round_keys[rnd]
                    if w_name in adv_counts:
                        adv_counts[w_name][rk] += 1

        # Convert counts to probabilities
        for name in self.team_data:
            for rk in round_keys:
                # If the round is already decided, keep 1.0 or 0.0
                if self.advancement[name][rk] == 1.0:
                    continue
                count = adv_counts[name].get(rk, 0)
                self.advancement[name][rk] = round(count / adv_sims, 4) if adv_sims > 0 else 0.0

    def get_state_response(self) -> BracketStateResponse:
        """Build the full bracket state response."""
        game_results = []
        for gid in sorted(self.games.keys()):
            g = self.games[gid]
            game_results.append(GameResult(
                id=g["id"],
                round=g["round"],
                round_name=g.get("round_name", ROUND_NAMES.get(g["round"], "")),
                region=g.get("region", ""),
                team_a=g["team_a"],
                seed_a=g["seed_a"],
                team_b=g["team_b"],
                seed_b=g["seed_b"],
                score_a=g["score_a"],
                score_b=g["score_b"],
                win_prob_a=g["win_prob_a"],
                spread=g["spread"],
                total=g["total"],
                score_a_p10=g["score_a_p10"],
                score_a_p50=g["score_a_p50"],
                score_a_p90=g["score_a_p90"],
                score_b_p10=g["score_b_p10"],
                score_b_p50=g["score_b_p50"],
                score_b_p90=g["score_b_p90"],
                winner=g["winner"],
                winner_seed=g["winner_seed"],
                simulated=g["simulated"],
            ))
        return BracketStateResponse(
            current_round=self.current_round,
            games=game_results,
            advancement=self.advancement,
        )

    def reset(self):
        """Reset bracket to initial state (re-resolve teams, rebuild games)."""
        self._build_initial_bracket()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NCAA Tournament Bracket Simulator",
    description="Interactive round-by-round March Madness bracket simulation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bracket state
bracket: Optional[BracketState] = None


@app.on_event("startup")
async def startup_event():
    """Load team data and initialize bracket on server startup."""
    global bracket
    print("=" * 60)
    print("  NCAA Tournament Bracket Simulator - Starting up")
    print(f"  GPU available: {HAS_GPU}")
    print(f"  DB path: {DB_PATH}")
    print("=" * 60)

    bracket = BracketState()
    bracket.load()

    n_teams = len(bracket.team_data)
    n_games = len(bracket.games)
    print(f"  Loaded {n_teams} teams, {n_games} game slots")
    print("  Ready to simulate!")


@app.get("/", summary="Serve bracket dashboard")
async def serve_dashboard():
    """Serve the bracket.html dashboard as a static file."""
    if not BRACKET_HTML.exists():
        raise HTTPException(status_code=404, detail="bracket.html not found")
    return FileResponse(str(BRACKET_HTML), media_type="text/html")


@app.get("/api/bracket-state", response_model=BracketStateResponse, summary="Get current bracket state")
async def get_bracket_state():
    """Return the current state of the entire bracket including all games and advancement probabilities."""
    if bracket is None:
        raise HTTPException(status_code=503, detail="Bracket not initialized")
    return bracket.get_state_response()


@app.post("/api/simulate-round", response_model=SimulateRoundResponse, summary="Simulate a round")
async def simulate_round(req: SimulateRoundRequest):
    """
    Run Monte Carlo simulation for all games in the requested round.

    Validates that the previous round is complete before proceeding.
    Advances winners into the next round's matchups.
    Returns game results with score distributions and updated advancement probabilities.
    """
    if bracket is None:
        raise HTTPException(status_code=503, detail="Bracket not initialized")

    round_num = req.round
    num_sims = req.num_sims

    # Check that this round hasn't already been simulated
    round_games = bracket.get_round_games(round_num)
    if not round_games:
        raise HTTPException(
            status_code=400,
            detail=f"No games found for round {round_num} ({ROUND_NAMES.get(round_num, 'Unknown')})"
        )

    already_simulated = all(g["simulated"] for g in round_games)
    if already_simulated:
        raise HTTPException(
            status_code=400,
            detail=f"Round {round_num} ({ROUND_NAMES.get(round_num, 'Unknown')}) has already been simulated. "
                   f"Use POST /api/reset to start over."
        )

    # Run simulation
    start_ms = time.perf_counter()
    try:
        results = bracket.simulate_round(round_num, num_sims)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    runtime_ms = (time.perf_counter() - start_ms) * 1000

    # Update current_round tracker
    bracket.current_round = round_num + 1

    # Build response
    game_results = []
    for g in results:
        game_results.append(GameResult(
            id=g["id"],
            round=g["round"],
            round_name=g.get("round_name", ROUND_NAMES.get(g["round"], "")),
            region=g.get("region", ""),
            team_a=g["team_a"],
            seed_a=g["seed_a"],
            team_b=g["team_b"],
            seed_b=g["seed_b"],
            score_a=g["score_a"],
            score_b=g["score_b"],
            win_prob_a=g["win_prob_a"],
            spread=g["spread"],
            total=g["total"],
            score_a_p10=g["score_a_p10"],
            score_a_p50=g["score_a_p50"],
            score_a_p90=g["score_a_p90"],
            score_b_p10=g["score_b_p10"],
            score_b_p50=g["score_b_p50"],
            score_b_p90=g["score_b_p90"],
            winner=g["winner"],
            winner_seed=g["winner_seed"],
            simulated=g["simulated"],
        ))

    return SimulateRoundResponse(
        round=round_num,
        round_name=ROUND_NAMES.get(round_num, f"Round {round_num}"),
        num_sims=num_sims,
        runtime_ms=round(runtime_ms, 1),
        gpu_used=HAS_GPU,
        games=game_results,
        advancement=bracket.advancement,
    )


@app.post("/api/reset", response_model=ResetResponse, summary="Reset bracket")
async def reset_bracket():
    """Reset the bracket to its initial state with no games simulated."""
    if bracket is None:
        raise HTTPException(status_code=503, detail="Bracket not initialized")

    bracket.reset()
    return ResetResponse(
        status="ok",
        message="Bracket reset to initial state. All simulation results cleared.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.api.bracket_api:app",
        host="0.0.0.0",
        port=2502,
        reload=False,
        log_level="info",
    )
