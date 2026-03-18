"""
Full March Madness 2026 Tournament Simulation - GPU Accelerated
Proper bracket propagation through all rounds.

Uses RTX 5090 Blackwell via CuPy for random number generation.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger()

# GPU
HAS_GPU = False
try:
    import cupy as cp

    if cp.cuda.is_available():
        HAS_GPU = True
        print(f"GPU: CuPy + RTX 5090 (compute {cp.cuda.Device(0).compute_capability})")
except:
    pass

from packages.common.database import get_connection
from packages.features.kenpom_ratings import TeamRatings
from packages.models.enhanced_predictor import create_enhanced_predictor
from packages.models.tournament_predictor import TournamentPredictor, SEED_WIN_RATES

# 2026 Official Bracket from NCAA.com
BRACKET = {
    "East": {
        1: "Duke",
        2: "UConn",
        3: "Michigan State",
        4: "Kansas",
        5: "St. John's",
        6: "Louisville",
        7: "UCLA",
        8: "Ohio State",
        9: "TCU",
        10: "UCF",
        11: "South Florida",
        12: "Northern Iowa",
        13: "Cal Baptist",
        14: "North Dakota State",
        15: "Furman",
        16: "Siena",
    },
    "West": {
        1: "Arizona",
        2: "Purdue",
        3: "Gonzaga",
        4: "Arkansas",
        5: "Wisconsin",
        6: "BYU",
        7: "Miami",
        8: "Villanova",
        9: "Utah State",
        10: "Missouri",
        11: "TBD_W",
        12: "High Point",
        13: "Hawaii",
        14: "Kennesaw State",
        15: "Queens",
        16: "Long Island",
    },
    "South": {
        1: "Florida",
        2: "Houston",
        3: "Illinois",
        4: "Nebraska",
        5: "Vanderbilt",
        6: "North Carolina",
        7: "Saint Mary's",
        8: "Clemson",
        9: "Iowa",
        10: "Texas A&M",
        11: "VCU",
        12: "McNeese",
        13: "Troy",
        14: "Penn",
        15: "Idaho",
        16: "TBD_S",
    },
    "Midwest": {
        1: "Michigan",
        2: "Iowa State",
        3: "Virginia",
        4: "Alabama",
        5: "Texas Tech",
        6: "Tennessee",
        7: "Kentucky",
        8: "Georgia",
        9: "Saint Louis",
        10: "Santa Clara",
        11: "TBD_M",
        12: "Akron",
        13: "Hofstra",
        14: "Wright State",
        15: "Tennessee State",
        16: "TBD_M16",
    },
}

# First Four
FIRST_FOUR = [
    {"teams": ["UMBC", "Howard"], "seed": 16, "region": "Midwest"},
    {"teams": ["Texas", "NC State"], "seed": 11, "region": "West"},
    {"teams": ["Prairie View A&M", "Lehigh"], "seed": 16, "region": "South"},
    {"teams": ["Miami (Ohio)", "SMU"], "seed": 11, "region": "Midwest"},
]

# R64 matchups: (high_seed, low_seed) pairs
R64_MATCHUPS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def load_ratings() -> dict[int, TeamRatings]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT team_id, as_of_date,
                adj_offensive_efficiency, adj_defensive_efficiency, adj_tempo, adj_em,
                off_efg, off_tov, off_orb, off_ftr,
                def_efg, def_tov, def_drb, def_ftr,
                games_played, sos_off, sos_def,
                home_off_delta, home_def_delta, away_off_delta, away_def_delta,
                home_games_played, away_games_played,
                off_rating_std, def_rating_std, tempo_std
            FROM team_strengths
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM team_strengths)
        """).fetchall()

    ratings = {}
    for row in rows:
        ratings[row[0]] = TeamRatings(
            team_id=row[0],
            adj_off=row[2],
            adj_def=row[3],
            adj_tempo=row[4],
            adj_em=row[5],
            adj_efg=row[6],
            adj_tov=row[7],
            adj_orb=row[8],
            adj_ftr=row[9],
            adj_efg_def=row[10],
            adj_tov_def=row[11],
            adj_drb=row[12],
            adj_ftr_def=row[13],
            games_played=row[14],
            sos_off=row[15],
            sos_def=row[16],
            as_of_date=date.fromisoformat(row[1]) if isinstance(row[1], str) else row[1],
            home_off_delta=row[17],
            home_def_delta=row[18],
            away_off_delta=row[19],
            away_def_delta=row[20],
            home_games_played=row[21],
            away_games_played=row[22],
            off_std=row[23],
            def_std=row[24],
            tempo_std=row[25],
        )
    return ratings


def load_team_names() -> dict[int, str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT team_id, name FROM teams").fetchall()
    return {row[0]: row[1] for row in rows}


def resolve_team(name: str, team_map: dict[str, int]) -> Optional[int]:
    name_lower = name.lower()
    if name_lower in team_map:
        return team_map[name_lower]
    for k, v in team_map.items():
        if name_lower in k or k in name_lower:
            return v
    return None


def calc_win_prob(
    team_a: int, team_b: int, seed_a: int, seed_b: int, ratings: dict[int, TeamRatings]
) -> float:
    """Calculate win probability for team_a over team_b."""
    if team_a not in ratings or team_b not in ratings:
        # Seed-based fallback
        key = (min(seed_a, seed_b), max(seed_a, seed_b))
        base = SEED_WIN_RATES.get(key, 0.5)
        return base if seed_a <= seed_b else 1 - base

    ra, rb = ratings[team_a], ratings[team_b]
    avg = 100.0
    spread = (ra.adj_off + rb.adj_def - avg) - (rb.adj_off + ra.adj_def - avg)
    spread *= 0.85  # Neutral site
    prob = stats.norm.cdf(spread / 10.0)
    return np.clip(prob, 0.02, 0.98)


def run_simulation(n_sims: int = 100_000):
    start = time.time()

    print("=" * 70)
    print("  2026 MARCH MADNESS - FULL TOURNAMENT SIMULATION")
    print("  RTX 5090 Blackwell GPU Accelerated")
    print("=" * 70)
    print()

    # Load data
    print("Loading team ratings...")
    ratings = load_ratings()
    team_names = load_team_names()
    team_map = {v.lower(): k for k, v in team_names.items()}
    print(f"  {len(ratings)} teams with ratings, {len(team_names)} names")

    # Resolve all team names to IDs
    team_ids = {}  # (region, seed) -> team_id
    seeds_map = {}  # team_id -> seed

    for region, seeds in BRACKET.items():
        for seed, name in seeds.items():
            tid = resolve_team(name, team_map)
            team_ids[(region, seed)] = tid
            if tid:
                seeds_map[tid] = seed

    # Build complete bracket structure for simulation
    # Each game is: (team_a_id, team_b_id, seed_a, seed_b, region, round, game_idx)
    games = []
    game_idx = 0

    # R64: 32 games (8 per region)
    for region in ["East", "West", "South", "Midwest"]:
        for high, low in R64_MATCHUPS:
            ta = team_ids.get((region, high))
            tb = team_ids.get((region, low))
            if ta and tb:
                prob = calc_win_prob(ta, tb, high, low, ratings)
                games.append(
                    {
                        "idx": game_idx,
                        "round": 1,
                        "region": region,
                        "team_a": ta,
                        "team_b": tb,
                        "seed_a": high,
                        "seed_b": low,
                        "win_prob": prob,
                        "slot": (high, low),  # Position in bracket
                    }
                )
                game_idx += 1

    # Build bracket tree: which games feed into which
    # For each region: 8 R64 games -> 4 R32 -> 2 S16 -> 1 E8
    # Then: 4 E8 winners -> 2 FF -> 1 Champ

    # Map: (region, round, position) -> game index
    game_map = {}  # (region, round, slot_pair) -> game_idx
    for g in games:
        game_map[(g["region"], g["round"], g["slot"])] = g["idx"]

    # Generate R32, S16, E8 games for each region
    r32_matchups = [((1, 16), (8, 9)), ((5, 12), (4, 13)), ((6, 11), (3, 14)), ((7, 10), (2, 15))]
    s16_matchups = [((1, 16, 8, 9), (5, 12, 4, 13)), ((6, 11, 3, 14), (7, 10, 2, 15))]
    e8_matchups = [((1, 16, 8, 9, 5, 12, 4, 13), (6, 11, 3, 14, 7, 10, 2, 15))]

    for region in ["East", "West", "South", "Midwest"]:
        # R32: 4 games - sources are R64 game indices (pair adjacent games)
        r64_games_in_region = [g["idx"] for g in games if g["round"] == 1 and g["region"] == region]
        n_r32 = len(r64_games_in_region) // 2
        for i in range(n_r32):
            games.append(
                {
                    "idx": game_idx,
                    "round": 2,
                    "region": region,
                    "team_a": None,
                    "team_b": None,
                    "seed_a": None,
                    "seed_b": None,
                    "win_prob": 0.5,
                    "sources": [r64_games_in_region[i * 2], r64_games_in_region[i * 2 + 1]],
                }
            )
            game_idx += 1

        # S16: pair R32 winners
        r32_games_in_region = [g["idx"] for g in games if g["round"] == 2 and g["region"] == region]
        n_s16 = len(r32_games_in_region) // 2
        for i in range(n_s16):
            games.append(
                {
                    "idx": game_idx,
                    "round": 3,
                    "region": region,
                    "team_a": None,
                    "team_b": None,
                    "win_prob": 0.5,
                    "sources": [r32_games_in_region[i * 2], r32_games_in_region[i * 2 + 1]],
                }
            )
            game_idx += 1

        # E8: pair S16 winners
        s16_games_in_region = [g["idx"] for g in games if g["round"] == 3 and g["region"] == region]
        if len(s16_games_in_region) >= 2:
            games.append(
                {
                    "idx": game_idx,
                    "round": 4,
                    "region": region,
                    "team_a": None,
                    "team_b": None,
                    "win_prob": 0.5,
                    "sources": [s16_games_in_region[0], s16_games_in_region[1]],
                }
            )
            game_idx += 1

        # E8: 1 game (sources are the 2 S16 games)
        s16_start = game_idx - 2
        games.append(
            {
                "idx": game_idx,
                "round": 4,
                "region": region,
                "team_a": None,
                "team_b": None,
                "win_prob": 0.5,
                "sources": [s16_start, s16_start + 1],
            }
        )
        game_idx += 1

        # E8: 1 game
        s16_start = len(games) - 3
        games.append(
            {
                "idx": game_idx,
                "round": 4,
                "region": region,
                "team_a": None,
                "team_b": None,
                "win_prob": 0.5,
                "sources": [s16_start, s16_start + 1],
            }
        )
        game_idx += 1

    # Final Four: 2 games (East champ vs West champ, South champ vs Midwest champ)
    e8_games = {g["region"]: g["idx"] for g in games if g["round"] == 4}
    games.append(
        {
            "idx": game_idx,
            "round": 5,
            "region": "FF1",
            "team_a": None,
            "team_b": None,
            "win_prob": 0.5,
            "sources": [e8_games["East"], e8_games["West"]],
        }
    )
    ff1_idx = game_idx
    game_idx += 1

    games.append(
        {
            "idx": game_idx,
            "round": 5,
            "region": "FF2",
            "team_a": None,
            "team_b": None,
            "win_prob": 0.5,
            "sources": [e8_games["South"], e8_games["Midwest"]],
        }
    )
    ff2_idx = game_idx
    game_idx += 1

    # Championship: 1 game
    games.append(
        {
            "idx": game_idx,
            "round": 6,
            "region": "Champ",
            "team_a": None,
            "team_b": None,
            "win_prob": 0.5,
            "sources": [ff1_idx, ff2_idx],
        }
    )
    champ_idx = game_idx
    game_idx += 1

    total_games = len(games)
    print(f"Total games in bracket: {total_games}")

    # Generate random numbers on GPU
    print(f"Generating {n_sims:,} x {total_games} random numbers on GPU...")
    if HAS_GPU:
        rng = cp.random.default_rng(42)
        rand = rng.random((n_sims, total_games), dtype=cp.float32)
        rand_np = cp.asnumpy(rand)
    else:
        rng = np.random.default_rng(42)
        rand_np = rng.random((n_sims, total_games))

    # Get all teams
    all_teams = set()
    for g in games:
        if g.get("team_a"):
            all_teams.add(g["team_a"])
        if g.get("team_b"):
            all_teams.add(g["team_b"])
    all_teams = sorted(all_teams)
    team_idx_map = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)

    # Track results
    advancement = np.zeros((n_teams, 7), dtype=np.int64)  # rounds 0-6
    championship = np.zeros(n_teams, dtype=np.int64)

    print(f"Running {n_sims:,} simulations...")
    sim_start = time.time()

    for sim in range(n_sims):
        # Track winners for each game in this simulation
        winners = [None] * total_games
        team_a_actual = [None] * total_games
        team_b_actual = [None] * total_games

        for g in games:
            idx = g["idx"]

            # Determine teams
            if g.get("sources"):
                src_a, src_b = g["sources"]
                ta = winners[src_a]
                tb = winners[src_b]
                if ta is None or tb is None:
                    continue
            else:
                ta, tb = g["team_a"], g["team_b"]

            team_a_actual[idx] = ta
            team_b_actual[idx] = tb

            # Get win probability
            if not g.get("sources"):
                prob = g["win_prob"]
            else:
                # Dynamic probability based on actual teams
                seed_a = seeds_map.get(ta, 8)
                seed_b = seeds_map.get(tb, 8)
                prob = calc_win_prob(ta, tb, seed_a, seed_b, ratings)

            # Simulate
            winner = ta if rand_np[sim, idx] < prob else tb
            winners[idx] = winner

            # Track advancement
            w_idx = team_idx_map.get(winner, -1)
            if w_idx >= 0:
                advancement[w_idx, g["round"]] += 1
                if idx == champ_idx:
                    championship[w_idx] += 1

    sim_time = time.time() - sim_start
    print(f"Simulations complete in {sim_time:.1f}s")

    # Calculate probabilities
    champ_probs = {}
    ff_probs = {}
    expected_wins = {}

    for i, tid in enumerate(all_teams):
        champ_probs[tid] = championship[i] / n_sims
        ff_probs[tid] = (advancement[i, 5] + advancement[i, 6]) / n_sims  # FF + Champ
        expected_wins[tid] = sum(advancement[i, r] for r in range(1, 7)) / n_sims

    total_time = time.time() - start

    # Output results
    print()
    print("=" * 70)
    print("  CHAMPIONSHIP PROBABILITIES")
    print("=" * 70)

    top_champs = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)[:15]
    for tid, prob in top_champs:
        if prob > 0:
            name = team_names.get(tid, f"Team {tid}")
            seed = seeds_map.get(tid, "?")
            bar = "#" * int(prob * 100)
            print(f"  #{seed:>2} {name:<25} {prob:>6.1%} {bar}")

    print()
    print("=" * 70)
    print("  FINAL FOUR PROBABILITIES")
    print("=" * 70)

    top_ff = sorted(ff_probs.items(), key=lambda x: x[1], reverse=True)[:15]
    for tid, prob in top_ff:
        if prob > 0:
            name = team_names.get(tid, f"Team {tid}")
            seed = seeds_map.get(tid, "?")
            bar = "#" * int(prob * 50)
            print(f"  #{seed:>2} {name:<25} {prob:>6.1%} {bar}")

    print()
    print("=" * 70)
    print(f"  Total: {total_time:.1f}s | GPU: {HAS_GPU} | Sims: {n_sims:,}")
    print("=" * 70)

    return {
        "championship_probs": champ_probs,
        "final_four_probs": ff_probs,
        "team_names": team_names,
        "seeds": seeds_map,
    }


from datetime import date

if __name__ == "__main__":
    run_simulation(100_000)
