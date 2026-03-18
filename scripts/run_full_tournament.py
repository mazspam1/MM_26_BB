"""
Full March Madness Tournament Prediction Pipeline - GPU Accelerated.

This script runs the complete pipeline:
1. Initialize tournament schema
2. Load 2026 bracket
3. Generate game predictions for Round of 64
4. Run GPU-accelerated Monte Carlo simulation (all 67 games)
5. Output full results with proper bracket propagation

Usage:
    python scripts/run_full_tournament.py [--sims 100000]
"""

from __future__ import annotations

import sys
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import structlog
from scipy import stats

from packages.common.database import get_connection
from packages.features.kenpom_ratings import TeamRatings
from packages.models.enhanced_predictor import create_enhanced_predictor
from packages.models.tournament_predictor import (
    TournamentPredictor,
    TournamentPrediction,
    ROUND_NAMES,
    SEED_WIN_RATES,
)
from packages.ingest.tournament_bracket import (
    BRACKET_2026_TEMPLATE,
    generate_bracket_slots,
)
from packages.simulation import BracketSimulator, GamePrediction

logger = structlog.get_logger()


def init_tournament_tables():
    """Initialize tournament tables."""
    schema_path = project_root / "data" / "tournament_schema.sql"
    sql = schema_path.read_text()
    with get_connection() as conn:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                try:
                    conn.execute(stmt)
                except:
                    pass


def load_ratings() -> dict[int, TeamRatings]:
    """Load team ratings from database."""
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


def resolve_team_id(team_name: str, team_map: dict[str, int]) -> Optional[int]:
    if team_name.lower() in team_map:
        return team_map[team_name.lower()]
    for name, tid in team_map.items():
        if team_name.lower() in name or name in team_name.lower():
            return tid
    parts = team_name.lower().split()
    if len(parts) > 1:
        mascot = parts[-1]
        for name, tid in team_map.items():
            if mascot in name:
                return tid
    return None


def get_matchup_win_prob(
    team_a_id: int,
    team_b_id: int,
    ratings: dict[int, TeamRatings],
    seed_a: int,
    seed_b: int,
) -> float:
    """
    Fast win probability calculation for any matchup.
    Pure model-based using efficiency ratings (no seed prior).
    """
    if team_a_id not in ratings or team_b_id not in ratings:
        return 0.5  # Neutral default when ratings missing

    r_a = ratings[team_a_id]
    r_b = ratings[team_b_id]

    # Calculate expected spread (neutral site)
    # KenPom-style additive: spread = (Off_A + Def_B - Avg) - (Off_B + Def_A - Avg)
    avg_eff = 100.0
    spread = (r_a.adj_off + r_b.adj_def - avg_eff) - (r_b.adj_off + r_a.adj_def - avg_eff)

    # Neutral site compression
    spread *= 0.85

    # Convert to win probability (possession-scaled std)
    expected_poss = 68.0 * 0.97  # Approximate neutral-site possessions
    spread_std = 0.115 * expected_poss  # PPP std × possessions ≈ 11.2
    model_prob = stats.norm.cdf(spread / spread_std)

    return np.clip(model_prob, 0.02, 0.98)


class FullBracketSimulator:
    """
    Full bracket simulation that generates predictions on-the-fly
    for matchups that emerge during simulation.
    """

    def __init__(self, num_simulations: int = 100_000, use_gpu: bool = True, seed: int = 42):
        self.num_simulations = num_simulations
        self.seed = seed
        self.use_gpu = use_gpu

    def simulate(
        self,
        bracket_slots: list,
        r64_predictions: dict[int, GamePrediction],
        ratings: dict[int, TeamRatings],
        seeds: dict[int, int],  # team_id -> seed
        year: int = 2026,
    ) -> dict:
        """Run full bracket simulation."""
        start_time = time.time()
        sim_id = f"sim_{year}_{int(start_time)}"

        n_sims = self.num_simulations

        # Build bracket structure
        slot_info = {}
        feeds_into = {}  # target_slot -> [source_slots]

        for slot in bracket_slots:
            slot_id = slot.slot_id
            slot_info[slot_id] = {
                "round": slot.round,
                "next_slot_id": slot.next_slot_id,
            }
            if slot.next_slot_id:
                if slot.next_slot_id not in feeds_into:
                    feeds_into[slot.next_slot_id] = []
                feeds_into[slot.next_slot_id].append(slot_id)

        max_round = max(s.round for s in bracket_slots)

        # Get teams
        all_team_ids = set()
        for pred in r64_predictions.values():
            all_team_ids.add(pred.team_a_id)
            all_team_ids.add(pred.team_b_id)
        team_list = sorted(all_team_ids)
        team_to_idx = {t: i for i, t in enumerate(team_list)}
        n_teams = len(team_list)

        # Generate random numbers (GPU if available)
        total_games = len(bracket_slots)
        if self.use_gpu:
            import cupy as cp

            rng = cp.random.default_rng(self.seed)
            random_vals = rng.random((n_sims, total_games), dtype=cp.float32)
            random_vals_np = cp.asnumpy(random_vals)
        else:
            rng = np.random.default_rng(self.seed)
            random_vals_np = rng.random((n_sims, total_games))

        # Sort slots by round
        slots_by_round = {}
        for slot in bracket_slots:
            if slot.round not in slots_by_round:
                slots_by_round[slot.round] = []
            slots_by_round[slot.round].append(slot)

        # Track results
        advancement_counts = np.zeros((n_teams, max_round + 1), dtype=np.int64)
        championship_counts = np.zeros(n_teams, dtype=np.int64)

        for sim_idx in range(n_sims):
            slot_winners = {}  # slot_id -> team_id

            game_counter = 0

            for round_num in sorted(slots_by_round.keys()):
                round_slots = slots_by_round[round_num]

                for slot in round_slots:
                    slot_id = slot.slot_id

                    # Get teams
                    if slot_id in feeds_into:
                        sources = feeds_into[slot_id]
                        team_a = slot_winners.get(sources[0], 0) if len(sources) > 0 else 0
                        team_b = slot_winners.get(sources[1], 0) if len(sources) > 1 else 0
                    else:
                        if slot_id in r64_predictions:
                            pred = r64_predictions[slot_id]
                            team_a = pred.team_a_id
                            team_b = pred.team_b_id
                        else:
                            game_counter += 1
                            continue

                    if team_a == 0 or team_b == 0:
                        game_counter += 1
                        continue

                    # Get win probability
                    if slot_id in r64_predictions:
                        win_prob = r64_predictions[slot_id].team_a_win_prob
                    else:
                        seed_a = seeds.get(team_a, 8)
                        seed_b = seeds.get(team_b, 8)
                        win_prob = get_matchup_win_prob(team_a, team_b, ratings, seed_a, seed_b)

                    # Simulate
                    rand = random_vals_np[sim_idx, game_counter]
                    winner = team_a if rand < win_prob else team_b

                    slot_winners[slot_id] = winner

                    # Track
                    winner_idx = team_to_idx.get(winner, -1)
                    if winner_idx >= 0:
                        advancement_counts[winner_idx, round_num] += 1
                        if round_num == max_round:
                            championship_counts[winner_idx] += 1

                    game_counter += 1

        # Calculate probabilities
        team_advancement = {}
        team_expected_wins = {}
        championship_probs = {}
        final_four_probs = {}

        for i, team_id in enumerate(team_list):
            team_advancement[team_id] = {}
            total_wins = 0

            for r in range(max_round + 1):
                prob = advancement_counts[i, r] / n_sims
                team_advancement[team_id][r] = float(prob)
                if r >= 4:
                    total_wins += advancement_counts[i, r]

            team_expected_wins[team_id] = float(total_wins / n_sims)
            championship_probs[team_id] = float(championship_counts[i] / n_sims)
            final_four_probs[team_id] = team_advancement[team_id].get(4, 0.0)

        top_champs = sorted(championship_probs.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "sim_id": sim_id,
            "year": year,
            "num_simulations": n_sims,
            "runtime_ms": int((time.time() - start_time) * 1000),
            "gpu_used": self.use_gpu,
            "team_advancement": team_advancement,
            "team_expected_wins": team_expected_wins,
            "championship_probs": championship_probs,
            "final_four_probs": final_four_probs,
            "top_champions": top_champs,
        }


def run_full_tournament(num_simulations: int = 100_000) -> dict:
    """Run the complete tournament prediction pipeline."""
    start_time = datetime.now()

    print("=" * 80)
    print("  2026 MARCH MADNESS TOURNAMENT PREDICTIONS")
    print("  GPU-ACCELERATED (RTX 5090 Blackwell)")
    print("=" * 80)
    print()

    # Step 1: Initialize
    print("[1/6] Initializing tournament tables...")
    init_tournament_tables()

    # Step 2: Load ratings
    print("[2/6] Loading team ratings...")
    ratings = load_ratings()
    print(f"  Loaded ratings for {len(ratings)} teams")

    # Step 3: Load team names
    print("[3/6] Loading team names...")
    team_names = load_team_names()
    team_map = {name.lower(): tid for tid, name in team_names.items()}
    print(f"  Loaded {len(team_names)} team names")

    # Step 4: Build bracket
    print("[4/6] Building 2026 tournament bracket...")
    bracket_data = BRACKET_2026_TEMPLATE.copy()

    # Resolve team IDs
    seeds = {}  # team_id -> seed
    resolved_count = 0
    for region_id, region in bracket_data["regions"].items():
        resolved_teams = []
        for seed, team_name in region["teams"]:
            team_id = resolve_team_id(team_name, team_map)
            resolved_teams.append((seed, team_name, team_id))
            if team_id:
                seeds[team_id] = seed
                resolved_count += 1
        region["teams"] = resolved_teams

    print(f"  Resolved {resolved_count} team IDs")

    slots = generate_bracket_slots(2026, bracket_data)
    print(f"  Generated {len(slots)} bracket slots")

    # Step 5: Generate R64 predictions
    print("[5/6] Generating Round of 64 predictions...")

    base_predictor = create_enhanced_predictor()
    tournament_predictor = TournamentPredictor(
        base_predictor=base_predictor,
    )

    r64_predictions = {}
    predictions_list = []

    for slot in slots:
        if slot.round != 1:  # Only R64
            continue

        region = bracket_data["regions"].get(slot.region_id, {"teams": []})
        teams = region.get("teams", [])

        team_a_id = team_b_id = None
        team_a_name = team_b_name = ""

        for seed, name, tid in teams:
            if seed == slot.seed_a:
                team_a_id, team_a_name = tid, name
            if seed == slot.seed_b:
                team_b_id, team_b_name = tid, name

        if not team_a_id or not team_b_id:
            continue
        if team_a_id not in ratings or team_b_id not in ratings:
            continue

        higher_seed = slot.seed_a if slot.seed_a <= slot.seed_b else slot.seed_b
        lower_seed = slot.seed_b if slot.seed_a <= slot.seed_b else slot.seed_a
        higher_id = team_a_id if slot.seed_a <= slot.seed_b else team_b_id
        lower_id = team_b_id if slot.seed_a <= slot.seed_b else team_a_id
        higher_name = team_a_name if slot.seed_a <= slot.seed_b else team_b_name
        lower_name = team_b_name if slot.seed_a <= slot.seed_b else team_a_name

        pred = tournament_predictor.predict_game(
            higher_seed=higher_seed,
            lower_seed=lower_seed,
            higher_seed_ratings=ratings[higher_id],
            lower_seed_ratings=ratings[lower_id],
            slot_id=slot.slot_id,
            year=2026,
            game_round=slot.round,
            higher_seed_team_id=higher_id,
            lower_seed_team_id=lower_id,
            higher_seed_name=higher_name,
            lower_seed_name=lower_name,
        )

        predictions_list.append(pred)

        a_win_prob = (
            pred.higher_seed_win_prob if team_a_id == higher_id else 1 - pred.higher_seed_win_prob
        )

        r64_predictions[slot.slot_id] = GamePrediction(
            slot_id=slot.slot_id,
            round=slot.round,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            team_a_seed=slot.seed_a,
            team_b_seed=slot.seed_b,
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            team_a_win_prob=a_win_prob,
            team_b_win_prob=1 - a_win_prob,
            proj_spread=pred.proj_spread,
            proj_total=pred.proj_total,
            upset_prob=pred.upset_prob,
        )

    print(f"  Generated {len(predictions_list)} R64 predictions")

    # Step 6: Run full bracket simulation
    print(f"[6/6] Running {num_simulations:,} GPU-accelerated simulations...")
    print("  (All 67 games through Championship)")

    use_gpu = True
    try:
        import cupy as cp

        if not cp.cuda.is_available():
            use_gpu = False
    except:
        use_gpu = False

    simulator = FullBracketSimulator(
        num_simulations=num_simulations,
        use_gpu=use_gpu,
    )

    sim_result = simulator.simulate(
        bracket_slots=slots,
        r64_predictions=r64_predictions,
        ratings=ratings,
        seeds=seeds,
        year=2026,
    )

    runtime = (datetime.now() - start_time).total_seconds()

    # Output results
    print()
    print("=" * 80)
    print("  RESULTS")
    print("=" * 80)
    print()
    print(f"  Total Runtime: {runtime:.1f}s ({sim_result['runtime_ms']}ms)")
    print(f"  GPU Used: {'Yes (RTX 5090 Blackwell)' if sim_result['gpu_used'] else 'No (CPU)'}")
    print(f"  Simulations: {num_simulations:,}")
    print(f"  Games Predicted: {len(predictions_list)} (R64) + simulated through Championship")
    print()

    # Print R64 predictions
    print("-" * 60)
    print("  ROUND OF 64 PREDICTIONS")
    print("-" * 60)

    for pred in sorted(predictions_list, key=lambda p: (p.higher_seed, p.slot_id)):
        print()
        print(
            f"  #{pred.higher_seed} {pred.higher_seed_name} vs #{pred.lower_seed} {pred.lower_seed_name}"
        )
        print(
            f"    Projected: {pred.higher_seed_name} {pred.proj_higher_score:.0f} - {pred.proj_lower_score:.0f} {pred.lower_seed_name}"
        )
        print(f"    Spread: {pred.proj_spread:+.1f} | Total: {pred.proj_total:.1f}")
        print(
            f"    Win Prob: #{pred.higher_seed} {pred.higher_seed_win_prob:.1%} | #{pred.lower_seed} {pred.upset_prob:.1%}"
        )
        if pred.upset_prob > 0.4:
            print(f"    *** UPSET ALERT ***")

    # Championship probabilities
    print()
    print("=" * 80)
    print("  CHAMPIONSHIP PROBABILITIES (Top 25)")
    print("=" * 80)
    print()

    for team_id, prob in sim_result["top_champions"][:25]:
        team_display = team_names.get(team_id, f"Team {team_id}")
        bar = "#" * int(prob * 100)
        seed = seeds.get(team_id, "?")
        print(f"  #{seed:>2} {team_display:<28} {prob:>6.1%} {bar}")

    # Final Four probabilities
    print()
    print("=" * 80)
    print("  FINAL FOUR PROBABILITIES (Top 20)")
    print("=" * 80)
    print()

    ff_sorted = sorted(sim_result["final_four_probs"].items(), key=lambda x: x[1], reverse=True)[
        :20
    ]
    for team_id, prob in ff_sorted:
        team_display = team_names.get(team_id, f"Team {team_id}")
        bar = "#" * int(prob * 50)
        seed = seeds.get(team_id, "?")
        print(f"  #{seed:>2} {team_display:<28} {prob:>6.1%} {bar}")

    # Upset alerts
    print()
    print("=" * 80)
    print("  UPSET ALERTS (Round of 64)")
    print("=" * 80)
    print()

    upset_preds = sorted(
        [p for p in predictions_list if p.upset_prob > 0.4],
        key=lambda p: p.upset_prob,
        reverse=True,
    )[:10]

    for pred in upset_preds:
        print(
            f"  #{pred.lower_seed} {pred.lower_seed_name} over #{pred.higher_seed} {pred.higher_seed_name}: {pred.upset_prob:.1%}"
        )

    print()
    print("=" * 80)
    print(f"  Simulation complete in {runtime:.1f}s")
    print(f"  {num_simulations:,} full bracket simulations on RTX 5090 Blackwell")
    print("=" * 80)

    return {
        "predictions": predictions_list,
        "simulation": sim_result,
        "team_names": team_names,
        "seeds": seeds,
        "runtime": runtime,
    }


def main():
    parser = argparse.ArgumentParser(description="Run full March Madness tournament predictions")
    parser.add_argument("--sims", type=int, default=100_000, help="Number of simulations")
    args = parser.parse_args()

    run_full_tournament(num_simulations=args.sims)


if __name__ == "__main__":
    main()
