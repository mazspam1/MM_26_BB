"""
Optimized backtest with probabilistic metrics (Brier, log loss, calibration).

Key optimizations:
- Compute ratings weekly instead of daily (major speedup)
- Cache ratings by week
- Add Brier score, log loss, calibration-by-bucket
- Add baseline comparison (always-predict-favorite)
"""

import sys
from datetime import date, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from packages.common.database import get_connection, init_database
from packages.features.kenpom_ratings import calculate_adjusted_ratings, TeamRatings
from packages.features.conference_hca import get_conference_hca_map
from packages.models.enhanced_predictor import create_enhanced_predictor
from packages.eval.metrics import (
    brier_score,
    log_loss,
    calibration_by_bucket,
    mean_absolute_error,
    root_mean_squared_error,
    interval_coverage,
)

logger = structlog.get_logger()


def run_optimized_backtest(
    start_date: date,
    end_date: date,
    rating_update_days: int = 7,  # Recompute ratings weekly
):
    """Run backtest with weekly rating updates and full probabilistic metrics."""
    init_database()

    with get_connection() as conn:
        games_df = conn.execute(
            """
            SELECT game_id, CAST(game_date AS DATE) AS game_date, game_datetime,
                   home_team_id, away_team_id, neutral_site,
                   home_score, away_score, conference_game, season_phase
            FROM games
            WHERE CAST(game_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND status = 'final'
              AND home_score IS NOT NULL
            ORDER BY CAST(game_date AS DATE)
        """,
            (start_date.isoformat(), end_date.isoformat()),
        ).fetchdf()

        team_stats = conn.execute("SELECT * FROM team_game_stats").fetchdf()
        conf_df = conn.execute(
            "SELECT team_id, conference_id, conference_name FROM team_conference_ids"
        ).fetchdf()

    if games_df.empty:
        print("No games found for date range")
        return

    games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.date
    print(
        f"Games: {len(games_df)} from {games_df['game_date'].min()} to {games_df['game_date'].max()}"
    )

    hca_map = get_conference_hca_map(conf_df) if not conf_df.empty else None
    predictor = create_enhanced_predictor(use_saved_calibration=False)

    # Cache ratings by week
    ratings_cache = {}
    current_week_start = None

    # Results storage
    pred_spreads, pred_totals = [], []
    actual_spreads, actual_totals = [], []
    pred_win_probs, actual_outcomes = [], []

    spread_cis_50_lower, spread_cis_50_upper = [], []
    spread_cis_80_lower, spread_cis_80_upper = [], []
    spread_cis_95_lower, spread_cis_95_upper = [], []
    total_cis_50_lower, total_cis_50_upper = [], []
    total_cis_80_lower, total_cis_80_upper = [], []
    total_cis_95_lower, total_cis_95_upper = [], []

    n_games_processed = 0
    n_games_skipped = 0

    for _, game in games_df.iterrows():
        game_date = game["game_date"]

        # Get ratings (cached by week)
        week_start = game_date - timedelta(days=game_date.weekday())
        if week_start not in ratings_cache:
            as_of = game_date - timedelta(days=1)
            try:
                ratings = calculate_adjusted_ratings(
                    team_stats=team_stats,
                    as_of_date=as_of,
                    use_recency_weights=True,
                    conference_hca=hca_map,
                )
                ratings_cache[week_start] = ratings
            except Exception as e:
                logger.warning("Rating computation failed", date=str(game_date), error=str(e))
                n_games_skipped += 1
                continue

        ratings = ratings_cache[week_start]
        home_id = int(game["home_team_id"])
        away_id = int(game["away_team_id"])

        if home_id not in ratings or away_id not in ratings:
            n_games_skipped += 1
            continue

        # Predict
        try:
            pred = predictor.predict_game(
                home_ratings=ratings[home_id],
                away_ratings=ratings[away_id],
                game_id=int(game["game_id"]),
                is_neutral=bool(game["neutral_site"]),
                home_rest_days=2,
                away_rest_days=2,
            )
        except Exception as e:
            n_games_skipped += 1
            continue

        # Store results
        actual_spread = float(game["home_score"] - game["away_score"])
        actual_total = float(game["home_score"] + game["away_score"])
        actual_win = 1.0 if actual_spread > 0 else 0.0

        pred_spreads.append(pred.spread)
        pred_totals.append(pred.total)
        actual_spreads.append(actual_spread)
        actual_totals.append(actual_total)
        pred_win_probs.append(pred.home_win_prob)
        actual_outcomes.append(actual_win)

        # CIs
        spread_cis_50_lower.append(pred.spread - stats.norm.ppf(0.75) * pred.spread_std)
        spread_cis_50_upper.append(pred.spread + stats.norm.ppf(0.75) * pred.spread_std)
        spread_cis_80_lower.append(pred.spread - stats.norm.ppf(0.90) * pred.spread_std)
        spread_cis_80_upper.append(pred.spread + stats.norm.ppf(0.90) * pred.spread_std)
        spread_cis_95_lower.append(pred.spread - stats.norm.ppf(0.975) * pred.spread_std)
        spread_cis_95_upper.append(pred.spread + stats.norm.ppf(0.975) * pred.spread_std)

        total_cis_50_lower.append(pred.total - stats.norm.ppf(0.75) * pred.total_std)
        total_cis_50_upper.append(pred.total + stats.norm.ppf(0.75) * pred.total_std)
        total_cis_80_lower.append(pred.total - stats.norm.ppf(0.90) * pred.total_std)
        total_cis_80_upper.append(pred.total + stats.norm.ppf(0.90) * pred.total_std)
        total_cis_95_lower.append(pred.total - stats.norm.ppf(0.975) * pred.total_std)
        total_cis_95_upper.append(pred.total + stats.norm.ppf(0.975) * pred.total_std)

        n_games_processed += 1

    # Convert to arrays
    pred_spreads = np.array(pred_spreads)
    pred_totals = np.array(pred_totals)
    actual_spreads = np.array(actual_spreads)
    actual_totals = np.array(actual_totals)
    pred_win_probs = np.array(pred_win_probs)
    actual_outcomes = np.array(actual_outcomes)

    # === ACCURACY METRICS ===
    spread_mae = mean_absolute_error(pred_spreads, actual_spreads)
    spread_rmse = root_mean_squared_error(pred_spreads, actual_spreads)
    total_mae = mean_absolute_error(pred_totals, actual_totals)
    total_rmse = root_mean_squared_error(pred_totals, actual_totals)

    # === PROBABILISTIC METRICS ===
    brier = brier_score(pred_win_probs, actual_outcomes)
    ll = log_loss(pred_win_probs, actual_outcomes)
    cal_buckets = calibration_by_bucket(pred_win_probs, actual_outcomes)

    # === CALIBRATION (CI COVERAGE) ===
    spread_50_cov = interval_coverage(
        actual_spreads, np.array(spread_cis_50_lower), np.array(spread_cis_50_upper)
    )
    spread_80_cov = interval_coverage(
        actual_spreads, np.array(spread_cis_80_lower), np.array(spread_cis_80_upper)
    )
    spread_95_cov = interval_coverage(
        actual_spreads, np.array(spread_cis_95_lower), np.array(spread_cis_95_upper)
    )
    total_50_cov = interval_coverage(
        actual_totals, np.array(total_cis_50_lower), np.array(total_cis_50_upper)
    )
    total_80_cov = interval_coverage(
        actual_totals, np.array(total_cis_80_lower), np.array(total_cis_80_upper)
    )
    total_95_cov = interval_coverage(
        actual_totals, np.array(total_cis_95_lower), np.array(total_cis_95_upper)
    )

    # === BASELINE COMPARISON (always predict the favorite by efficiency margin) ===
    baseline_correct = 0
    for i in range(len(pred_spreads)):
        baseline_pred_home = pred_spreads[i] > 0  # Model's baseline: who does it think wins?
        actual_home_won = actual_outcomes[i] > 0.5
        if baseline_pred_home == actual_home_won:
            baseline_correct += 1
    baseline_accuracy = baseline_correct / len(pred_spreads) if len(pred_spreads) > 0 else 0

    # Model accuracy
    model_correct = np.sum((pred_win_probs > 0.5) == (actual_outcomes > 0.5))
    model_accuracy = model_correct / len(pred_win_probs) if len(pred_win_probs) > 0 else 0

    # === PRINT RESULTS ===
    print()
    print("=" * 70)
    print("  BACKTEST RESULTS (Optimized, Weekly Ratings)")
    print("=" * 70)
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Games processed: {n_games_processed}")
    print(f"  Games skipped: {n_games_skipped}")
    print(f"  Rating snapshots: {len(ratings_cache)}")
    print()
    print("  ACCURACY")
    print("  " + "-" * 40)
    print(f"  Spread MAE:  {spread_mae:.2f} pts")
    print(f"  Spread RMSE: {spread_rmse:.2f} pts")
    print(f"  Total MAE:   {total_mae:.2f} pts")
    print(f"  Total RMSE:  {total_rmse:.2f} pts")
    print()
    print("  PROBABILISTIC METRICS")
    print("  " + "-" * 40)
    print(f"  Brier Score: {brier:.4f} (perfect=0, random=0.25)")
    print(f"  Log Loss:    {ll:.4f} (perfect=0, random=0.693)")
    print(f"  Model accuracy:    {model_accuracy:.1%}")
    print(f"  Baseline accuracy: {baseline_accuracy:.1%}")
    print()
    print("  CONFIDENCE INTERVAL CALIBRATION")
    print("  " + "-" * 40)
    print(f"  Spread 50% CI: {spread_50_cov:.1%} (target: 50%)")
    print(f"  Spread 80% CI: {spread_80_cov:.1%} (target: 80%)")
    print(f"  Spread 95% CI: {spread_95_cov:.1%} (target: 95%)")
    print(f"  Total  50% CI: {total_50_cov:.1%} (target: 50%)")
    print(f"  Total  80% CI: {total_80_cov:.1%} (target: 80%)")
    print(f"  Total  95% CI: {total_95_cov:.1%} (target: 95%)")
    print()

    # Calibration buckets
    print("  CALIBRATION BUCKETS (Win Probability)")
    print("  " + "-" * 40)
    print(f"  {'Bucket':<12} {'N':>5} {'Pred':>8} {'Actual':>8} {'Cal.Err':>8}")
    print("  " + "-" * 40)
    for bucket in cal_buckets:
        if bucket["n_games"] > 0:
            print(
                f"  {bucket['bucket_lower']:.1f}-{bucket['bucket_upper']:.1f}    "
                f"{bucket['n_games']:>5} "
                f"{bucket['mean_predicted']:>7.1%} "
                f"{bucket['actual_win_rate']:>7.1%} "
                f"{bucket['calibration_error']:>7.3f}"
            )
        else:
            print(
                f"  {bucket['bucket_lower']:.1f}-{bucket['bucket_upper']:.1f}    "
                f"{bucket['n_games']:>5}    (empty)"
            )

    print()
    print("=" * 70)

    return {
        "spread_mae": spread_mae,
        "total_mae": total_mae,
        "brier": brier,
        "log_loss": ll,
        "calibration": cal_buckets,
    }


if __name__ == "__main__":
    results = run_optimized_backtest(
        start_date=date(2026, 1, 15),
        end_date=date(2026, 3, 15),
        rating_update_days=7,
    )
