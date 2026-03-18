"""
Fit calibration parameters from latest backtest run and persist to disk.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from packages.common.config import get_settings
from packages.common.database import get_connection
from packages.models.calibration import ModelCalibrationParams, save_model_calibration

logger = structlog.get_logger()


def _best_anchor_weight(pred: np.ndarray, actual: np.ndarray, market: np.ndarray) -> float:
    mask = ~np.isnan(market)
    if mask.sum() < 200:
        return 0.0

    pred = pred[mask]
    actual = actual[mask]
    market = market[mask]

    weights = np.linspace(0.0, 0.5, 11)
    best_weight = 0.0
    best_mae = float("inf")
    for w in weights:
        blended = (1 - w) * pred + w * market
        mae = float(np.mean(np.abs(blended - actual)))
        if mae < best_mae:
            best_mae = mae
            best_weight = float(w)

    return best_weight


def _fit_scale_bias(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    if len(pred) < 50:
        return 1.0, 0.0, float(np.std(actual - pred)) if len(pred) > 0 else 11.0

    X = np.column_stack([pred, np.ones(len(pred))])
    coeffs, _, _, _ = np.linalg.lstsq(X, actual, rcond=None)
    scale = float(coeffs[0])
    bias = float(coeffs[1])
    residuals = actual - (scale * pred + bias)
    std = float(np.std(residuals))
    return scale, bias, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit model calibration from a backtest run")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Use only the most recent N backtest predictions for calibration",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit backtest run_id to calibrate from",
    )
    args = parser.parse_args()

    settings = get_settings()
    calibration_path = Path(settings.model_calibration_path)

    with get_connection() as conn:
        run_row = None
        if args.run_id:
            run_row = (args.run_id,)
        else:
            run_row = conn.execute(
                """
                SELECT run_id
                FROM backtest_runs
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()

        if not run_row:
            raise RuntimeError("No backtest runs found. Run scripts/run_backtest.py first.")

        run_id = run_row[0]

        df = conn.execute(
            """
            SELECT
                b.game_id,
                b.proj_spread,
                b.proj_total,
                b.market_spread,
                b.market_total,
                CAST(g.game_date AS DATE) AS game_date,
                g.home_score,
                g.away_score
            FROM backtest_predictions b
            JOIN games g ON g.game_id = b.game_id
            WHERE b.run_id = ?
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
            ORDER BY CAST(g.game_date AS DATE), b.game_id
            """,
            (run_id,),
        ).fetchdf()

    if df.empty:
        raise RuntimeError("Backtest predictions missing actuals. Check games table.")

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.tail(args.max_samples).copy()

    df["actual_spread"] = df["home_score"] - df["away_score"]
    df["actual_total"] = df["home_score"] + df["away_score"]

    pred_spread = df["proj_spread"].to_numpy(dtype=float)
    pred_total = df["proj_total"].to_numpy(dtype=float)
    actual_spread = df["actual_spread"].to_numpy(dtype=float)
    actual_total = df["actual_total"].to_numpy(dtype=float)
    market_spread = df["market_spread"].to_numpy(dtype=float)
    market_total = df["market_total"].to_numpy(dtype=float)

    spread_anchor_weight = _best_anchor_weight(pred_spread, actual_spread, market_spread)
    total_anchor_weight = _best_anchor_weight(pred_total, actual_total, market_total)

    market_spread_filled = np.where(np.isnan(market_spread), pred_spread, market_spread)
    market_total_filled = np.where(np.isnan(market_total), pred_total, market_total)
    spread_blend = (
        1 - spread_anchor_weight
    ) * pred_spread + spread_anchor_weight * market_spread_filled
    total_blend = (1 - total_anchor_weight) * pred_total + total_anchor_weight * market_total_filled

    spread_scale, spread_bias, spread_std = _fit_scale_bias(spread_blend, actual_spread)
    total_scale, total_bias, total_std = _fit_scale_bias(total_blend, actual_total)

    params = ModelCalibrationParams(
        spread_bias=spread_bias,
        spread_scale=spread_scale,
        total_bias=total_bias,
        total_scale=total_scale,
        base_spread_std=max(6.0, spread_std),
        base_total_std=max(7.0, total_std),
        market_anchor_weight_spread=spread_anchor_weight,
        market_anchor_weight_total=total_anchor_weight,
        n_samples=len(df),
        fitted_at=datetime.utcnow().isoformat(),
        source_run_id=run_id,
    )

    save_model_calibration(params, calibration_path)
    logger.info("Calibration saved", path=str(calibration_path), run_id=run_id)

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Calibration samples: {len(df)}")
    print(f"Spread scale/bias: {spread_scale:.3f} / {spread_bias:+.2f}")
    print(f"Total scale/bias: {total_scale:.3f} / {total_bias:+.2f}")
    print(f"Spread std: {params.base_spread_std:.2f}")
    print(f"Total std: {params.base_total_std:.2f}")
    print(f"Market anchor (spread/total): {spread_anchor_weight:.2f} / {total_anchor_weight:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
