"""
Run rolling-origin backtest and persist results.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta

import structlog

from packages.eval.backtest import BacktestConfig, run_backtest

logger = structlog.get_logger()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-origin backtest")
    parser.add_argument("--start", type=str, required=False, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=False, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=30, help="Days back from yesterday")
    parser.add_argument("--edge-threshold", type=float, default=0.0, help="Min edge to bet")
    parser.add_argument("--bet-time", type=str, default="10:00", help="Bet time (HH:MM)")
    parser.add_argument(
        "--uncalibrated",
        action="store_true",
        help="Disable saved model calibration for this backtest run",
    )
    args = parser.parse_args()

    if args.start and args.end:
        start_date = _parse_date(args.start)
        end_date = _parse_date(args.end)
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=max(args.days - 1, 0))

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        edge_threshold=args.edge_threshold,
        bet_time=_parse_time(args.bet_time),
        use_saved_calibration=not args.uncalibrated,
    )

    summary = run_backtest(config, save_to_db=True)

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    print(f"Run ID: {summary.run_id}")
    print(f"Period: {summary.start_date} -> {summary.end_date}")
    print(f"Games: {summary.total_games}")
    print(f"Spread MAE: {summary.spread_mae:.2f}")
    print(f"Total MAE: {summary.total_mae:.2f}")
    print(f"Spread RMSE: {summary.spread_rmse:.2f}")
    print(f"Total RMSE: {summary.total_rmse:.2f}")
    print(
        f"Spread CI 50/80/95: {summary.spread_50_coverage:.2f} / {summary.spread_80_coverage:.2f} / {summary.spread_95_coverage:.2f}"
    )
    print(
        f"Mean CLV (Spread/Total): {summary.mean_spread_clv:+.2f} / {summary.mean_total_clv:+.2f}"
    )
    print(f"CLV+ Rate: {summary.clv_positive_rate:.2%}")
    print(f"Simulated ROI: {summary.simulated_roi:+.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
