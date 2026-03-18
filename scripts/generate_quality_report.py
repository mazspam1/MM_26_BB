"""
Generate a daily data-quality report for the slate.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from packages.eval.quality_report import generate_quality_report, write_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily data-quality report")
    parser.add_argument("--date", type=str, default=date.today().isoformat(), help="Target date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="", help="Optional output path")
    parser.add_argument("--min-games", type=int, default=None, help="Override min games played threshold")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date)
    report = generate_quality_report(target_date, min_games_played=args.min_games)

    output_path = Path(args.output) if args.output else None
    saved_path = write_quality_report(report, output_path=output_path)

    summary = report.get("summary", {})
    print("QUALITY REPORT")
    print(f"Date: {report.get('date')}")
    print(f"Generated: {report.get('generated_at')}")
    print(f"Ratings as of: {report.get('ratings_as_of_date')}")
    print(f"Min games played: {report.get('min_games_played')}")
    print()
    print(f"Total games: {summary.get('total_games', 0)}")
    print(f"Missing ratings: {summary.get('missing_ratings_games', 0)}")
    print(f"Low sample games: {summary.get('low_sample_games', 0)}")
    print(f"Missing lines: {summary.get('missing_lines_games', 0)}")
    print(f"Line availability rate: {summary.get('line_availability_rate', 0):.1%}")
    print(f"Ratings coverage rate: {summary.get('ratings_coverage_rate', 0):.1%}")
    print()
    print(f"Saved report: {saved_path}")


if __name__ == "__main__":
    main()
