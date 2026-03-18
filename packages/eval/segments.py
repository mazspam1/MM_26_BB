"""
Segment-aware evaluation helpers for backtests.

Defines segment classifiers and metrics aggregation for diagnostic reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from packages.eval.metrics import (
    calculate_accuracy_metrics,
    calculate_calibration_metrics,
    simulate_betting,
)
from packages.features.kenpom_ratings import TeamRatings


SPREAD_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 5.0, "0-5"),
    (5.0, 10.0, "5-10"),
    (10.0, 15.0, "10-15"),
    (15.0, float("inf"), "15+"),
)


@dataclass
class SegmentMetrics:
    segment_type: str
    segment_value: str
    total_games: int
    market_spread_count: int
    market_total_count: int
    closing_spread_count: int
    closing_total_count: int
    spread_mae: float
    spread_rmse: float
    total_mae: float
    total_rmse: float
    spread_50_coverage: float
    spread_80_coverage: float
    spread_95_coverage: float
    total_50_coverage: float
    total_80_coverage: float
    total_95_coverage: float
    mean_spread_clv: float
    mean_total_clv: float
    clv_positive_rate: float
    spread_wins: int
    spread_losses: int
    spread_pushes: int
    total_wins: int
    total_losses: int
    total_pushes: int
    simulated_roi: float


def build_rank_map(ratings: dict[int, TeamRatings]) -> dict[int, int]:
    """Build rank lookup by adjusted efficiency margin."""
    sorted_teams = sorted(ratings.values(), key=lambda r: r.adj_em, reverse=True)
    return {team.team_id: idx + 1 for idx, team in enumerate(sorted_teams)}


def classify_spread_bucket(spread: float) -> str:
    """Bucket spreads by absolute magnitude."""
    spread_abs = abs(spread)
    for low, high, label in SPREAD_BUCKETS:
        if low <= spread_abs < high:
            return label
    return "15+"


def classify_season_timing(season_phase: Optional[str]) -> str:
    """Classify games into early vs late timing."""
    if not season_phase:
        return "unknown"
    return "early" if season_phase == "early" else "late"


def classify_conference_segment(
    conference_game: Optional[bool],
    season_phase: Optional[str],
) -> str:
    """Classify conference vs non-conference (tournament is separate)."""
    if season_phase == "tournament":
        return "tournament"
    if conference_game is None:
        return "unknown"
    return "conference" if bool(conference_game) else "non_conference"


def classify_tier_matchup(
    home_rank: Optional[int],
    away_rank: Optional[int],
    total_teams: int,
    top_cut: int = 25,
    bottom_cut: int = 25,
) -> str:
    """Classify matchup tier based on rating ranks."""
    if total_teams <= 0:
        return "unknown"

    def _tier(rank: Optional[int]) -> str:
        if rank is None or rank <= 0:
            return "unknown"
        if rank <= top_cut:
            return "top"
        if rank > total_teams - bottom_cut:
            return "bottom"
        return "mid"

    home_tier = _tier(home_rank)
    away_tier = _tier(away_rank)
    if "unknown" in (home_tier, away_tier):
        return "unknown"

    if home_tier == away_tier:
        return f"{home_tier}_vs_{away_tier}"

    order = {"top": 0, "mid": 1, "bottom": 2}
    tiers = sorted([home_tier, away_tier], key=lambda t: order[t])
    return f"{tiers[0]}_vs_{tiers[1]}"


def summarize_segment(df: pd.DataFrame, edge_threshold: float) -> SegmentMetrics:
    """Compute summary metrics for a segment subset."""
    if df.empty:
        return SegmentMetrics(
            segment_type="",
            segment_value="",
            total_games=0,
            market_spread_count=0,
            market_total_count=0,
            closing_spread_count=0,
            closing_total_count=0,
            spread_mae=0.0,
            spread_rmse=0.0,
            total_mae=0.0,
            total_rmse=0.0,
            spread_50_coverage=0.0,
            spread_80_coverage=0.0,
            spread_95_coverage=0.0,
            total_50_coverage=0.0,
            total_80_coverage=0.0,
            total_95_coverage=0.0,
            mean_spread_clv=0.0,
            mean_total_clv=0.0,
            clv_positive_rate=0.0,
            spread_wins=0,
            spread_losses=0,
            spread_pushes=0,
            total_wins=0,
            total_losses=0,
            total_pushes=0,
            simulated_roi=0.0,
        )

    pred_spreads = df["pred_spread"].to_numpy(dtype=float)
    actual_spreads = df["actual_spread"].to_numpy(dtype=float)
    pred_totals = df["pred_total"].to_numpy(dtype=float)
    actual_totals = df["actual_total"].to_numpy(dtype=float)

    accuracy = calculate_accuracy_metrics(pred_spreads, actual_spreads, pred_totals, actual_totals)
    calibration = calculate_calibration_metrics(
        actual_spreads,
        df["spread_ci_50_lower"].to_numpy(dtype=float),
        df["spread_ci_50_upper"].to_numpy(dtype=float),
        df["spread_ci_80_lower"].to_numpy(dtype=float),
        df["spread_ci_80_upper"].to_numpy(dtype=float),
        df["spread_ci_95_lower"].to_numpy(dtype=float),
        df["spread_ci_95_upper"].to_numpy(dtype=float),
        actual_totals,
        df["total_ci_50_lower"].to_numpy(dtype=float),
        df["total_ci_50_upper"].to_numpy(dtype=float),
        df["total_ci_80_lower"].to_numpy(dtype=float),
        df["total_ci_80_upper"].to_numpy(dtype=float),
        df["total_ci_95_lower"].to_numpy(dtype=float),
        df["total_ci_95_upper"].to_numpy(dtype=float),
    )

    spread_mask = df["market_spread"].notna()
    total_mask = df["market_total"].notna()

    betting = simulate_betting(
        df.loc[spread_mask, "pred_spread"].to_numpy(dtype=float),
        df.loc[spread_mask, "market_spread"].to_numpy(dtype=float),
        df.loc[spread_mask, "actual_spread"].to_numpy(dtype=float),
        df.loc[total_mask, "pred_total"].to_numpy(dtype=float),
        df.loc[total_mask, "market_total"].to_numpy(dtype=float),
        df.loc[total_mask, "actual_total"].to_numpy(dtype=float),
        edge_threshold=edge_threshold,
    )

    spread_clv = df["spread_clv"].dropna().to_numpy(dtype=float)
    total_clv = df["total_clv"].dropna().to_numpy(dtype=float)

    roi_values = []
    if betting.n_spread_bets > 0:
        roi_values.append(betting.spread_roi)
    if betting.n_total_bets > 0:
        roi_values.append(betting.total_roi)
    simulated_roi = float(np.mean(roi_values)) if roi_values else 0.0

    return SegmentMetrics(
        segment_type="",
        segment_value="",
        total_games=accuracy.n_games,
        market_spread_count=int(spread_mask.sum()),
        market_total_count=int(total_mask.sum()),
        closing_spread_count=int(df["closing_spread"].notna().sum()),
        closing_total_count=int(df["closing_total"].notna().sum()),
        spread_mae=accuracy.spread_mae,
        spread_rmse=accuracy.spread_rmse,
        total_mae=accuracy.total_mae,
        total_rmse=accuracy.total_rmse,
        spread_50_coverage=calibration.spread_50_coverage,
        spread_80_coverage=calibration.spread_80_coverage,
        spread_95_coverage=calibration.spread_95_coverage,
        total_50_coverage=calibration.total_50_coverage,
        total_80_coverage=calibration.total_80_coverage,
        total_95_coverage=calibration.total_95_coverage,
        mean_spread_clv=float(np.mean(spread_clv)) if len(spread_clv) else 0.0,
        mean_total_clv=float(np.mean(total_clv)) if len(total_clv) else 0.0,
        clv_positive_rate=float(np.mean(spread_clv > 0)) if len(spread_clv) else 0.0,
        spread_wins=betting.spread_wins,
        spread_losses=betting.spread_losses,
        spread_pushes=betting.spread_pushes,
        total_wins=betting.total_wins,
        total_losses=betting.total_losses,
        total_pushes=betting.total_pushes,
        simulated_roi=simulated_roi,
    )
