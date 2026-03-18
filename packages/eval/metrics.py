"""
Evaluation metrics for NCAA basketball predictions.

Includes:
- Point spread and total accuracy (MAE, RMSE)
- Calibration metrics (interval coverage)
- CLV (Closing Line Value) calculations
- Simulated betting ROI
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for spread/total predictions."""

    n_games: int
    spread_mae: float
    spread_rmse: float
    spread_median_ae: float
    total_mae: float
    total_rmse: float
    total_median_ae: float


@dataclass
class CalibrationMetrics:
    """Calibration metrics for prediction intervals."""

    n_games: int
    spread_50_coverage: float  # % within 50% CI
    spread_80_coverage: float
    spread_95_coverage: float
    total_50_coverage: float
    total_80_coverage: float
    total_95_coverage: float


@dataclass
class CLVMetrics:
    """Closing Line Value metrics."""

    n_bets: int
    mean_spread_clv: float
    mean_total_clv: float
    spread_clv_positive_rate: float  # % with positive CLV
    total_clv_positive_rate: float
    median_spread_clv: float
    median_total_clv: float


@dataclass
class BettingMetrics:
    """Simulated betting results."""

    n_spread_bets: int
    spread_wins: int
    spread_losses: int
    spread_pushes: int
    spread_win_rate: float
    spread_roi: float

    n_total_bets: int
    total_wins: int
    total_losses: int
    total_pushes: int
    total_win_rate: float
    total_roi: float


def mean_absolute_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        predictions: Array of predicted values
        actuals: Array of actual values

    Returns:
        Mean absolute error
    """
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(np.abs(predictions - actuals)))


def root_mean_squared_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        predictions: Array of predicted values
        actuals: Array of actual values

    Returns:
        Root mean squared error
    """
    if len(predictions) == 0:
        return 0.0
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def median_absolute_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Median Absolute Error.

    More robust to outliers than MAE.

    Args:
        predictions: Array of predicted values
        actuals: Array of actual values

    Returns:
        Median absolute error
    """
    if len(predictions) == 0:
        return 0.0
    return float(np.median(np.abs(predictions - actuals)))


def calculate_accuracy_metrics(
    pred_spreads: np.ndarray,
    actual_spreads: np.ndarray,
    pred_totals: np.ndarray,
    actual_totals: np.ndarray,
) -> AccuracyMetrics:
    """
    Calculate all accuracy metrics for spread and total predictions.

    Args:
        pred_spreads: Predicted spreads
        actual_spreads: Actual spreads
        pred_totals: Predicted totals
        actual_totals: Actual totals

    Returns:
        AccuracyMetrics dataclass
    """
    n_games = len(pred_spreads)

    return AccuracyMetrics(
        n_games=n_games,
        spread_mae=mean_absolute_error(pred_spreads, actual_spreads),
        spread_rmse=root_mean_squared_error(pred_spreads, actual_spreads),
        spread_median_ae=median_absolute_error(pred_spreads, actual_spreads),
        total_mae=mean_absolute_error(pred_totals, actual_totals),
        total_rmse=root_mean_squared_error(pred_totals, actual_totals),
        total_median_ae=median_absolute_error(pred_totals, actual_totals),
    )


def interval_coverage(
    actuals: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
) -> float:
    """
    Calculate what fraction of actuals fall within confidence intervals.

    Args:
        actuals: Actual values
        ci_lower: Lower bounds of intervals
        ci_upper: Upper bounds of intervals

    Returns:
        Coverage rate (0 to 1)
    """
    if len(actuals) == 0:
        return 0.0

    within_interval = (actuals >= ci_lower) & (actuals <= ci_upper)
    return float(np.mean(within_interval))


def calculate_calibration_metrics(
    actual_spreads: np.ndarray,
    spread_50_lower: np.ndarray,
    spread_50_upper: np.ndarray,
    spread_80_lower: np.ndarray,
    spread_80_upper: np.ndarray,
    spread_95_lower: np.ndarray,
    spread_95_upper: np.ndarray,
    actual_totals: np.ndarray,
    total_50_lower: np.ndarray,
    total_50_upper: np.ndarray,
    total_80_lower: np.ndarray,
    total_80_upper: np.ndarray,
    total_95_lower: np.ndarray,
    total_95_upper: np.ndarray,
) -> CalibrationMetrics:
    """
    Calculate calibration metrics for prediction intervals.

    A well-calibrated model should have:
    - ~50% of actuals within 50% CI
    - ~80% of actuals within 80% CI
    - ~95% of actuals within 95% CI

    Args:
        All arrays of actual values and CI bounds

    Returns:
        CalibrationMetrics dataclass
    """
    return CalibrationMetrics(
        n_games=len(actual_spreads),
        spread_50_coverage=interval_coverage(actual_spreads, spread_50_lower, spread_50_upper),
        spread_80_coverage=interval_coverage(actual_spreads, spread_80_lower, spread_80_upper),
        spread_95_coverage=interval_coverage(actual_spreads, spread_95_lower, spread_95_upper),
        total_50_coverage=interval_coverage(actual_totals, total_50_lower, total_50_upper),
        total_80_coverage=interval_coverage(actual_totals, total_80_lower, total_80_upper),
        total_95_coverage=interval_coverage(actual_totals, total_95_lower, total_95_upper),
    )


def calculate_clv(
    market_line_at_bet: float,
    closing_line: float,
    bet_side: str,  # "home" or "away" for spread, "over" or "under" for total
) -> float:
    """
    Calculate Closing Line Value for a single bet.

    CLV measures how much value we got vs the closing line.
    Positive CLV = we beat the close (good).

    For spread bets:
    - If we bet home at -5.5 and it closed at -7, we got 1.5 points of CLV
    - If we bet away at +5.5 and it closed at +7, we got 1.5 points of CLV

    Args:
        market_line_at_bet: The line when we placed the bet
        closing_line: The closing line
        bet_side: Which side we bet

    Returns:
        CLV in points (positive = beat the close)
    """
    if bet_side in ("home", "under"):
        # For home spread or under, lower closing line = we got value
        return market_line_at_bet - closing_line
    else:
        # For away spread or over, higher closing line = we got value
        return closing_line - market_line_at_bet


def calculate_clv_metrics(
    our_spreads: np.ndarray,
    market_spreads_at_bet: np.ndarray,
    closing_spreads: np.ndarray,
    spread_bet_sides: list[str],
    our_totals: np.ndarray,
    market_totals_at_bet: np.ndarray,
    closing_totals: np.ndarray,
    total_bet_sides: list[str],
) -> CLVMetrics:
    """
    Calculate aggregate CLV metrics.

    Args:
        Arrays of predictions, market lines, and closing lines

    Returns:
        CLVMetrics dataclass
    """
    # Calculate individual CLVs
    spread_clvs = []
    for i in range(len(our_spreads)):
        clv = calculate_clv(market_spreads_at_bet[i], closing_spreads[i], spread_bet_sides[i])
        spread_clvs.append(clv)

    total_clvs = []
    for i in range(len(our_totals)):
        clv = calculate_clv(market_totals_at_bet[i], closing_totals[i], total_bet_sides[i])
        total_clvs.append(clv)

    spread_clvs = np.array(spread_clvs)
    total_clvs = np.array(total_clvs)

    return CLVMetrics(
        n_bets=len(spread_clvs) + len(total_clvs),
        mean_spread_clv=float(np.mean(spread_clvs)) if len(spread_clvs) > 0 else 0.0,
        mean_total_clv=float(np.mean(total_clvs)) if len(total_clvs) > 0 else 0.0,
        spread_clv_positive_rate=float(np.mean(spread_clvs > 0)) if len(spread_clvs) > 0 else 0.0,
        total_clv_positive_rate=float(np.mean(total_clvs > 0)) if len(total_clvs) > 0 else 0.0,
        median_spread_clv=float(np.median(spread_clvs)) if len(spread_clvs) > 0 else 0.0,
        median_total_clv=float(np.median(total_clvs)) if len(total_clvs) > 0 else 0.0,
    )


def evaluate_spread_bet(
    pred_spread: float,
    market_spread: float,
    actual_spread: float,
    edge_threshold: float = 0.0,
) -> Optional[tuple[str, str]]:
    """
    Evaluate a spread bet outcome.

    Args:
        pred_spread: Our projected spread (home - away, positive = home favored)
        market_spread: Market spread (home - away)
        actual_spread: Actual result
        edge_threshold: Minimum edge to place bet

    Returns:
        Tuple of (side_bet, outcome) or None if no bet
        side_bet: "home" or "away"
        outcome: "win", "loss", or "push"
    """
    edge = pred_spread - market_spread

    if abs(edge) < edge_threshold:
        return None

    if edge > 0:
        # We think home will cover more than market
        # Bet home -market_spread
        side = "home"
        covers = actual_spread > market_spread
        pushes = actual_spread == market_spread
    else:
        # We think away will cover
        # Bet away +market_spread
        side = "away"
        covers = actual_spread < market_spread
        pushes = actual_spread == market_spread

    if pushes:
        return (side, "push")
    elif covers:
        return (side, "win")
    else:
        return (side, "loss")


def evaluate_total_bet(
    pred_total: float,
    market_total: float,
    actual_total: float,
    edge_threshold: float = 0.0,
) -> Optional[tuple[str, str]]:
    """
    Evaluate a total (over/under) bet outcome.

    Args:
        pred_total: Our projected total
        market_total: Market total line
        actual_total: Actual combined score
        edge_threshold: Minimum edge to place bet

    Returns:
        Tuple of (side_bet, outcome) or None if no bet
    """
    edge = pred_total - market_total

    if abs(edge) < edge_threshold:
        return None

    if edge > 0:
        # We project higher, bet over
        side = "over"
        covers = actual_total > market_total
        pushes = actual_total == market_total
    else:
        # We project lower, bet under
        side = "under"
        covers = actual_total < market_total
        pushes = actual_total == market_total

    if pushes:
        return (side, "push")
    elif covers:
        return (side, "win")
    else:
        return (side, "loss")


def simulate_betting(
    pred_spreads: np.ndarray,
    market_spreads: np.ndarray,
    actual_spreads: np.ndarray,
    pred_totals: np.ndarray,
    market_totals: np.ndarray,
    actual_totals: np.ndarray,
    edge_threshold: float = 0.0,
    vig: float = 0.0476,  # Standard -110 vig
) -> BettingMetrics:
    """
    Simulate betting based on model predictions vs market.

    Args:
        pred_spreads: Model predicted spreads
        market_spreads: Market spread lines
        actual_spreads: Actual game spreads
        pred_totals: Model predicted totals
        market_totals: Market total lines
        actual_totals: Actual game totals
        edge_threshold: Minimum edge to place bet
        vig: Juice/vig percentage (0.0476 for -110)

    Returns:
        BettingMetrics dataclass
    """
    spread_wins = 0
    spread_losses = 0
    spread_pushes = 0

    for i in range(len(pred_spreads)):
        result = evaluate_spread_bet(
            pred_spreads[i], market_spreads[i], actual_spreads[i], edge_threshold
        )
        if result is not None:
            _, outcome = result
            if outcome == "win":
                spread_wins += 1
            elif outcome == "loss":
                spread_losses += 1
            else:
                spread_pushes += 1

    total_wins = 0
    total_losses = 0
    total_pushes = 0

    for i in range(len(pred_totals)):
        result = evaluate_total_bet(
            pred_totals[i], market_totals[i], actual_totals[i], edge_threshold
        )
        if result is not None:
            _, outcome = result
            if outcome == "win":
                total_wins += 1
            elif outcome == "loss":
                total_losses += 1
            else:
                total_pushes += 1

    # Calculate ROI
    # Win pays 1 unit * (1 - vig), loss costs 1 unit
    n_spread_bets = spread_wins + spread_losses + spread_pushes
    n_total_bets = total_wins + total_losses + total_pushes

    if n_spread_bets > 0:
        spread_profit = spread_wins * (1 - vig) - spread_losses
        spread_roi = spread_profit / (spread_wins + spread_losses) if (spread_wins + spread_losses) > 0 else 0
        spread_win_rate = spread_wins / (spread_wins + spread_losses) if (spread_wins + spread_losses) > 0 else 0
    else:
        spread_roi = 0.0
        spread_win_rate = 0.0

    if n_total_bets > 0:
        total_profit = total_wins * (1 - vig) - total_losses
        total_roi = total_profit / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
        total_win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
    else:
        total_roi = 0.0
        total_win_rate = 0.0

    return BettingMetrics(
        n_spread_bets=n_spread_bets,
        spread_wins=spread_wins,
        spread_losses=spread_losses,
        spread_pushes=spread_pushes,
        spread_win_rate=spread_win_rate,
        spread_roi=spread_roi,
        n_total_bets=n_total_bets,
        total_wins=total_wins,
        total_losses=total_losses,
        total_pushes=total_pushes,
        total_win_rate=total_win_rate,
        total_roi=total_roi,
    )


@dataclass
class ProbabilisticMetrics:
    """Probabilistic forecast quality metrics."""

    n_games: int
    brier_score: float  # Lower = better. Random = 0.25, perfect = 0.0
    log_loss: float  # Lower = better. Random ≈ 0.693
    calibration_buckets: list[dict]  # Per-decile predicted vs actual


def brier_score(predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
    """
    Brier score for binary win probability predictions.

    Brier = mean((predicted_prob - actual_outcome)^2)
    - Perfect = 0.0
    - Random (all 0.5) = 0.25
    - Always wrong = 1.0

    Args:
        predicted_probs: Win probabilities (0-1) for one side
        actual_outcomes: Binary outcomes (1 = that side won, 0 = lost)

    Returns:
        Brier score (lower is better)
    """
    if len(predicted_probs) == 0:
        return 0.25
    return float(np.mean((predicted_probs - actual_outcomes) ** 2))


def log_loss(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """
    Log loss (negative log-likelihood) for binary predictions.

    LogLoss = -mean(y*log(p) + (1-y)*log(1-p))
    - Perfect = 0.0
    - Random (all 0.5) ≈ 0.693
    - Overconfident wrong predictions → very high

    Args:
        predicted_probs: Win probabilities (0-1)
        actual_outcomes: Binary outcomes (1 = win, 0 = loss)
        eps: Clipping epsilon to prevent log(0)

    Returns:
        Log loss (lower is better)
    """
    if len(predicted_probs) == 0:
        return 0.693
    p = np.clip(predicted_probs, eps, 1 - eps)
    return float(-np.mean(actual_outcomes * np.log(p) + (1 - actual_outcomes) * np.log(1 - p)))


def calibration_by_bucket(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_buckets: int = 10,
) -> list[dict]:
    """
    Compute calibration table: for each probability bucket, what's the actual win rate?

    A well-calibrated model: when it says 70%, the team should win ~70% of the time.

    Args:
        predicted_probs: Win probabilities (0-1)
        actual_outcomes: Binary outcomes (1 = win, 0 = loss)
        n_buckets: Number of probability buckets (default 10 = deciles)

    Returns:
        List of dicts with keys: bucket_lower, bucket_upper, n_games,
        mean_predicted, actual_win_rate, calibration_error
    """
    if len(predicted_probs) == 0:
        return []

    buckets = []
    edges = np.linspace(0, 1, n_buckets + 1)

    for i in range(n_buckets):
        lower, upper = edges[i], edges[i + 1]

        # Include upper boundary in last bucket
        if i == n_buckets - 1:
            mask = (predicted_probs >= lower) & (predicted_probs <= upper)
        else:
            mask = (predicted_probs >= lower) & (predicted_probs < upper)

        n = int(np.sum(mask))
        if n == 0:
            buckets.append({
                "bucket_lower": float(lower),
                "bucket_upper": float(upper),
                "n_games": 0,
                "mean_predicted": float((lower + upper) / 2),
                "actual_win_rate": None,
                "calibration_error": None,
            })
            continue

        mean_pred = float(np.mean(predicted_probs[mask]))
        actual_rate = float(np.mean(actual_outcomes[mask]))

        buckets.append({
            "bucket_lower": float(lower),
            "bucket_upper": float(upper),
            "n_games": n,
            "mean_predicted": mean_pred,
            "actual_win_rate": actual_rate,
            "calibration_error": abs(mean_pred - actual_rate),
        })

    return buckets


def calculate_probabilistic_metrics(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_buckets: int = 10,
) -> ProbabilisticMetrics:
    """
    Calculate all probabilistic forecast quality metrics.

    Args:
        predicted_probs: Win probabilities (0-1) for the favored side
        actual_outcomes: Binary outcomes (1 = favored side won, 0 = lost)
        n_buckets: Number of calibration buckets

    Returns:
        ProbabilisticMetrics dataclass
    """
    return ProbabilisticMetrics(
        n_games=len(predicted_probs),
        brier_score=brier_score(predicted_probs, actual_outcomes),
        log_loss=log_loss(predicted_probs, actual_outcomes),
        calibration_buckets=calibration_by_bucket(predicted_probs, actual_outcomes, n_buckets),
    )


def break_even_win_rate(vig: float = 0.0476) -> float:
    """
    Calculate break-even win rate given the vig.

    For -110 lines, need ~52.4% to break even.

    Args:
        vig: Juice percentage

    Returns:
        Required win rate to break even
    """
    return 1 / (2 - vig)
