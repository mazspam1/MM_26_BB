"""
Bayesian market anchoring for NCAA basketball predictions.

Instead of linear blending, uses proper Bayesian updating:
- Prior: Market line with learned variance
- Likelihood: Model prediction with learned variance
- Posterior: Optimal combination that minimizes expected loss

Reference: Bayesian approach to market efficiency
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger()


@dataclass
class BayesianAnchorResult:
    """Result of Bayesian anchoring."""

    anchored_spread: float
    anchored_total: float
    spread_posterior_std: float
    total_posterior_std: float
    model_weight_spread: float  # How much the model moved the prior
    model_weight_total: float


class BayesianMarketAnchor:
    """
    Bayesian market anchoring with learned uncertainty.

    Replaces linear blending with proper Bayesian updating:
    - Prior: N(market_line, tau^2) where tau = market uncertainty
    - Likelihood: N(model_pred, sigma^2) where sigma = model uncertainty
    - Posterior: N(posterior_mean, posterior_var) via precision-weighted average

    The key insight: when the market has low uncertainty (tau small),
    it dominates. When the model has low uncertainty (sigma small),
    it dominates. The weights emerge naturally from the math.
    """

    def __init__(
        self,
        # Market uncertainty (learned from backtests)
        market_spread_std: float = 5.0,  # Market spread uncertainty ~5 pts
        market_total_std: float = 6.0,  # Market total uncertainty ~6 pts
        # Regime-specific market confidence
        early_season_market_std_mult: float = 1.3,  # Market less confident early
        conference_play_market_std_mult: float = 0.9,  # Market more confident conf play
        tournament_market_std_mult: float = 0.85,  # Market very confident tournament
        # Minimum model weight to preserve model signal
        min_model_weight: float = 0.15,
        max_model_weight: float = 0.70,
    ):
        self.market_spread_std = market_spread_std
        self.market_total_std = market_total_std
        self.early_season_mult = early_season_market_std_mult
        self.conference_play_mult = conference_play_market_std_mult
        self.tournament_mult = tournament_market_std_mult
        self.min_model_weight = min_model_weight
        self.max_model_weight = max_model_weight

    def anchor_prediction(
        self,
        model_spread: float,
        model_spread_std: float,
        model_total: float,
        model_total_std: float,
        market_spread: Optional[float] = None,
        market_total: Optional[float] = None,
        season_phase: str = "non_conference",
    ) -> BayesianAnchorResult:
        """
        Apply Bayesian anchoring to model prediction.

        Args:
            model_spread: Model's raw spread prediction
            model_spread_std: Model's spread uncertainty
            model_total: Model's raw total prediction
            model_total_std: Model's total uncertainty
            market_spread: Market spread (if available)
            market_total: Market total (if available)
            season_phase: Current season phase for regime adjustment

        Returns:
            BayesianAnchorResult with anchored predictions
        """
        # Get regime-adjusted market uncertainty
        regime_mult = self._get_regime_multiplier(season_phase)
        market_spread_tau = self.market_spread_std * regime_mult
        market_total_tau = self.market_total_std * regime_mult

        # Anchor spread
        if market_spread is not None:
            anchored_spread, spread_post_std, spread_model_weight = self._bayesian_update(
                prior_mean=market_spread,
                prior_std=market_spread_tau,
                likelihood_mean=model_spread,
                likelihood_std=model_spread_std,
            )
        else:
            anchored_spread = model_spread
            spread_post_std = model_spread_std
            spread_model_weight = 1.0

        # Anchor total
        if market_total is not None:
            anchored_total, total_post_std, total_model_weight = self._bayesian_update(
                prior_mean=market_total,
                prior_std=market_total_tau,
                likelihood_mean=model_total,
                likelihood_std=model_total_std,
            )
        else:
            anchored_total = model_total
            total_post_std = model_total_std
            total_model_weight = 1.0

        return BayesianAnchorResult(
            anchored_spread=anchored_spread,
            anchored_total=anchored_total,
            spread_posterior_std=spread_post_std,
            total_posterior_std=total_post_std,
            model_weight_spread=spread_model_weight,
            model_weight_total=total_model_weight,
        )

    def _bayesian_update(
        self,
        prior_mean: float,
        prior_std: float,
        likelihood_mean: float,
        likelihood_std: float,
    ) -> tuple[float, float, float]:
        """
        Standard Bayesian update with Gaussian prior and likelihood.

        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * likelihood_mean) / posterior_precision
        posterior_std = 1 / sqrt(posterior_precision)

        Model weight = likelihood_precision / posterior_precision
        """
        prior_precision = 1.0 / (prior_std**2)
        likelihood_precision = 1.0 / (likelihood_std**2)
        posterior_precision = prior_precision + likelihood_precision

        posterior_mean = (
            prior_precision * prior_mean + likelihood_precision * likelihood_mean
        ) / posterior_precision
        posterior_std = np.sqrt(1.0 / posterior_precision)

        model_weight = likelihood_precision / posterior_precision

        # Clamp model weight to reasonable range
        model_weight = np.clip(model_weight, self.min_model_weight, self.max_model_weight)

        # Recalculate posterior with clamped weight
        posterior_mean = (1 - model_weight) * prior_mean + model_weight * likelihood_mean

        return float(posterior_mean), float(posterior_std), float(model_weight)

    def _get_regime_multiplier(self, season_phase: str) -> float:
        """Get market uncertainty multiplier based on season phase."""
        if season_phase == "early":
            return self.early_season_mult
        elif season_phase == "conference":
            return self.conference_play_mult
        elif season_phase == "tournament":
            return self.tournament_mult
        else:
            return 1.0

    def calibrate_from_backtest(
        self,
        model_spreads: np.ndarray,
        market_spreads: np.ndarray,
        actual_spreads: np.ndarray,
        model_spread_stds: np.ndarray,
    ) -> None:
        """
        Calibrate market uncertainty from backtest data.

        Finds the market_spread_std that minimizes the negative log-likelihood
        of actual outcomes under the Bayesian posterior.
        """
        from scipy.optimize import minimize_scalar

        def neg_log_likelihood(tau: float) -> float:
            nll = 0.0
            for i in range(len(actual_spreads)):
                posterior_mean, posterior_std, _ = self._bayesian_update(
                    prior_mean=market_spreads[i],
                    prior_std=tau,
                    likelihood_mean=model_spreads[i],
                    likelihood_std=model_spread_stds[i],
                )
                nll -= stats.norm.logpdf(actual_spreads[i], loc=posterior_mean, scale=posterior_std)
            return nll

        result = minimize_scalar(neg_log_likelihood, bounds=(1.0, 15.0), method="bounded")
        self.market_spread_std = result.x
        logger.info("Bayesian anchoring calibrated", market_spread_std=self.market_spread_std)


def create_bayesian_anchor() -> BayesianMarketAnchor:
    """Create Bayesian market anchor with default settings."""
    return BayesianMarketAnchor()
