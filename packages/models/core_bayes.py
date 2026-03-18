"""
Core Bayesian prediction model for NCAA basketball.

Implements a hierarchical model for spread/total prediction using:
- Tempo-based possession modeling
- KenPom-style adjusted efficiencies
- Hierarchical home court advantage
- Market-aware priors (optional anchoring)
- Uncertainty quantification

This is a simplified baseline model that can be extended with PyMC
for full Bayesian inference.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import stats
import structlog

from packages.common.config import get_settings
from packages.common.schemas import FeatureRow, PredictionRow
from packages.features.adjusted_efficiency import project_game_score

logger = structlog.get_logger()

# Model version
MODEL_VERSION = "v0.1.0"

# Typical standard deviations for NCAAB
SPREAD_STD = 11.0  # Spread outcome std dev
TOTAL_STD = 13.0  # Total outcome std dev


@dataclass
class ModelConfig:
    """Configuration for the prediction model."""

    home_court_advantage: float = 3.5
    league_avg_efficiency: float = 100.0
    spread_std: float = SPREAD_STD
    total_std: float = TOTAL_STD
    market_anchor_weight: float = 0.0  # 0 = pure model, 1 = pure market
    version: str = MODEL_VERSION


@dataclass
class GamePrediction:
    """Raw prediction output before formatting."""

    game_id: int
    home_score: float
    away_score: float
    spread: float
    total: float
    possessions: float
    home_win_prob: float
    spread_std: float
    total_std: float


class BaselinePredictor:
    """
    Baseline predictor using tempo-based efficiency model.

    This model projects scores using:
    1. Expected possessions from tempo matchup
    2. Expected PPP from efficiency matchup
    3. Home court adjustment
    4. Gaussian uncertainty model
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize predictor.

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        logger.info(
            "Initialized BaselinePredictor",
            version=self.config.version,
            hca=self.config.home_court_advantage,
        )

    def predict_game(
        self,
        home_adj_off: float,
        home_adj_def: float,
        home_adj_tempo: float,
        away_adj_off: float,
        away_adj_def: float,
        away_adj_tempo: float,
        game_id: int,
        is_neutral: bool = False,
        market_spread: Optional[float] = None,
        market_total: Optional[float] = None,
    ) -> GamePrediction:
        """
        Generate prediction for a single game.

        Args:
            home_adj_off: Home team adjusted offensive efficiency
            home_adj_def: Home team adjusted defensive efficiency
            home_adj_tempo: Home team adjusted tempo
            away_adj_off: Away team adjusted offensive efficiency
            away_adj_def: Away team adjusted defensive efficiency
            away_adj_tempo: Away team adjusted tempo
            game_id: Game identifier
            is_neutral: Whether game is at neutral site
            market_spread: Optional market spread for anchoring
            market_total: Optional market total for anchoring

        Returns:
            GamePrediction dataclass
        """
        # Project scores using efficiency model
        home_score, away_score, possessions = project_game_score(
            home_adj_off=home_adj_off,
            home_adj_def=home_adj_def,
            home_adj_tempo=home_adj_tempo,
            away_adj_off=away_adj_off,
            away_adj_def=away_adj_def,
            away_adj_tempo=away_adj_tempo,
            home_court_advantage=self.config.home_court_advantage,
            is_neutral=is_neutral,
            league_avg=self.config.league_avg_efficiency,
        )

        # Calculate spread and total
        model_spread = home_score - away_score
        model_total = home_score + away_score

        # Apply market anchoring if available and configured
        if market_spread is not None and self.config.market_anchor_weight > 0:
            spread = (
                (1 - self.config.market_anchor_weight) * model_spread
                + self.config.market_anchor_weight * market_spread
            )
        else:
            spread = model_spread

        if market_total is not None and self.config.market_anchor_weight > 0:
            total = (
                (1 - self.config.market_anchor_weight) * model_total
                + self.config.market_anchor_weight * market_total
            )
        else:
            total = model_total

        # Recalculate scores from spread/total
        final_home = (total + spread) / 2
        final_away = (total - spread) / 2

        # Calculate win probability
        home_win_prob = stats.norm.cdf(spread / self.config.spread_std)

        return GamePrediction(
            game_id=game_id,
            home_score=final_home,
            away_score=final_away,
            spread=spread,
            total=total,
            possessions=possessions,
            home_win_prob=home_win_prob,
            spread_std=self.config.spread_std,
            total_std=self.config.total_std,
        )

    def predict_from_features(
        self,
        features: FeatureRow,
        market_spread: Optional[float] = None,
        market_total: Optional[float] = None,
    ) -> GamePrediction:
        """
        Generate prediction from FeatureRow.

        Args:
            features: Engineered features for the matchup
            market_spread: Optional market spread
            market_total: Optional market total

        Returns:
            GamePrediction dataclass
        """
        return self.predict_game(
            home_adj_off=features.home_adj_off_eff,
            home_adj_def=features.home_adj_def_eff,
            home_adj_tempo=features.home_adj_tempo,
            away_adj_off=features.away_adj_off_eff,
            away_adj_def=features.away_adj_def_eff,
            away_adj_tempo=features.away_adj_tempo,
            game_id=features.game_id,
            is_neutral=features.is_neutral,
            market_spread=market_spread or features.current_spread,
            market_total=market_total,
        )

    def generate_prediction_intervals(
        self,
        prediction: GamePrediction,
        alpha_levels: list[float] = [0.50, 0.80, 0.95],
    ) -> dict[str, dict[float, tuple[float, float]]]:
        """
        Generate prediction intervals for spread and total.

        Uses Gaussian model for uncertainty.

        Args:
            prediction: GamePrediction object
            alpha_levels: Confidence levels (e.g., [0.50, 0.80, 0.95])

        Returns:
            Dictionary with spread and total intervals at each level
        """
        intervals = {"spread": {}, "total": {}}

        for alpha in alpha_levels:
            # Z-score for this confidence level
            z = stats.norm.ppf((1 + alpha) / 2)

            # Spread intervals
            spread_lower = prediction.spread - z * prediction.spread_std
            spread_upper = prediction.spread + z * prediction.spread_std
            intervals["spread"][alpha] = (spread_lower, spread_upper)

            # Total intervals
            total_lower = prediction.total - z * prediction.total_std
            total_upper = prediction.total + z * prediction.total_std
            intervals["total"][alpha] = (total_lower, total_upper)

        return intervals

    def to_prediction_row(
        self,
        prediction: GamePrediction,
        market_spread: Optional[float] = None,
        market_total: Optional[float] = None,
    ) -> PredictionRow:
        """
        Convert GamePrediction to PredictionRow schema.

        Args:
            prediction: GamePrediction object
            market_spread: Optional market spread for edge calculation
            market_total: Optional market total for edge calculation

        Returns:
            PredictionRow object
        """
        # Generate intervals
        intervals = self.generate_prediction_intervals(prediction)

        # Calculate market edge
        edge_spread = None
        edge_total = None
        if market_spread is not None:
            edge_spread = prediction.spread - market_spread
        if market_total is not None:
            edge_total = prediction.total - market_total

        # Determine recommendation
        recommended_side = None
        recommended_units = None
        confidence_rating = None

        if edge_spread is not None:
            if abs(edge_spread) >= 3.0:
                recommended_side = "home_spread" if edge_spread > 0 else "away_spread"
                recommended_units = min(abs(edge_spread) / 3.0, 3.0)
                confidence_rating = (
                    "high" if abs(edge_spread) >= 5.0 else "medium" if abs(edge_spread) >= 3.0 else "low"
                )
            elif edge_total is not None and abs(edge_total) >= 4.0:
                recommended_side = "over" if edge_total > 0 else "under"
                recommended_units = min(abs(edge_total) / 4.0, 3.0)
                confidence_rating = (
                    "high" if abs(edge_total) >= 6.0 else "medium" if abs(edge_total) >= 4.0 else "low"
                )
            else:
                recommended_side = "no_bet"
                recommended_units = 0.0
                confidence_rating = "low"

        return PredictionRow(
            game_id=prediction.game_id,
            prediction_timestamp=datetime.utcnow(),
            model_version=self.config.version,
            proj_home_score=prediction.home_score,
            proj_away_score=prediction.away_score,
            proj_spread=prediction.spread,
            proj_total=prediction.total,
            proj_possessions=prediction.possessions,
            home_win_prob=prediction.home_win_prob,
            spread_ci_50_lower=intervals["spread"][0.50][0],
            spread_ci_50_upper=intervals["spread"][0.50][1],
            spread_ci_80_lower=intervals["spread"][0.80][0],
            spread_ci_80_upper=intervals["spread"][0.80][1],
            spread_ci_95_lower=intervals["spread"][0.95][0],
            spread_ci_95_upper=intervals["spread"][0.95][1],
            total_ci_50_lower=intervals["total"][0.50][0],
            total_ci_50_upper=intervals["total"][0.50][1],
            total_ci_80_lower=intervals["total"][0.80][0],
            total_ci_80_upper=intervals["total"][0.80][1],
            total_ci_95_lower=intervals["total"][0.95][0],
            total_ci_95_upper=intervals["total"][0.95][1],
            market_spread=market_spread,
            edge_vs_market_spread=edge_spread,
            market_total=market_total,
            edge_vs_market_total=edge_total,
            recommended_side=recommended_side,
            recommended_units=recommended_units,
            confidence_rating=confidence_rating,
        )


def create_default_predictor() -> BaselinePredictor:
    """
    Create predictor with default settings.

    Returns:
        BaselinePredictor instance
    """
    settings = get_settings()
    config = ModelConfig(
        home_court_advantage=settings.home_court_advantage,
        league_avg_efficiency=settings.league_avg_efficiency,
        version=settings.model_version,
    )
    return BaselinePredictor(config)


def batch_predict(
    predictor: BaselinePredictor,
    features_list: list[FeatureRow],
    market_spreads: Optional[list[Optional[float]]] = None,
    market_totals: Optional[list[Optional[float]]] = None,
) -> list[PredictionRow]:
    """
    Generate predictions for multiple games.

    Args:
        predictor: BaselinePredictor instance
        features_list: List of FeatureRow objects
        market_spreads: Optional list of market spreads
        market_totals: Optional list of market totals

    Returns:
        List of PredictionRow objects
    """
    if market_spreads is None:
        market_spreads = [None] * len(features_list)
    if market_totals is None:
        market_totals = [None] * len(features_list)

    predictions = []
    for features, mkt_spread, mkt_total in zip(features_list, market_spreads, market_totals):
        pred = predictor.predict_from_features(
            features=features,
            market_spread=mkt_spread,
            market_total=mkt_total,
        )
        pred_row = predictor.to_prediction_row(
            prediction=pred,
            market_spread=mkt_spread,
            market_total=mkt_total,
        )
        predictions.append(pred_row)

    logger.info("Batch prediction complete", n_games=len(predictions))
    return predictions
