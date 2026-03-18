"""
Tournament-specific prediction adjustments for March Madness.

Extends the base predictor with:
- Neutral site adjustments (no true home court)
- Round-specific effects (early round variance, late round pressure)
- Seed-based priors (historical seed performance)
- Fatigue and rest effects specific to tournament format
- Upset probability modeling

NO MOCK DATA. NO FALLBACKS. REAL PREDICTIONS ONLY.
"""

from builtins import round as builtin_round
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import structlog
from scipy import stats

from packages.models.enhanced_predictor import EnhancedPredictor, EnhancedPrediction
from packages.features.kenpom_ratings import TeamRatings

logger = structlog.get_logger()

MODEL_VERSION = "v2.0.0-tournament"

# Tournament round constants
ROUND_FIRST_FOUR = 0
ROUND_OF_64 = 1
ROUND_OF_32 = 2
ROUND_SWEET_16 = 3
ROUND_ELITE_8 = 4
ROUND_FINAL_FOUR = 5
ROUND_CHAMPIONSHIP = 6

ROUND_NAMES = {
    ROUND_FIRST_FOUR: "First Four",
    ROUND_OF_64: "Round of 64",
    ROUND_OF_32: "Round of 32",
    ROUND_SWEET_16: "Sweet 16",
    ROUND_ELITE_8: "Elite 8",
    ROUND_FINAL_FOUR: "Final Four",
    ROUND_CHAMPIONSHIP: "Championship",
}

# Historical seed performance data (based on NCAA tournament history)
# Win rates by seed matchup (higher_seed, lower_seed) -> higher_seed_win_probability
# These are calibrated from historical tournament data
SEED_WIN_RATES = {
    # Seed vs seed matchup -> higher seed win probability
    (1, 16): 0.993,  # 1 seeds almost never lose to 16 seeds
    (2, 15): 0.943,
    (3, 14): 0.852,
    (4, 13): 0.793,
    (5, 12): 0.649,
    (6, 11): 0.637,
    (7, 10): 0.609,
    (8, 9): 0.512,  # Essentially a coin flip
}

# Seed round advancement rates (probability of reaching each round)
SEED_ADVANCEMENT_RATES = {
    1: {1: 1.0, 2: 0.993, 3: 0.849, 4: 0.702, 5: 0.579, 6: 0.391},
    2: {1: 1.0, 2: 0.943, 3: 0.708, 4: 0.455, 5: 0.319, 6: 0.181},
    3: {1: 1.0, 2: 0.852, 3: 0.561, 4: 0.326, 5: 0.198, 6: 0.099},
    4: {1: 1.0, 2: 0.793, 3: 0.445, 4: 0.219, 5: 0.117, 6: 0.052},
    5: {1: 1.0, 2: 0.649, 3: 0.332, 4: 0.141, 5: 0.062, 6: 0.024},
    6: {1: 1.0, 2: 0.637, 3: 0.298, 4: 0.112, 5: 0.043, 6: 0.015},
    7: {1: 1.0, 2: 0.609, 3: 0.251, 4: 0.082, 5: 0.028, 6: 0.009},
    8: {1: 1.0, 2: 0.512, 3: 0.167, 4: 0.048, 5: 0.014, 6: 0.004},
    9: {1: 1.0, 2: 0.488, 3: 0.145, 4: 0.038, 5: 0.010, 6: 0.003},
    10: {1: 1.0, 2: 0.391, 3: 0.101, 4: 0.026, 5: 0.006, 6: 0.001},
    11: {1: 1.0, 2: 0.363, 3: 0.086, 4: 0.020, 5: 0.005, 6: 0.001},
    12: {1: 1.0, 2: 0.351, 3: 0.079, 4: 0.017, 5: 0.004, 6: 0.001},
    13: {1: 1.0, 2: 0.207, 3: 0.032, 4: 0.005, 5: 0.001, 6: 0.000},
    14: {1: 1.0, 2: 0.148, 3: 0.019, 4: 0.002, 5: 0.000, 6: 0.000},
    15: {1: 1.0, 2: 0.057, 3: 0.005, 4: 0.001, 5: 0.000, 6: 0.000},
    16: {1: 1.0, 2: 0.007, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000},
}

# Round-specific variance adjustments
# Tournament games have more variance, especially early rounds
ROUND_VARIANCE_MULTIPLIERS = {
    ROUND_FIRST_FOUR: 1.15,  # First Four teams are often less established
    ROUND_OF_64: 1.12,  # First round has most upsets
    ROUND_OF_32: 1.08,  # Still some variance
    ROUND_SWEET_16: 1.05,  # Teams settling in
    ROUND_ELITE_8: 1.03,  # Elite teams, less variance
    ROUND_FINAL_FOUR: 1.08,  # Pressure increases variance
    ROUND_CHAMPIONSHIP: 1.10,  # Championship pressure
}

# Neutral site adjustments
# Without home court, games tend to be closer
NEUTRAL_SITE_SPREAD_COMPRESSION = 0.85  # Spreads compress by ~15%
NEUTRAL_SITE_TOTAL_ADJUSTMENT = -1.5  # Slightly lower totals (no crowd energy)

# Fatigue effects in tournament (games in quick succession)
TOURNAMENT_FATIGUE_PENALTY = {
    0: 0.0,  # First game
    1: -0.3,  # Second game in 2 days
    2: -0.8,  # Third game in 4 days (Sweet 16)
    3: -1.2,  # Fourth game in 6 days (Elite 8)
    4: -0.5,  # Final Four (week break helps)
    5: -0.8,  # Championship (back-to-back)
}


@dataclass
class TournamentPrediction:
    """Tournament-specific prediction with round adjustments."""

    slot_id: int
    year: int
    game_round: int

    # Teams
    higher_seed: int
    lower_seed: int
    higher_seed_team_id: int
    lower_seed_team_id: int
    higher_seed_name: str
    lower_seed_name: str

    # Core predictions
    proj_higher_score: float
    proj_lower_score: float
    proj_spread: float  # Positive = higher seed favored
    proj_total: float
    proj_possessions: float

    # Probabilities
    higher_seed_win_prob: float
    upset_prob: float
    cover_prob: Optional[float]

    # Uncertainty
    spread_std: float
    total_std: float
    spread_ci_50: tuple[float, float]
    spread_ci_80: tuple[float, float]
    spread_ci_95: tuple[float, float]
    total_ci_50: tuple[float, float]
    total_ci_80: tuple[float, float]
    total_ci_95: tuple[float, float]
    higher_score_p10: float
    higher_score_p90: float
    lower_score_p10: float
    lower_score_p90: float

    # Tournament-specific factors
    tournament_adjustment: float
    seed_momentum_factor: float
    fatigue_factor: float
    round_variance_multiplier: float

    # Market comparison
    market_spread: Optional[float]
    market_total: Optional[float]
    edge_vs_market_spread: Optional[float]
    edge_vs_market_total: Optional[float]

    # Betting recommendation
    recommended_side: Optional[str]
    recommended_units: Optional[float]
    confidence_rating: Optional[str]

    # Metadata
    model_version: str
    prediction_time: datetime


class TournamentPredictor:
    """
    Tournament-specific predictor extending the base model.

    Applies tournament adjustments for:
    - Neutral site compression
    - Round-specific variance
    - Seed-based priors
    - Fatigue effects
    """

    def __init__(
        self,
        base_predictor: Optional[EnhancedPredictor] = None,
        use_round_adjustments: bool = True,
        use_fatigue_adjustments: bool = True,
        min_edge_threshold: float = 2.5,
    ):
        self.base_predictor = base_predictor or EnhancedPredictor()
        self.use_round_adjustments = use_round_adjustments
        self.use_fatigue_adjustments = use_fatigue_adjustments
        self.min_edge_threshold = min_edge_threshold

        logger.info(
            "Initialized TournamentPredictor",
            version=MODEL_VERSION,
            use_round_adjustments=use_round_adjustments,
        )

    def predict_game(
        self,
        higher_seed: int,
        lower_seed: int,
        higher_seed_ratings: TeamRatings,
        lower_seed_ratings: TeamRatings,
        slot_id: int,
        year: int,
        game_round: int,
        higher_seed_team_id: int,
        lower_seed_team_id: int,
        higher_seed_name: str = "",
        lower_seed_name: str = "",
        games_played_higher: int = 0,
        games_played_lower: int = 0,
        market_spread: Optional[float] = None,
        market_total: Optional[float] = None,
    ) -> TournamentPrediction:
        """
        Generate tournament prediction with all adjustments.

        Higher seed is treated as "home" team for base model purposes.
        """
        # 1. Get base prediction (treating higher seed as "home")
        base_pred = self.base_predictor.predict_game(
            home_ratings=higher_seed_ratings,
            away_ratings=lower_seed_ratings,
            game_id=slot_id,
            is_neutral=True,  # Tournament is always neutral site
            home_rest_days=2,  # Default
            away_rest_days=2,
        )

        # 2. Apply neutral site compression to spread
        compressed_spread = base_pred.spread * NEUTRAL_SITE_SPREAD_COMPRESSION

        # 3. Apply neutral site total adjustment
        adjusted_total = base_pred.total + NEUTRAL_SITE_TOTAL_ADJUSTMENT

        # 4. Get round-specific variance multiplier
        round_var_mult = ROUND_VARIANCE_MULTIPLIERS.get(game_round, 1.0)
        adjusted_spread_std = base_pred.spread_std * round_var_mult
        adjusted_total_std = base_pred.total_std * round_var_mult

        # 5. Apply fatigue adjustments
        fatigue_higher = TOURNAMENT_FATIGUE_PENALTY.get(games_played_higher, 0.0)
        fatigue_lower = TOURNAMENT_FATIGUE_PENALTY.get(games_played_lower, 0.0)
        fatigue_adjustment = fatigue_higher - fatigue_lower

        # 6. Calculate seed-based prior (DISABLED: use model-only win probability)
        seed_prior_spread = 0.0

        # 7. Combine all adjustments
        tournament_adjustment = (
            (compressed_spread - base_pred.spread)  # Neutral site compression
            + fatigue_adjustment  # Fatigue effects
        )

        final_spread = base_pred.spread + tournament_adjustment
        final_total = adjusted_total

        # 8. Adjust scores proportionally
        spread_delta = final_spread - base_pred.spread
        higher_score = base_pred.home_score + spread_delta / 2
        lower_score = base_pred.away_score - spread_delta / 2

        # 9. Calculate win probabilities using seed-aware model
        higher_win_prob = self._calculate_win_probability(
            final_spread, adjusted_spread_std, higher_seed, lower_seed
        )
        upset_prob = 1.0 - higher_win_prob

        # 10. Calculate confidence intervals
        z50 = stats.norm.ppf(0.75)
        z80 = stats.norm.ppf(0.90)
        z95 = stats.norm.ppf(0.975)

        spread_ci_50 = (
            final_spread - z50 * adjusted_spread_std,
            final_spread + z50 * adjusted_spread_std,
        )
        spread_ci_80 = (
            final_spread - z80 * adjusted_spread_std,
            final_spread + z80 * adjusted_spread_std,
        )
        spread_ci_95 = (
            final_spread - z95 * adjusted_spread_std,
            final_spread + z95 * adjusted_spread_std,
        )

        total_ci_50 = (
            final_total - z50 * adjusted_total_std,
            final_total + z50 * adjusted_total_std,
        )
        total_ci_80 = (
            final_total - z80 * adjusted_total_std,
            final_total + z80 * adjusted_total_std,
        )
        total_ci_95 = (
            final_total - z95 * adjusted_total_std,
            final_total + z95 * adjusted_total_std,
        )
        higher_score_p10, higher_score_p90, lower_score_p10, lower_score_p90 = (
            self._calculate_score_percentiles(
                higher_score=higher_score,
                lower_score=lower_score,
                spread_std=adjusted_spread_std,
                total_std=adjusted_total_std,
            )
        )

        # 11. Calculate edge vs market
        edge_spread = None
        edge_total = None
        if market_spread is not None:
            edge_spread = final_spread - market_spread
        if market_total is not None:
            edge_total = final_total - market_total

        # 12. Determine recommendation
        recommended_side = None
        recommended_units = None
        confidence_rating = None

        if edge_spread is not None and abs(edge_spread) >= self.min_edge_threshold:
            recommended_side = "higher_seed" if edge_spread > 0 else "lower_seed"
            recommended_units = min(abs(edge_spread) / 3.0, 3.0)
            confidence_rating = (
                "high"
                if abs(edge_spread) >= 5.0
                else "medium"
                if abs(edge_spread) >= 3.0
                else "low"
            )
        elif edge_total is not None and abs(edge_total) >= 4.0:
            recommended_side = "over" if edge_total > 0 else "under"
            recommended_units = min(abs(edge_total) / 4.0, 3.0)
            confidence_rating = (
                "high" if abs(edge_total) >= 6.0 else "medium" if abs(edge_total) >= 4.0 else "low"
            )

        return TournamentPrediction(
            slot_id=slot_id,
            year=year,
            game_round=game_round,
            higher_seed=higher_seed,
            lower_seed=lower_seed,
            higher_seed_team_id=higher_seed_team_id,
            lower_seed_team_id=lower_seed_team_id,
            higher_seed_name=higher_seed_name,
            lower_seed_name=lower_seed_name,
            proj_higher_score=round(higher_score, 1),
            proj_lower_score=round(lower_score, 1),
            proj_spread=round(final_spread, 1),
            proj_total=round(final_total, 1),
            proj_possessions=round(base_pred.possessions, 1),
            higher_seed_win_prob=round(higher_win_prob, 3),
            upset_prob=round(upset_prob, 3),
            cover_prob=round(base_pred.home_cover_prob, 3) if base_pred.home_cover_prob else None,
            spread_std=round(adjusted_spread_std, 2),
            total_std=round(adjusted_total_std, 2),
            spread_ci_50=(round(spread_ci_50[0], 1), round(spread_ci_50[1], 1)),
            spread_ci_80=(round(spread_ci_80[0], 1), round(spread_ci_80[1], 1)),
            spread_ci_95=(round(spread_ci_95[0], 1), round(spread_ci_95[1], 1)),
            total_ci_50=(round(total_ci_50[0], 1), round(total_ci_50[1], 1)),
            total_ci_80=(round(total_ci_80[0], 1), round(total_ci_80[1], 1)),
            total_ci_95=(round(total_ci_95[0], 1), round(total_ci_95[1], 1)),
            higher_score_p10=round(higher_score_p10, 1),
            higher_score_p90=round(higher_score_p90, 1),
            lower_score_p10=round(lower_score_p10, 1),
            lower_score_p90=round(lower_score_p90, 1),
            tournament_adjustment=round(tournament_adjustment, 2),
            seed_momentum_factor=round(seed_prior_spread, 2),
            fatigue_factor=round(fatigue_adjustment, 2),
            round_variance_multiplier=round_var_mult,
            market_spread=market_spread,
            market_total=market_total,
            edge_vs_market_spread=round(edge_spread, 1) if edge_spread is not None else None,
            edge_vs_market_total=round(edge_total, 1) if edge_total is not None else None,
            recommended_side=recommended_side,
            recommended_units=recommended_units,
            confidence_rating=confidence_rating,
            model_version=MODEL_VERSION,
            prediction_time=datetime.now(timezone.utc),
        )

    def _calculate_seed_prior(self, higher_seed: int, lower_seed: int) -> float:
        """
        Calculate spread prior based on seed matchup history.

        Returns an adjustment to the spread (positive = higher seed favored more).
        """
        # Historical seed matchup win rates
        matchup_key = (higher_seed, lower_seed)
        historical_win_rate = SEED_WIN_RATES.get(matchup_key, 0.5)

        # Convert win rate to spread equivalent
        # Using inverse normal CDF: if win_prob = norm.cdf(spread / std), then spread = std * norm.ppf(win_prob)
        std_approx = 10.0  # Approximate game std
        implied_spread = std_approx * stats.norm.ppf(historical_win_rate)

        return implied_spread

    def _calculate_score_percentiles(
        self,
        higher_score: float,
        lower_score: float,
        spread_std: float,
        total_std: float,
    ) -> tuple[float, float, float, float]:
        """Approximate 10th and 90th percentile team scores from spread/total uncertainty."""
        z90 = stats.norm.ppf(0.90)
        score_std = float(np.sqrt(spread_std**2 + total_std**2) / 2.0)

        higher_p10 = max(30.0, higher_score - z90 * score_std)
        higher_p90 = min(125.0, higher_score + z90 * score_std)
        lower_p10 = max(30.0, lower_score - z90 * score_std)
        lower_p90 = min(125.0, lower_score + z90 * score_std)
        return higher_p10, higher_p90, lower_p10, lower_p90

    def _calculate_win_probability(
        self,
        spread: float,
        spread_std: float,
        higher_seed: int,
        lower_seed: int,
    ) -> float:
        """
        Calculate win probability from model spread and uncertainty.

        Pure model-based: P(win) = CDF(spread / spread_std)
        No seed prior blending - upsets emerge from the model distribution.
        """
        model_prob = stats.norm.cdf(spread / spread_std)
        return np.clip(model_prob, 0.01, 0.99)


def save_tournament_prediction(
    prediction: TournamentPrediction,
    conn=None,
) -> None:
    """Save tournament prediction to database."""
    if conn is None:
        from packages.common.database import get_connection

        with get_connection() as conn:
            _save_prediction(conn, prediction)
    else:
        _save_prediction(conn, prediction)


def _save_prediction(conn, prediction: TournamentPrediction) -> None:
    """Internal save function."""
    import uuid

    pred_id = str(uuid.uuid4())

    conn.execute(
        """
        INSERT OR REPLACE INTO tournament_predictions (
            prediction_id, slot_id, year, round, prediction_timestamp, model_version,
            higher_seed, lower_seed, higher_seed_team_id, lower_seed_team_id,
            proj_higher_score, proj_lower_score, proj_spread, proj_total, proj_possessions,
            higher_seed_win_prob, upset_prob, cover_prob,
            spread_std, total_std,
            spread_ci_50_lower, spread_ci_50_upper,
            spread_ci_80_lower, spread_ci_80_upper,
            spread_ci_95_lower, spread_ci_95_upper,
            total_ci_50_lower, total_ci_50_upper,
            total_ci_80_lower, total_ci_80_upper,
            total_ci_95_lower, total_ci_95_upper,
            market_spread, market_total, edge_vs_market_spread, edge_vs_market_total,
            tournament_adjustment, seed_momentum_factor, fatigue_factor,
            recommended_side, recommended_units, confidence_rating
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pred_id,
            prediction.slot_id,
            prediction.year,
            prediction.game_round,
            prediction.prediction_time.isoformat(),
            prediction.model_version,
            prediction.higher_seed,
            prediction.lower_seed,
            prediction.higher_seed_team_id,
            prediction.lower_seed_team_id,
            prediction.proj_higher_score,
            prediction.proj_lower_score,
            prediction.proj_spread,
            prediction.proj_total,
            prediction.proj_possessions,
            prediction.higher_seed_win_prob,
            prediction.upset_prob,
            prediction.cover_prob,
            prediction.spread_std,
            prediction.total_std,
            prediction.spread_ci_50[0],
            prediction.spread_ci_50[1],
            prediction.spread_ci_80[0],
            prediction.spread_ci_80[1],
            prediction.spread_ci_95[0],
            prediction.spread_ci_95[1],
            prediction.total_ci_50[0],
            prediction.total_ci_50[1],
            prediction.total_ci_80[0],
            prediction.total_ci_80[1],
            prediction.total_ci_95[0],
            prediction.total_ci_95[1],
            prediction.market_spread,
            prediction.market_total,
            prediction.edge_vs_market_spread,
            prediction.edge_vs_market_total,
            prediction.tournament_adjustment,
            prediction.seed_momentum_factor,
            prediction.fatigue_factor,
            prediction.recommended_side,
            prediction.recommended_units,
            prediction.confidence_rating,
        ),
    )
