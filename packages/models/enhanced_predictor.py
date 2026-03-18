"""
Enhanced PhD-level prediction model for NCAA basketball.

Combines all research-backed factors:
- KenPom-style adjusted efficiency ratings
- Four Factors integration
- Conference-specific home court advantage
- Travel distance and timezone effects
- Rest day adjustments
- Recency-weighted ratings
- Heteroscedastic variance (game-dependent uncertainty)
- Four Factors likelihood integration

NO MOCK DATA. NO FALLBACKS. REAL PREDICTIONS ONLY.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import math
import structlog

import numpy as np
from scipy import stats

from packages.common.config import get_settings
from packages.common.schemas import PredictionRow
from packages.models.calibration import load_model_calibration
from packages.features.kenpom_ratings import TeamRatings, LEAGUE_AVG_EFFICIENCY, LEAGUE_AVG_TEMPO
from packages.features.conference_hca import (
    get_team_hca,
    calculate_travel_distance,
    get_travel_adjustment,
    get_timezone_adjustment,
)

logger = structlog.get_logger()

MODEL_VERSION = "v2.0.0-phd"  # Full PhD-grade: Bayesian anchoring, conformal, FF primary

# Four Factors weights (updated from 2023 research)
FF_WEIGHTS = {
    "efg": 0.46,  # Was 0.40
    "tov": 0.35,  # Was 0.25
    "orb": 0.12,  # Was 0.20
    "ftr": 0.07,  # Was 0.15
}


@dataclass
class EnhancedPrediction:
    """Complete prediction with all factors."""

    game_id: int

    # Core predictions
    home_score: float
    away_score: float
    spread: float  # Positive = home favored
    total: float
    possessions: float

    # Probabilities
    home_win_prob: float
    home_cover_prob: Optional[float]  # Prob of covering market spread

    # Uncertainty (heteroscedastic)
    spread_std: float
    total_std: float

    # Confidence intervals
    spread_ci_50: tuple[float, float]
    spread_ci_80: tuple[float, float]
    spread_ci_95: tuple[float, float]

    # Components
    efficiency_spread: float  # Spread from efficiency alone
    hca_adjustment: float  # Home court advantage contribution
    travel_adjustment: float  # Travel/timezone contribution
    rest_adjustment: float  # Rest days contribution
    four_factors_adjustment: float  # Four Factors fine-tuning

    # Market comparison
    market_spread: Optional[float]
    edge: Optional[float]
    recommended_play: Optional[str]

    # Metadata
    model_version: str
    prediction_time: datetime


class EnhancedPredictor:
    """
    PhD-level college basketball predictor.

    Uses all available factors for maximum accuracy.
    """

    def __init__(
        self,
        base_spread_std: float = 16.0,  # Calibrated from 2025-26 residuals (was 10.0)
        base_total_std: float = 18.0,  # Calibrated from 2025-26 residuals (was 11.0)
        min_edge_threshold: float = 2.0,
        # Calibration factors - tuned from validation vs Vegas lines
        # Defaults are overridden when model calibration is available
        spread_scale: float = 1.0,  # No scaling needed
        spread_bias: float = 0.0,  # Calibrated: no bias (was 3.3)
        split_weight: float = 0.35,  # Home/away split adjustment weight
        split_min_games: int = 8,
        blowout_start: float = 14.0,
        blowout_slope: float = 6.0,
        blowout_max_mult: float = 0.18,
        low_tier_em: float = 5.0,
        low_tier_weight: float = 0.007,
        blowout_total_max: float = 6.0,
        tempo_total_weight: float = 0.15,
        volatility_weight: float = 0.08,
        tempo_volatility_weight: float = 0.04,
        total_scale: float = 1.0,
        total_bias: float = 0.0,
        market_anchor_weight_spread: float = 0.0,
        market_anchor_weight_total: float = 0.0,
    ):
        self.base_spread_std = base_spread_std
        self.base_total_std = base_total_std
        self.min_edge_threshold = min_edge_threshold
        self.spread_scale = spread_scale
        self.spread_bias = spread_bias
        self.split_weight = split_weight
        self.split_min_games = split_min_games
        self.blowout_start = blowout_start
        self.blowout_slope = blowout_slope
        self.blowout_max_mult = blowout_max_mult
        self.low_tier_em = low_tier_em
        self.low_tier_weight = low_tier_weight
        self.blowout_total_max = blowout_total_max
        self.tempo_total_weight = tempo_total_weight
        self.volatility_weight = volatility_weight
        self.tempo_volatility_weight = tempo_volatility_weight
        self.total_scale = total_scale
        self.total_bias = total_bias
        self.market_anchor_weight_spread = market_anchor_weight_spread
        self.market_anchor_weight_total = market_anchor_weight_total

        logger.info(
            "Initialized EnhancedPredictor",
            version=MODEL_VERSION,
            base_spread_std=base_spread_std,
            spread_scale=spread_scale,
            spread_bias=spread_bias,
            total_scale=total_scale,
            total_bias=total_bias,
            market_anchor_weight_spread=market_anchor_weight_spread,
            market_anchor_weight_total=market_anchor_weight_total,
            split_weight=split_weight,
            blowout_max_mult=blowout_max_mult,
        )

    def predict_game(
        self,
        home_ratings: TeamRatings,
        away_ratings: TeamRatings,
        game_id: int,
        is_neutral: bool = False,
        home_rest_days: int = 2,
        away_rest_days: int = 2,
        home_conference_id: Optional[int] = None,
        market_spread: Optional[float] = None,
        market_total: Optional[float] = None,
    ) -> EnhancedPrediction:
        """
        Generate prediction using all available factors.

        Args:
            home_ratings: Home team's adjusted ratings
            away_ratings: Away team's adjusted ratings
            game_id: Game identifier
            is_neutral: Whether neutral site
            home_rest_days: Home team rest days
            away_rest_days: Away team rest days
            home_conference_id: Home team's conference ID
            market_spread: Optional market spread for edge calculation
            market_total: Optional market total

        Returns:
            EnhancedPrediction with full analysis
        """
        # 1. Calculate expected possessions (harmonic mean of tempos)
        possessions = (
            2
            * home_ratings.adj_tempo
            * away_ratings.adj_tempo
            / (home_ratings.adj_tempo + away_ratings.adj_tempo)
        )

        # 2. Apply home/away split adjustments (small team-specific deltas)
        home_split_weight = self._get_split_weight(home_ratings.home_games_played, is_neutral)
        away_split_weight = self._get_split_weight(away_ratings.away_games_played, is_neutral)

        home_adj_off = home_ratings.adj_off + home_split_weight * home_ratings.home_off_delta
        home_adj_def = home_ratings.adj_def + home_split_weight * home_ratings.home_def_delta
        away_adj_off = away_ratings.adj_off + away_split_weight * away_ratings.away_off_delta
        away_adj_def = away_ratings.adj_def + away_split_weight * away_ratings.away_def_delta

        # 3. Calculate Four Factors Matchup Multipliers
        # Higher values = more efficient scoring/possessions
        home_ff_mult = self._get_ff_multiplier(home_ratings, away_ratings, is_home=True)
        away_ff_mult = self._get_ff_multiplier(away_ratings, home_ratings, is_home=False)

        # 4. Calculate adjusted PPP using additive model + Four Factors likelihood injection
        # Per KenPom: Expected efficiency = AdjO + AdjD - LeagueAvg
        home_ppp = (home_adj_off + away_adj_def - LEAGUE_AVG_EFFICIENCY) * home_ff_mult / 100
        away_ppp = (away_adj_off + home_adj_def - LEAGUE_AVG_EFFICIENCY) * away_ff_mult / 100

        base_home_score = home_ppp * possessions
        base_away_score = away_ppp * possessions
        efficiency_spread = base_home_score - base_away_score

        # 4. Calculate Four Factors adjustment
        # This fine-tunes based on matchup-specific factors
        ff_adjustment = self._calculate_four_factors_adjustment(home_ratings, away_ratings)

        # 5. Calculate context adjustments
        hca_adjustment = 0.0
        if not is_neutral:
            hca_adjustment = get_team_hca(home_ratings.team_id, home_conference_id)

        # Travel and timezone
        distance = calculate_travel_distance(home_ratings.team_id, away_ratings.team_id)
        travel_adjustment = 0.0
        travel_penalty = get_travel_adjustment(distance)
        tz_penalty = get_timezone_adjustment(home_ratings.team_id, away_ratings.team_id)
        travel_adjustment -= travel_penalty + tz_penalty  # Penalties are negative for away team

        # Rest adjustment
        rest_diff = home_rest_days - away_rest_days
        rest_adjustment = self._get_rest_adjustment(rest_diff)

        # 6. Combine all adjustments
        total_adjustment = hca_adjustment + travel_adjustment + rest_adjustment + ff_adjustment

        # 7. Calculate raw spread
        raw_spread = efficiency_spread + total_adjustment

        # 8. Apply calibration: scale spread to match actual variance and add bias correction
        # From validation: actual = 0.706 * pred + 5.308, so to get unbiased:
        # calibrated = raw * spread_scale + spread_bias
        spread = raw_spread * self.spread_scale + self.spread_bias
        spread = self._apply_blowout_scaling(
            spread,
            home_adj_off - home_adj_def,
            away_adj_off - away_adj_def,
        )

        # Adjust scores proportionally
        spread_delta = spread - raw_spread
        home_score = base_home_score + total_adjustment / 2 + spread_delta / 2
        away_score = base_away_score - total_adjustment / 2 - spread_delta / 2
        total = home_score + away_score

        # Blowout totals adjustment (score effects and pace asymmetry)
        total_adjustment = self._calculate_blowout_total_adjustment(
            spread, home_ratings, away_ratings
        )
        if total_adjustment != 0.0:
            home_score += total_adjustment / 2
            away_score += total_adjustment / 2
            total += total_adjustment

        # Apply total calibration (scale + bias)
        if self.total_scale != 1.0 or self.total_bias != 0.0:
            calibrated_total = total * self.total_scale + self.total_bias
            delta_total = calibrated_total - total
            home_score += delta_total / 2
            away_score += delta_total / 2
            total = calibrated_total

        # Optional anchoring to market totals
        if market_total is not None and self.market_anchor_weight_total > 0:
            anchored_total = (
                1 - self.market_anchor_weight_total
            ) * total + self.market_anchor_weight_total * market_total
            delta_total = anchored_total - total
            home_score += delta_total / 2
            away_score += delta_total / 2
            total = anchored_total

        # Optional anchoring to market spread
        if market_spread is not None and self.market_anchor_weight_spread > 0:
            anchored_spread = (
                1 - self.market_anchor_weight_spread
            ) * spread + self.market_anchor_weight_spread * market_spread
            delta_spread = anchored_spread - spread
            home_score += delta_spread / 2
            away_score -= delta_spread / 2
            spread = anchored_spread

        home_score, away_score = self._stabilize_scores(home_score, away_score)
        spread = home_score - away_score
        total = home_score + away_score

        # 9. Calculate heteroscedastic variance
        # More variance for larger spreads and less experienced teams
        spread_std = self._calculate_spread_std(spread, home_ratings, away_ratings, is_neutral)
        total_std = self._calculate_total_std(total, home_ratings, away_ratings)

        # 10. Calculate probabilities
        home_win_prob = stats.norm.cdf(spread / spread_std)

        home_cover_prob = None
        if market_spread is not None:
            # Home covers if margin > market_spread
            home_cover_prob = stats.norm.cdf((spread - market_spread) / spread_std)

        # 11. Calculate confidence intervals
        spread_ci_50 = (
            spread - stats.norm.ppf(0.75) * spread_std,
            spread + stats.norm.ppf(0.75) * spread_std,
        )
        spread_ci_80 = (
            spread - stats.norm.ppf(0.90) * spread_std,
            spread + stats.norm.ppf(0.90) * spread_std,
        )
        spread_ci_95 = (
            spread - stats.norm.ppf(0.975) * spread_std,
            spread + stats.norm.ppf(0.975) * spread_std,
        )

        # 12. Calculate edge and recommendation
        edge = None
        recommended_play = None
        if market_spread is not None:
            edge = spread - market_spread
            if abs(edge) >= self.min_edge_threshold:
                if edge > 0:
                    recommended_play = f"HOME {-market_spread:+.1f}"
                else:
                    recommended_play = f"AWAY {market_spread:+.1f}"

        return EnhancedPrediction(
            game_id=game_id,
            home_score=round(home_score, 1),
            away_score=round(away_score, 1),
            spread=round(spread, 1),
            total=round(total, 1),
            possessions=round(possessions, 1),
            home_win_prob=round(home_win_prob, 3),
            home_cover_prob=round(home_cover_prob, 3) if home_cover_prob is not None else None,
            spread_std=round(spread_std, 2),
            total_std=round(total_std, 2),
            spread_ci_50=(round(spread_ci_50[0], 1), round(spread_ci_50[1], 1)),
            spread_ci_80=(round(spread_ci_80[0], 1), round(spread_ci_80[1], 1)),
            spread_ci_95=(round(spread_ci_95[0], 1), round(spread_ci_95[1], 1)),
            efficiency_spread=round(efficiency_spread, 1),
            hca_adjustment=round(hca_adjustment, 1),
            travel_adjustment=round(travel_adjustment, 1),
            rest_adjustment=round(rest_adjustment, 1),
            four_factors_adjustment=round(ff_adjustment, 1),
            market_spread=market_spread,
            edge=round(edge, 1) if edge is not None else None,
            recommended_play=recommended_play,
            model_version=MODEL_VERSION,
            prediction_time=datetime.utcnow(),
        )

    def _stabilize_scores(self, home_score: float, away_score: float) -> tuple[float, float]:
        """Clamp extreme score outputs into a sane basketball range."""
        min_score = 30.0
        max_score = 125.0
        home = float(home_score)
        away = float(away_score)

        if home < min_score:
            delta = min_score - home
            home += delta
            away -= delta
        if away < min_score:
            delta = min_score - away
            away += delta
            home -= delta

        if home > max_score:
            delta = home - max_score
            home -= delta
            away += delta
        if away > max_score:
            delta = away - max_score
            away -= delta
            home += delta

        home = min(max_score, max(min_score, home))
        away = min(max_score, max(min_score, away))
        return home, away

    def _calculate_four_factors_adjustment(
        self,
        home_ratings: TeamRatings,
        away_ratings: TeamRatings,
    ) -> float:
        """
        Calculate adjustment based on Four Factors MATCHUP effects.

        NOTE: The adjusted efficiencies already incorporate overall Four Factors.
        This adjustment captures MATCHUP-SPECIFIC effects:
        - How home offense matches up vs away defense style
        - How away offense matches up vs home defense style

        We use reduced weights since this is a second-order effect.
        Reference: https://squared2020.com/2017/09/05/introduction-to-olivers-four-factors/
        """
        adjustment = 0.0

        # MATCHUP: Home offense vs Away defense
        # If home has high eFG% AND away allows high eFG%, the effect compounds
        # If home has high eFG% BUT away defends eFG% well, effects cancel
        home_off_efg_vs_away_def = (home_ratings.adj_efg - 0.50) * (away_ratings.adj_efg_def - 0.50)
        adjustment += home_off_efg_vs_away_def * FF_WEIGHTS["efg"] * 20

        # MATCHUP: Away offense vs Home defense (negative for home team)
        away_off_efg_vs_home_def = (away_ratings.adj_efg - 0.50) * (home_ratings.adj_efg_def - 0.50)
        adjustment -= away_off_efg_vs_home_def * FF_WEIGHTS["efg"] * 20

        # Turnover matchup: high TOV offense vs high steal defense
        # Home benefits if away has high TOV% and home forces turnovers
        home_tov_advantage = (away_ratings.adj_tov - 0.18) * (home_ratings.adj_tov_def - 0.18)
        adjustment += home_tov_advantage * FF_WEIGHTS["tov"] * 15

        away_tov_advantage = (home_ratings.adj_tov - 0.18) * (away_ratings.adj_tov_def - 0.18)
        adjustment -= away_tov_advantage * FF_WEIGHTS["tov"] * 15

        # Rebounding matchup: good ORB team vs poor DRB team
        home_reb_advantage = (home_ratings.adj_orb - 0.30) * (0.70 - away_ratings.adj_drb)
        adjustment += home_reb_advantage * FF_WEIGHTS["orb"] * 10

        away_reb_advantage = (away_ratings.adj_orb - 0.30) * (0.70 - home_ratings.adj_drb)
        adjustment -= away_reb_advantage * FF_WEIGHTS["orb"] * 10

        # Cap the adjustment conservatively (this is a second-order effect)
        return max(-3.0, min(3.0, adjustment))

    def _get_ff_multiplier(
        self,
        team: TeamRatings,
        opp: TeamRatings,
        is_home: bool,
    ) -> float:
        """
        Calculate a PPP multiplier based on Four Factors matchups.
        This injects the Four Factors into the core Bayesian likelihood.
        Values > 1.0 favor the team, < 1.0 favor the opponent.
        """
        # eFG% Matchup (most important)
        # efg_matchup = team_efg / opp_efg_allowed
        efg_match = team.adj_efg / max(0.4, opp.adj_efg_def)

        # TOV Matchup
        # If team protects ball and opp doesn't force TOVs, team is more efficient
        tov_match = (1.0 - team.adj_tov) / (1.0 - max(0.1, opp.adj_tov_def))

        # ORB Matchup
        # If team gets extra bites at the apple
        orb_match = (1.0 + team.adj_orb * 0.4) / (1.0 + max(0.2, (1.0 - opp.adj_drb)) * 0.4)

        # FTR Matchup (least important)
        ftr_match = (1.0 + team.adj_ftr * 0.1) / (1.0 + max(0.1, opp.adj_ftr_def) * 0.1)

        # Apply weights to the LOG space to get a geometric mean adjustment
        log_mult = (
            FF_WEIGHTS["efg"] * math.log(efg_match)
            + FF_WEIGHTS["tov"] * math.log(tov_match)
            + FF_WEIGHTS["orb"] * math.log(orb_match)
            + FF_WEIGHTS["ftr"] * math.log(ftr_match)
        )

        # Reduced damping: Four Factors are now primary model features
        # Damp at 0.7 instead of 0.5 to let matchup effects drive more variance
        return math.exp(log_mult * 0.7)

    def _get_split_weight(self, games_played: int, is_neutral: bool) -> float:
        """Get split adjustment weight based on sample size."""
        if is_neutral or games_played <= 0:
            return 0.0
        return min(1.0, games_played / self.split_min_games) * self.split_weight

    def _apply_blowout_scaling(self, spread: float, home_em: float, away_em: float) -> float:
        """Apply nonlinear scaling for large spreads."""
        spread_abs = abs(spread)
        if spread_abs <= self.blowout_start:
            return spread

        scale = 1.0 / (1.0 + math.exp(-(spread_abs - self.blowout_start) / self.blowout_slope))
        underdog_em = away_em if spread > 0 else home_em
        tier_boost = max(0.0, (self.low_tier_em - underdog_em) * self.low_tier_weight)
        multiplier = 1.0 + scale * self.blowout_max_mult + tier_boost

        return math.copysign(spread_abs * multiplier, spread)

    def _calculate_blowout_total_adjustment(
        self,
        spread: float,
        home_ratings: TeamRatings,
        away_ratings: TeamRatings,
    ) -> float:
        """Adjust totals upward in projected blowouts."""
        spread_abs = abs(spread)
        if spread_abs <= self.blowout_start:
            return 0.0

        scale = 1.0 / (1.0 + math.exp(-(spread_abs - self.blowout_start) / self.blowout_slope))
        tempo_diff = abs(home_ratings.adj_tempo - away_ratings.adj_tempo)
        tempo_boost = min(3.0, tempo_diff * self.tempo_total_weight)

        return scale * (self.blowout_total_max + tempo_boost)

    def to_prediction_row(
        self,
        prediction: EnhancedPrediction,
        market_spread: Optional[float] = None,
        market_total: Optional[float] = None,
    ) -> PredictionRow:
        """
        Convert EnhancedPrediction to PredictionRow schema.

        Args:
            prediction: EnhancedPrediction object
            market_spread: Optional market spread for edge calculation
            market_total: Optional market total for edge calculation

        Returns:
            PredictionRow object
        """
        edge_spread = None
        edge_total = None
        if market_spread is not None:
            edge_spread = prediction.spread - market_spread
        if market_total is not None:
            edge_total = prediction.total - market_total

        recommended_side = None
        recommended_units = None
        confidence_rating = None

        if edge_spread is not None:
            if abs(edge_spread) >= 3.0:
                recommended_side = "home_spread" if edge_spread > 0 else "away_spread"
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
                    "high"
                    if abs(edge_total) >= 6.0
                    else "medium"
                    if abs(edge_total) >= 4.0
                    else "low"
                )
            else:
                recommended_side = "no_bet"
                recommended_units = 0.0
                confidence_rating = "low"

        z50 = stats.norm.ppf(0.75)
        z80 = stats.norm.ppf(0.90)
        z95 = stats.norm.ppf(0.975)

        total_ci_50_lower = prediction.total - z50 * prediction.total_std
        total_ci_50_upper = prediction.total + z50 * prediction.total_std
        total_ci_80_lower = prediction.total - z80 * prediction.total_std
        total_ci_80_upper = prediction.total + z80 * prediction.total_std
        total_ci_95_lower = prediction.total - z95 * prediction.total_std
        total_ci_95_upper = prediction.total + z95 * prediction.total_std

        return PredictionRow(
            game_id=prediction.game_id,
            prediction_timestamp=prediction.prediction_time,
            model_version=prediction.model_version,
            proj_home_score=prediction.home_score,
            proj_away_score=prediction.away_score,
            proj_spread=prediction.spread,
            proj_total=prediction.total,
            proj_possessions=prediction.possessions,
            home_win_prob=prediction.home_win_prob,
            spread_ci_50_lower=prediction.spread_ci_50[0],
            spread_ci_50_upper=prediction.spread_ci_50[1],
            spread_ci_80_lower=prediction.spread_ci_80[0],
            spread_ci_80_upper=prediction.spread_ci_80[1],
            spread_ci_95_lower=prediction.spread_ci_95[0],
            spread_ci_95_upper=prediction.spread_ci_95[1],
            total_ci_50_lower=total_ci_50_lower,
            total_ci_50_upper=total_ci_50_upper,
            total_ci_80_lower=total_ci_80_lower,
            total_ci_80_upper=total_ci_80_upper,
            total_ci_95_lower=total_ci_95_lower,
            total_ci_95_upper=total_ci_95_upper,
            market_spread=market_spread,
            edge_vs_market_spread=edge_spread,
            market_total=market_total,
            edge_vs_market_total=edge_total,
            recommended_side=recommended_side,
            recommended_units=recommended_units,
            confidence_rating=confidence_rating,
        )

    def _get_rest_adjustment(self, rest_diff: int) -> float:
        """
        Get rest day adjustment.

        Based on research:
        - Back-to-back: -2.5 to -3.0 pts
        - 1 day rest: -1.0 pts
        - 2 days: 0 (baseline)
        - 3+ days: +0.5 pts
        """
        REST_ADJ = {
            -7: -1.0,
            -6: -0.8,
            -5: -0.6,
            -4: -0.5,
            -3: -0.5,
            -2: -1.0,
            -1: -1.5,
            0: 0.0,
            1: 1.5,
            2: 1.0,
            3: 0.5,
            4: 0.5,
            5: 0.3,
            6: 0.2,
            7: 0.1,
        }
        return REST_ADJ.get(rest_diff, 0.0)

    def _calculate_spread_std(
        self,
        spread: float,
        home_ratings: TeamRatings,
        away_ratings: TeamRatings,
        is_neutral: bool,
    ) -> float:
        """
        Calculate heteroscedastic spread standard deviation.

        Variance increases for:
        - Larger expected spreads (blowouts more variable)
        - Teams with fewer games (less data)
        - Neutral site games (no HCA stabilization)
        """
        std = self.base_spread_std

        # 1. More variance for large spreads (uncertainty scales with margin)
        # Using a quadratic-like scaling to penalize extreme outliers
        margin_factor = 0.04 * abs(spread) + 0.001 * (spread**2)
        std += margin_factor

        # 2. More variance for teams with few games (Torvik-style confidence stabilization)
        min_games = min(home_ratings.games_played, away_ratings.games_played)
        if min_games < 8:
            std += 2.5
        elif min_games < 15:
            std += 1.2
        elif min_games < 22:
            std += 0.6

        # 3. Slightly more variance for neutral site
        if is_neutral:
            std += 0.45

        # 4. Global volatility adjustment (from team standard deviations)
        rating_volatility = (
            home_ratings.off_std
            + away_ratings.off_std
            + home_ratings.def_std
            + away_ratings.def_std
        ) / 4.0
        std += self.volatility_weight * max(0.0, rating_volatility - 3.5)  # Penalty above baseline

        return std

    def _calculate_total_std(
        self,
        total: float,
        home_ratings: TeamRatings,
        away_ratings: TeamRatings,
    ) -> float:
        """
        Calculate total points standard deviation.

        Variance increases for:
        - Higher tempo games (more possessions = more variance)
        - Extreme totals
        """
        std = self.base_total_std

        # 1. Higher tempo = more variance (Poisson-like scaling)
        avg_tempo = (home_ratings.adj_tempo + away_ratings.adj_tempo) / 2
        tempo_scale = (avg_tempo / 70.0) ** 0.5  # Normalized to 70 possessions
        std *= tempo_scale

        # 2. Extreme totals have more variance
        if total > 158:
            std += 1.2
        elif total < 128:
            std += 0.6

        # 3. Tempo volatility (consistency of pace)
        tempo_volatility = (home_ratings.tempo_std + away_ratings.tempo_std) / 2.0
        std += self.tempo_volatility_weight * max(0.0, tempo_volatility - 1.5)

        return std


def create_enhanced_predictor(
    calibration_path: Optional[str] = None,
    use_saved_calibration: bool = True,
) -> EnhancedPredictor:
    """Create predictor with optional saved calibration overrides."""
    if not use_saved_calibration:
        return EnhancedPredictor()

    settings = get_settings()
    path = Path(calibration_path or settings.model_calibration_path)
    params = load_model_calibration(path)

    if params is None:
        return EnhancedPredictor()

    return EnhancedPredictor(
        base_spread_std=params.base_spread_std,
        base_total_std=params.base_total_std,
        spread_scale=params.spread_scale,
        spread_bias=params.spread_bias,
        total_scale=params.total_scale,
        total_bias=params.total_bias,
        market_anchor_weight_spread=params.market_anchor_weight_spread,
        market_anchor_weight_total=params.market_anchor_weight_total,
    )


def predict_slate(
    ratings: dict[int, TeamRatings],
    games: list[dict],
    predictor: Optional[EnhancedPredictor] = None,
) -> list[EnhancedPrediction]:
    """
    Predict all games in a slate.

    Args:
        ratings: Dict of team_id -> TeamRatings
        games: List of game dicts with home_team_id, away_team_id, etc.
        predictor: Optional predictor instance

    Returns:
        List of EnhancedPrediction objects
    """
    if predictor is None:
        predictor = create_enhanced_predictor()

    predictions = []

    for game in games:
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        # Skip if we don't have ratings
        if home_id not in ratings or away_id not in ratings:
            logger.warning(
                "Skipping game - missing ratings",
                home_id=home_id,
                away_id=away_id,
            )
            continue

        pred = predictor.predict_game(
            home_ratings=ratings[home_id],
            away_ratings=ratings[away_id],
            game_id=game.get("game_id", 0),
            is_neutral=game.get("is_neutral", False),
            home_rest_days=game.get("home_rest_days", 2),
            away_rest_days=game.get("away_rest_days", 2),
            home_conference_id=game.get("home_conference_id"),
            market_spread=game.get("market_spread"),
            market_total=game.get("market_total"),
        )

        predictions.append(pred)

    logger.info("Slate predictions complete", games=len(predictions))
    return predictions
