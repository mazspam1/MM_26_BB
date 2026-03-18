"""Tests for data-quality guardrails."""

from datetime import datetime

from packages.common.schemas import PredictionRow
from packages.eval.guardrails import apply_min_games_guardrail


def _sample_prediction_row(recommended_side: str | None = "home_spread") -> PredictionRow:
    return PredictionRow(
        game_id=1,
        prediction_timestamp=datetime.utcnow(),
        model_version="v1.0.0-test",
        proj_home_score=75.0,
        proj_away_score=70.0,
        proj_spread=5.0,
        proj_total=145.0,
        proj_possessions=70.0,
        home_win_prob=0.6,
        spread_ci_50_lower=2.0,
        spread_ci_50_upper=8.0,
        spread_ci_80_lower=0.0,
        spread_ci_80_upper=10.0,
        spread_ci_95_lower=-2.0,
        spread_ci_95_upper=12.0,
        total_ci_50_lower=140.0,
        total_ci_50_upper=150.0,
        total_ci_80_lower=135.0,
        total_ci_80_upper=155.0,
        total_ci_95_lower=130.0,
        total_ci_95_upper=160.0,
        market_spread=4.5,
        edge_vs_market_spread=0.5,
        market_total=145.0,
        edge_vs_market_total=0.0,
        recommended_side=recommended_side,
        recommended_units=1.0 if recommended_side else None,
        confidence_rating="medium" if recommended_side else None,
    )


def test_guardrail_applies_with_low_samples() -> None:
    pred = _sample_prediction_row()
    guarded = apply_min_games_guardrail(pred, home_games_played=4, away_games_played=12, min_games_played=8)
    assert guarded.recommended_side == "no_bet"
    assert guarded.recommended_units == 0.0
    assert guarded.confidence_rating == "low"


def test_guardrail_skips_when_no_recommendation() -> None:
    pred = _sample_prediction_row(recommended_side=None)
    guarded = apply_min_games_guardrail(pred, home_games_played=2, away_games_played=2, min_games_played=8)
    assert guarded.recommended_side is None
