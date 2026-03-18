"""Tests for EnhancedPredictor market anchoring."""

from datetime import date

from packages.features.kenpom_ratings import TeamRatings
from packages.models.enhanced_predictor import EnhancedPredictor


def _make_ratings(team_id: int) -> TeamRatings:
    return TeamRatings(
        team_id=team_id,
        adj_off=100.0,
        adj_def=100.0,
        adj_tempo=68.0,
        adj_em=0.0,
        adj_efg=0.50,
        adj_tov=0.18,
        adj_orb=0.30,
        adj_ftr=0.30,
        adj_efg_def=0.50,
        adj_tov_def=0.18,
        adj_drb=0.70,
        adj_ftr_def=0.30,
        games_played=10,
        sos_off=0.0,
        sos_def=0.0,
        as_of_date=date(2025, 1, 1),
        home_off_delta=0.0,
        home_def_delta=0.0,
        away_off_delta=0.0,
        away_def_delta=0.0,
        home_games_played=5,
        away_games_played=5,
        off_std=0.0,
        def_std=0.0,
        tempo_std=0.0,
    )


def test_market_anchoring_hits_market_lines() -> None:
    home = _make_ratings(9001)
    away = _make_ratings(9002)

    predictor = EnhancedPredictor(
        spread_bias=0.0,
        spread_scale=1.0,
        total_bias=0.0,
        total_scale=1.0,
        market_anchor_weight_spread=1.0,
        market_anchor_weight_total=1.0,
    )

    pred = predictor.predict_game(
        home_ratings=home,
        away_ratings=away,
        game_id=1,
        is_neutral=True,
        market_spread=7.0,
        market_total=140.0,
    )

    assert pred.spread == 7.0
    assert pred.total == 140.0
