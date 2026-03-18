"""Tests for tournament score band calculations."""

from packages.models.tournament_predictor import TournamentPredictor


def test_score_percentiles_stay_in_reasonable_range() -> None:
    predictor = TournamentPredictor()

    higher_p10, higher_p90, lower_p10, lower_p90 = predictor._calculate_score_percentiles(
        higher_score=82.0,
        lower_score=61.0,
        spread_std=13.0,
        total_std=18.0,
    )

    assert 30.0 <= higher_p10 < 82.0 < higher_p90 <= 125.0
    assert 30.0 <= lower_p10 < 61.0 < lower_p90 <= 125.0
