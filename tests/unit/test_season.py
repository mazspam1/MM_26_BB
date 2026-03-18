"""Tests for season helpers."""

from datetime import date

from packages.common.season import infer_season_year, season_start_date


def test_infer_season_year_november_maps_to_next_year() -> None:
    assert infer_season_year(date(2024, 11, 5)) == 2025


def test_infer_season_year_march_maps_to_same_year() -> None:
    assert infer_season_year(date(2025, 3, 10)) == 2025


def test_season_start_date_january() -> None:
    assert season_start_date(date(2025, 1, 15)) == date(2024, 7, 1)


def test_season_start_date_august() -> None:
    assert season_start_date(date(2025, 8, 1)) == date(2025, 7, 1)
