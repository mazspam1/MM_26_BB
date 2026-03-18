"""Tests for Division I rating-universe filtering."""

from datetime import date

import pandas as pd

from packages.features.kenpom_ratings import calculate_adjusted_ratings


def test_calculate_adjusted_ratings_filters_to_d1_universe() -> None:
    team_stats = pd.DataFrame(
        [
            {
                "game_date": date(2025, 11, 1),
                "team_id": 1,
                "opponent_id": 99,
                "off_rating": 112.0,
                "def_rating": 88.0,
                "possessions": 70.0,
                "off_efg": 0.52,
                "off_tov": 0.16,
                "off_orb": 0.31,
                "off_ftr": 0.29,
                "def_efg": 0.44,
                "def_tov": 0.18,
                "def_orb": 0.27,
                "def_ftr": 0.22,
                "is_home": True,
                "is_neutral": False,
            },
            {
                "game_date": date(2025, 11, 10),
                "team_id": 1,
                "opponent_id": 2,
                "off_rating": 101.0,
                "def_rating": 97.0,
                "possessions": 68.0,
                "off_efg": 0.50,
                "off_tov": 0.18,
                "off_orb": 0.28,
                "off_ftr": 0.27,
                "def_efg": 0.48,
                "def_tov": 0.19,
                "def_orb": 0.30,
                "def_ftr": 0.25,
                "is_home": False,
                "is_neutral": True,
            },
            {
                "game_date": date(2025, 11, 10),
                "team_id": 2,
                "opponent_id": 1,
                "off_rating": 97.0,
                "def_rating": 101.0,
                "possessions": 68.0,
                "off_efg": 0.48,
                "off_tov": 0.19,
                "off_orb": 0.30,
                "off_ftr": 0.25,
                "def_efg": 0.50,
                "def_tov": 0.18,
                "def_orb": 0.28,
                "def_ftr": 0.27,
                "is_home": True,
                "is_neutral": True,
            },
            {
                "game_date": date(2025, 11, 1),
                "team_id": 99,
                "opponent_id": 1,
                "off_rating": 88.0,
                "def_rating": 112.0,
                "possessions": 70.0,
                "off_efg": 0.44,
                "off_tov": 0.18,
                "off_orb": 0.27,
                "off_ftr": 0.22,
                "def_efg": 0.52,
                "def_tov": 0.16,
                "def_orb": 0.31,
                "def_ftr": 0.29,
                "is_home": False,
                "is_neutral": False,
            },
        ]
    )

    ratings = calculate_adjusted_ratings(
        team_stats=team_stats,
        as_of_date=date(2025, 11, 15),
        use_recency_weights=False,
        division_i_team_ids={1, 2},
    )

    assert set(ratings) == {1, 2}
    assert ratings[1].games_played == 2
    assert ratings[2].games_played == 1
