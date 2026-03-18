"""Tests for backtest segment helpers."""

import pandas as pd

from packages.eval.segments import (
    build_rank_map,
    classify_conference_segment,
    classify_season_timing,
    classify_spread_bucket,
    classify_tier_matchup,
    summarize_segment,
)
from packages.features.kenpom_ratings import TeamRatings


def test_classify_spread_bucket() -> None:
    assert classify_spread_bucket(0.0) == "0-5"
    assert classify_spread_bucket(4.9) == "0-5"
    assert classify_spread_bucket(5.0) == "5-10"
    assert classify_spread_bucket(-9.9) == "5-10"
    assert classify_spread_bucket(10.0) == "10-15"
    assert classify_spread_bucket(15.0) == "15+"


def test_classify_season_timing() -> None:
    assert classify_season_timing("early") == "early"
    assert classify_season_timing("conference") == "late"
    assert classify_season_timing(None) == "unknown"


def test_classify_conference_segment() -> None:
    assert classify_conference_segment(True, "conference") == "conference"
    assert classify_conference_segment(False, "non_conference") == "non_conference"
    assert classify_conference_segment(False, "tournament") == "tournament"
    assert classify_conference_segment(None, None) == "unknown"


def test_classify_tier_matchup() -> None:
    ratings = {
        1: TeamRatings(
            team_id=1,
            adj_off=110.0,
            adj_def=95.0,
            adj_tempo=70.0,
            adj_em=15.0,
            adj_efg=0.55,
            adj_tov=0.18,
            adj_orb=0.32,
            adj_ftr=0.32,
            adj_efg_def=0.49,
            adj_tov_def=0.18,
            adj_drb=0.71,
            adj_ftr_def=0.30,
            games_played=12,
            sos_off=0.0,
            sos_def=0.0,
            as_of_date=pd.Timestamp("2025-01-01").date(),
            home_off_delta=0.0,
            home_def_delta=0.0,
            away_off_delta=0.0,
            away_def_delta=0.0,
            home_games_played=6,
            away_games_played=6,
            off_std=1.0,
            def_std=1.0,
            tempo_std=1.0,
        ),
        2: TeamRatings(
            team_id=2,
            adj_off=90.0,
            adj_def=105.0,
            adj_tempo=68.0,
            adj_em=-15.0,
            adj_efg=0.45,
            adj_tov=0.22,
            adj_orb=0.28,
            adj_ftr=0.25,
            adj_efg_def=0.52,
            adj_tov_def=0.19,
            adj_drb=0.68,
            adj_ftr_def=0.35,
            games_played=12,
            sos_off=0.0,
            sos_def=0.0,
            as_of_date=pd.Timestamp("2025-01-01").date(),
            home_off_delta=0.0,
            home_def_delta=0.0,
            away_off_delta=0.0,
            away_def_delta=0.0,
            home_games_played=6,
            away_games_played=6,
            off_std=1.0,
            def_std=1.0,
            tempo_std=1.0,
        ),
    }
    rank_map = build_rank_map(ratings)
    total_teams = len(rank_map)
    matchup = classify_tier_matchup(rank_map.get(1), rank_map.get(2), total_teams, top_cut=1, bottom_cut=1)
    assert matchup == "top_vs_bottom"


def test_summarize_segment_counts() -> None:
    df = pd.DataFrame(
        {
            "pred_spread": [1.0, -2.0],
            "pred_total": [140.0, 150.0],
            "actual_spread": [0.0, -1.0],
            "actual_total": [138.0, 152.0],
            "spread_ci_50_lower": [-1.0, -3.0],
            "spread_ci_50_upper": [2.0, 1.0],
            "spread_ci_80_lower": [-3.0, -5.0],
            "spread_ci_80_upper": [4.0, 3.0],
            "spread_ci_95_lower": [-5.0, -7.0],
            "spread_ci_95_upper": [6.0, 5.0],
            "total_ci_50_lower": [130.0, 140.0],
            "total_ci_50_upper": [150.0, 160.0],
            "total_ci_80_lower": [125.0, 135.0],
            "total_ci_80_upper": [155.0, 165.0],
            "total_ci_95_lower": [120.0, 130.0],
            "total_ci_95_upper": [160.0, 170.0],
            "market_spread": [1.0, None],
            "market_total": [140.0, 150.0],
            "closing_spread": [0.5, None],
            "closing_total": [141.0, None],
            "spread_clv": [0.5, None],
            "total_clv": [1.0, None],
        }
    )

    metrics = summarize_segment(df, edge_threshold=0.0)
    assert metrics.total_games == 2
    assert metrics.market_spread_count == 1
    assert metrics.market_total_count == 2
    assert metrics.closing_spread_count == 1
    assert metrics.mean_spread_clv == 0.5
    assert metrics.clv_positive_rate == 1.0
