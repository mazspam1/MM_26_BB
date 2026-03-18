"""
Generate predictions for a target date and persist to the database.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import structlog

project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from packages.common.database import get_connection
from packages.common.sportsdataverse_mbb import load_mbb
from packages.eval.guardrails import apply_min_games_guardrail
from packages.features.kenpom_ratings import TeamRatings
from packages.models.enhanced_predictor import EnhancedPredictor, MODEL_VERSION

logger = structlog.get_logger()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_ratings() -> dict[int, TeamRatings]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                team_id, as_of_date,
                adj_offensive_efficiency, adj_defensive_efficiency, adj_tempo, adj_em,
                off_efg, off_tov, off_orb, off_ftr,
                def_efg, def_tov, def_drb, def_ftr,
                games_played, sos_off, sos_def,
                home_off_delta, home_def_delta, away_off_delta, away_def_delta,
                home_games_played, away_games_played,
                off_rating_std, def_rating_std, tempo_std
            FROM team_strengths
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM team_strengths)
            """
        ).fetchall()

    ratings: dict[int, TeamRatings] = {}
    for row in rows:
        ratings[row[0]] = TeamRatings(
            team_id=row[0],
            adj_off=row[2],
            adj_def=row[3],
            adj_tempo=row[4],
            adj_em=row[5],
            adj_efg=row[6],
            adj_tov=row[7],
            adj_orb=row[8],
            adj_ftr=row[9],
            adj_efg_def=row[10],
            adj_tov_def=row[11],
            adj_drb=row[12],
            adj_ftr_def=row[13],
            games_played=row[14],
            sos_off=row[15],
            sos_def=row[16],
            as_of_date=date.fromisoformat(row[1]) if isinstance(row[1], str) else row[1],
            home_off_delta=row[17],
            home_def_delta=row[18],
            away_off_delta=row[19],
            away_def_delta=row[20],
            home_games_played=row[21],
            away_games_played=row[22],
            off_std=row[23],
            def_std=row[24],
            tempo_std=row[25],
        )

    return ratings


def load_valid_game_ids(target_date: date) -> set[int]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT game_id FROM games WHERE game_date = ?",
            (target_date.isoformat(),),
        ).fetchall()
    return {row[0] for row in rows}


def run_predictions(target_date: date) -> int:
    logger.info("Generating predictions", date=target_date.isoformat(), model_version=MODEL_VERSION)

    ratings = load_ratings()
    if not ratings:
        raise RuntimeError("No ratings found. Run: .\\start.ps1 ratings")

    mbb = load_mbb()
    schedule = mbb.espn_mbb_schedule(
        dates=target_date.strftime("%Y%m%d"),
        groups=50,
        return_as_pandas=True,
    )

    valid_game_ids = load_valid_game_ids(target_date)
    predictor = EnhancedPredictor()

    predictions = []
    for _, game in schedule.iterrows():
        try:
            home_id = int(game.get("home_id"))
            away_id = int(game.get("away_id"))
            game_id = int(game.get("game_id")) if game.get("game_id") else 0
        except (ValueError, TypeError):
            continue

        if game_id not in valid_game_ids:
            continue
        if home_id not in ratings or away_id not in ratings:
            continue

        pred = predictor.predict_game(
            home_ratings=ratings[home_id],
            away_ratings=ratings[away_id],
            game_id=game_id,
            is_neutral=bool(game.get("neutral_site", False)),
        )
        pred_row = predictor.to_prediction_row(prediction=pred)
        pred_row = apply_min_games_guardrail(
            pred_row,
            home_games_played=ratings[home_id].games_played,
            away_games_played=ratings[away_id].games_played,
        )
        predictions.append((game, pred_row))

    with get_connection() as conn:
        for _, pred in predictions:
            conn.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    game_id, prediction_timestamp, model_version,
                    proj_home_score, proj_away_score,
                    proj_spread, proj_total, proj_possessions,
                    home_win_prob,
                    efficiency_spread, hca_adjustment, travel_adjustment, rest_adjustment, four_factors_adjustment,
                    spread_ci_50_lower, spread_ci_50_upper,
                    spread_ci_80_lower, spread_ci_80_upper,
                    spread_ci_95_lower, spread_ci_95_upper,
                    total_ci_50_lower, total_ci_50_upper,
                    total_ci_80_lower, total_ci_80_upper,
                    total_ci_95_lower, total_ci_95_upper,
                    market_spread, edge_vs_market_spread,
                    market_total, edge_vs_market_total,
                    recommended_side, recommended_units, confidence_rating
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pred.game_id,
                    pred.prediction_timestamp.isoformat(),
                    pred.model_version,
                    pred.proj_home_score,
                    pred.proj_away_score,
                    pred.proj_spread,
                    pred.proj_total,
                    pred.proj_possessions,
                    pred.home_win_prob,
                    pred.efficiency_spread,
                    pred.hca_adjustment,
                    pred.travel_adjustment,
                    pred.rest_adjustment,
                    pred.four_factors_adjustment,
                    pred.spread_ci_50_lower,
                    pred.spread_ci_50_upper,
                    pred.spread_ci_80_lower,
                    pred.spread_ci_80_upper,
                    pred.spread_ci_95_lower,
                    pred.spread_ci_95_upper,
                    pred.total_ci_50_lower,
                    pred.total_ci_50_upper,
                    pred.total_ci_80_lower,
                    pred.total_ci_80_upper,
                    pred.total_ci_95_lower,
                    pred.total_ci_95_upper,
                    pred.market_spread,
                    pred.edge_vs_market_spread,
                    pred.market_total,
                    pred.edge_vs_market_total,
                    pred.recommended_side,
                    pred.recommended_units,
                    pred.confidence_rating,
                ),
            )

    logger.info("Predictions saved", count=len(predictions))
    return len(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions for a date")
    parser.add_argument("--date", type=str, default=date.today().isoformat(), help="Target date (YYYY-MM-DD)")
    args = parser.parse_args()

    target_date = parse_date(args.date)
    count = run_predictions(target_date)
    print(f"Generated {count} predictions for {target_date.isoformat()}")


if __name__ == "__main__":
    main()
