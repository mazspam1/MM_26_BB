"""
Rolling-origin backtest for NCAA basketball predictions.

Produces append-only backtest_predictions + backtest_runs records,
and optionally CLV reports when market/closing lines are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from packages.common.database import get_connection, init_database
from packages.common.schemas import PredictionRow
from packages.eval.metrics import (
    calculate_accuracy_metrics,
    calculate_calibration_metrics,
    calculate_clv,
    calculate_clv_metrics,
    simulate_betting,
)
from packages.eval.guardrails import apply_min_games_guardrail
from packages.eval.segments import (
    build_rank_map,
    classify_conference_segment,
    classify_season_timing,
    classify_spread_bucket,
    classify_tier_matchup,
    summarize_segment,
)
from packages.features.conference_hca import get_conference_hca_map
from packages.features.kenpom_ratings import TeamRatings, calculate_adjusted_ratings
from packages.models.enhanced_predictor import (
    EnhancedPredictor,
    MODEL_VERSION,
    create_enhanced_predictor,
)

logger = structlog.get_logger()


@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    model_version: str = MODEL_VERSION
    edge_threshold: float = 0.0
    bet_time: time = time(10, 0)
    use_saved_calibration: bool = True


@dataclass
class BacktestSummary:
    run_id: str
    start_date: date
    end_date: date
    total_games: int
    games_with_lines: int
    spread_mae: float
    spread_rmse: float
    total_mae: float
    total_rmse: float
    spread_50_coverage: float
    spread_80_coverage: float
    spread_95_coverage: float
    mean_spread_clv: float
    mean_total_clv: float
    clv_positive_rate: float
    spread_wins: int
    spread_losses: int
    spread_pushes: int
    total_wins: int
    total_losses: int
    total_pushes: int
    simulated_roi: float


def _market_spread_from_snapshot(
    spread_home: Optional[float],
    spread_away: Optional[float],
) -> Optional[float]:
    if spread_home is not None:
        return -float(spread_home)
    if spread_away is not None:
        return float(spread_away)
    return None


def _fetch_line_snapshot(
    conn,
    game_id: int,
    as_of_ts: datetime,
) -> tuple[Optional[float], Optional[float], Optional[datetime]]:
    row = conn.execute(
        """
        SELECT spread_home, spread_away, total_line, snapshot_timestamp
        FROM line_snapshots
        WHERE game_id = ?
          AND snapshot_timestamp <= ?
        ORDER BY snapshot_timestamp DESC
        LIMIT 1
        """,
        (game_id, as_of_ts),
    ).fetchone()

    if not row:
        return None, None, None

    market_spread = _market_spread_from_snapshot(row[0], row[1])
    market_total = row[2] if row[2] is not None else None
    return market_spread, market_total, row[3]


def _fetch_splits_snapshot(
    conn,
    game_id: int,
    as_of_ts: datetime,
) -> tuple[Optional[float], Optional[float], Optional[datetime]]:
    row = conn.execute(
        """
        SELECT spread_line_home, total_line, snapshot_timestamp
        FROM betting_splits
        WHERE game_id = ?
          AND snapshot_timestamp <= ?
        ORDER BY snapshot_timestamp DESC
        LIMIT 1
        """,
        (game_id, as_of_ts),
    ).fetchone()

    if not row:
        return None, None, None

    return row[0], row[1], row[2]


def _resolve_market_lines(
    conn,
    game_id: int,
    bet_ts: datetime,
    close_ts: datetime,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    # Prefer line_snapshots, fall back to betting_splits.
    market_spread, market_total, _ = _fetch_line_snapshot(conn, game_id, bet_ts)
    if market_spread is None and market_total is None:
        market_spread, market_total, _ = _fetch_splits_snapshot(conn, game_id, bet_ts)

    closing_spread, closing_total, _ = _fetch_line_snapshot(conn, game_id, close_ts)
    if closing_spread is None and closing_total is None:
        closing_spread, closing_total, _ = _fetch_splits_snapshot(conn, game_id, close_ts)

    return market_spread, market_total, closing_spread, closing_total


def _prediction_timestamp(
    game_date: date, game_datetime: Optional[datetime], bet_time: time
) -> datetime:
    ts = datetime.combine(game_date, bet_time)
    if game_datetime and ts > game_datetime:
        return game_datetime - timedelta(hours=1)
    return ts


def run_backtest(config: BacktestConfig, save_to_db: bool = True) -> BacktestSummary:
    init_database()
    run_id = datetime.utcnow().strftime("bt_%Y%m%d_%H%M%S")

    with get_connection() as conn:
        games_df = conn.execute(
            """
            SELECT
                game_id, CAST(game_date AS DATE) AS game_date, game_datetime,
                home_team_id, away_team_id, neutral_site,
                home_score, away_score,
                conference_game, season_phase
            FROM games
            WHERE CAST(game_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            ORDER BY CAST(game_date AS DATE), game_datetime
            """,
            (config.start_date.isoformat(), config.end_date.isoformat()),
        ).fetchdf()

        if games_df.empty:
            raise ValueError("No final games found for backtest range")

        games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.date

        team_stats = conn.execute("SELECT * FROM team_game_stats").fetchdf()
        if team_stats.empty:
            raise ValueError("team_game_stats is empty - run boxscore ingestion first")

        rest_df = conn.execute("SELECT game_id, team_id, rest_days FROM team_game_stats").fetchdf()

        conf_df = conn.execute(
            "SELECT team_id, conference_id, conference_name FROM team_conference_ids"
        ).fetchdf()

    rest_lookup = {
        (int(row.game_id), int(row.team_id)): int(row.rest_days)
        for row in rest_df.itertuples(index=False)
    }
    conf_map = {
        int(row.team_id): int(row.conference_id)
        for row in conf_df.itertuples(index=False)
        if row.conference_id is not None
    }

    hca_map = get_conference_hca_map(conf_df) if not conf_df.empty else None

    predictor = create_enhanced_predictor(use_saved_calibration=config.use_saved_calibration)
    predictions: list[
        tuple[
            int,
            datetime,
            PredictionRow,
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            dict[str, str],
        ]
    ] = []

    with get_connection() as conn:
        for game_date in sorted(games_df["game_date"].unique()):
            as_of_date = game_date - timedelta(days=1)
            ratings = calculate_adjusted_ratings(
                team_stats=team_stats,
                as_of_date=as_of_date,
                use_recency_weights=True,
                conference_hca=hca_map,
            )
            rank_map = build_rank_map(ratings)
            total_teams = len(rank_map)

            date_games = games_df[games_df["game_date"] == game_date]
            for row in date_games.itertuples(index=False):
                home_id = int(row.home_team_id)
                away_id = int(row.away_team_id)
                if home_id not in ratings or away_id not in ratings:
                    continue

                game_datetime = row.game_datetime if pd.notna(row.game_datetime) else None
                if isinstance(game_datetime, pd.Timestamp):
                    game_datetime = game_datetime.to_pydatetime()
                elif isinstance(game_datetime, str):
                    game_datetime = datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))

                bet_ts = _prediction_timestamp(row.game_date, game_datetime, config.bet_time)
                close_ts = game_datetime or datetime.combine(row.game_date, time(23, 59))

                market_spread, market_total, closing_spread, closing_total = _resolve_market_lines(
                    conn, int(row.game_id), bet_ts, close_ts
                )

                home_rest = rest_lookup.get((int(row.game_id), home_id), 2)
                away_rest = rest_lookup.get((int(row.game_id), away_id), 2)
                home_conf_id = conf_map.get(home_id)

                pred = predictor.predict_game(
                    home_ratings=ratings[home_id],
                    away_ratings=ratings[away_id],
                    game_id=int(row.game_id),
                    is_neutral=bool(row.neutral_site),
                    home_rest_days=home_rest,
                    away_rest_days=away_rest,
                    home_conference_id=home_conf_id,
                    market_spread=market_spread,
                    market_total=market_total,
                )
                pred_row = predictor.to_prediction_row(
                    prediction=pred,
                    market_spread=market_spread,
                    market_total=market_total,
                )
                pred_row = apply_min_games_guardrail(
                    pred_row,
                    home_games_played=ratings[home_id].games_played,
                    away_games_played=ratings[away_id].games_played,
                )

                segment_info = {
                    "conference_segment": classify_conference_segment(
                        row.conference_game if hasattr(row, "conference_game") else None,
                        row.season_phase if hasattr(row, "season_phase") else None,
                    ),
                    "season_timing": classify_season_timing(
                        row.season_phase if hasattr(row, "season_phase") else None
                    ),
                    "spread_bucket": classify_spread_bucket(pred_row.proj_spread),
                    "tier_matchup": classify_tier_matchup(
                        rank_map.get(home_id),
                        rank_map.get(away_id),
                        total_teams=total_teams,
                    ),
                }

                predictions.append(
                    (
                        int(row.game_id),
                        bet_ts,
                        pred_row,
                        market_spread,
                        market_total,
                        closing_spread,
                        closing_total,
                        segment_info,
                    )
                )

    if not predictions:
        raise ValueError("No predictions generated for backtest range")

    # Build actuals lookup
    actuals = {
        int(row.game_id): (
            float(row.home_score - row.away_score),
            float(row.home_score + row.away_score),
        )
        for row in games_df.itertuples(index=False)
    }

    pred_spreads = []
    pred_totals = []
    actual_spreads = []
    actual_totals = []
    spread_ci_50_lower = []
    spread_ci_50_upper = []
    spread_ci_80_lower = []
    spread_ci_80_upper = []
    spread_ci_95_lower = []
    spread_ci_95_upper = []
    total_ci_50_lower = []
    total_ci_50_upper = []
    total_ci_80_lower = []
    total_ci_80_upper = []
    total_ci_95_lower = []
    total_ci_95_upper = []

    clv_spreads = []
    clv_totals = []
    clv_pred_spreads = []
    clv_pred_totals = []
    clv_market_spreads = []
    clv_market_totals = []
    clv_closing_spreads = []
    clv_closing_totals = []
    clv_spread_sides = []
    clv_total_sides = []

    bet_pred_spreads = []
    bet_market_spreads = []
    bet_actual_spreads = []
    bet_pred_totals = []
    bet_market_totals = []
    bet_actual_totals = []

    backtest_rows = []
    clv_rows = []
    segment_records = []

    for (
        game_id,
        bet_ts,
        pred_row,
        market_spread,
        market_total,
        closing_spread,
        closing_total,
        segment_info,
    ) in predictions:
        actual = actuals.get(game_id)
        if actual is None:
            continue

        actual_spread, actual_total = actual

        pred_spreads.append(pred_row.proj_spread)
        pred_totals.append(pred_row.proj_total)
        actual_spreads.append(actual_spread)
        actual_totals.append(actual_total)

        spread_ci_50_lower.append(pred_row.spread_ci_50_lower)
        spread_ci_50_upper.append(pred_row.spread_ci_50_upper)
        spread_ci_80_lower.append(pred_row.spread_ci_80_lower)
        spread_ci_80_upper.append(pred_row.spread_ci_80_upper)
        spread_ci_95_lower.append(pred_row.spread_ci_95_lower)
        spread_ci_95_upper.append(pred_row.spread_ci_95_upper)
        total_ci_50_lower.append(pred_row.total_ci_50_lower)
        total_ci_50_upper.append(pred_row.total_ci_50_upper)
        total_ci_80_lower.append(pred_row.total_ci_80_lower)
        total_ci_80_upper.append(pred_row.total_ci_80_upper)
        total_ci_95_lower.append(pred_row.total_ci_95_lower)
        total_ci_95_upper.append(pred_row.total_ci_95_upper)

        spread_clv = None
        total_clv = None

        spread_bet_side = None
        total_bet_side = None

        if market_spread is not None:
            bet_pred_spreads.append(pred_row.proj_spread)
            bet_market_spreads.append(market_spread)
            bet_actual_spreads.append(actual_spread)

            if closing_spread is not None:
                edge = pred_row.edge_vs_market_spread or 0.0
                spread_bet_side = "home" if edge > 0 else "away"
                spread_clv = calculate_clv(market_spread, closing_spread, spread_bet_side)
                clv_spreads.append(spread_clv)
                clv_pred_spreads.append(pred_row.proj_spread)
                clv_market_spreads.append(market_spread)
                clv_closing_spreads.append(closing_spread)
                clv_spread_sides.append(spread_bet_side)

        if market_total is not None:
            bet_pred_totals.append(pred_row.proj_total)
            bet_market_totals.append(market_total)
            bet_actual_totals.append(actual_total)

            if closing_total is not None:
                edge = pred_row.edge_vs_market_total or 0.0
                total_bet_side = "over" if edge > 0 else "under"
                total_clv = calculate_clv(market_total, closing_total, total_bet_side)
                clv_totals.append(total_clv)
                clv_pred_totals.append(pred_row.proj_total)
                clv_market_totals.append(market_total)
                clv_closing_totals.append(closing_total)
                clv_total_sides.append(total_bet_side)

        backtest_rows.append(
            (
                run_id,
                game_id,
                bet_ts.isoformat(),
                pred_row.model_version,
                pred_row.proj_home_score,
                pred_row.proj_away_score,
                pred_row.proj_spread,
                pred_row.proj_total,
                pred_row.proj_possessions,
                pred_row.home_win_prob,
                pred_row.spread_ci_50_lower,
                pred_row.spread_ci_50_upper,
                pred_row.spread_ci_80_lower,
                pred_row.spread_ci_80_upper,
                pred_row.spread_ci_95_lower,
                pred_row.spread_ci_95_upper,
                pred_row.total_ci_50_lower,
                pred_row.total_ci_50_upper,
                pred_row.total_ci_80_lower,
                pred_row.total_ci_80_upper,
                pred_row.total_ci_95_lower,
                pred_row.total_ci_95_upper,
                market_spread,
                pred_row.edge_vs_market_spread,
                market_total,
                pred_row.edge_vs_market_total,
                closing_spread,
                closing_total,
                spread_clv,
                total_clv,
                pred_row.recommended_side,
                pred_row.recommended_units,
                pred_row.confidence_rating,
            )
        )

        if (
            market_spread is not None
            and market_total is not None
            and closing_spread is not None
            and closing_total is not None
        ):
            clv_rows.append(
                (
                    game_id,
                    bet_ts.isoformat(),
                    bet_ts.isoformat(),
                    pred_row.proj_spread,
                    pred_row.proj_total,
                    market_spread,
                    market_total,
                    closing_spread,
                    closing_total,
                    spread_clv,
                    total_clv,
                    actual_spread,
                    int(actual_total),
                    spread_bet_side,
                    None,
                    total_bet_side,
                    None,
                )
            )

        segment_records.append(
            {
                "game_id": game_id,
                "pred_spread": pred_row.proj_spread,
                "pred_total": pred_row.proj_total,
                "actual_spread": actual_spread,
                "actual_total": actual_total,
                "spread_ci_50_lower": pred_row.spread_ci_50_lower,
                "spread_ci_50_upper": pred_row.spread_ci_50_upper,
                "spread_ci_80_lower": pred_row.spread_ci_80_lower,
                "spread_ci_80_upper": pred_row.spread_ci_80_upper,
                "spread_ci_95_lower": pred_row.spread_ci_95_lower,
                "spread_ci_95_upper": pred_row.spread_ci_95_upper,
                "total_ci_50_lower": pred_row.total_ci_50_lower,
                "total_ci_50_upper": pred_row.total_ci_50_upper,
                "total_ci_80_lower": pred_row.total_ci_80_lower,
                "total_ci_80_upper": pred_row.total_ci_80_upper,
                "total_ci_95_lower": pred_row.total_ci_95_lower,
                "total_ci_95_upper": pred_row.total_ci_95_upper,
                "market_spread": market_spread,
                "market_total": market_total,
                "closing_spread": closing_spread,
                "closing_total": closing_total,
                "spread_clv": spread_clv,
                "total_clv": total_clv,
                "segment_conference": segment_info.get("conference_segment", "unknown"),
                "segment_timing": segment_info.get("season_timing", "unknown"),
                "segment_spread_bucket": segment_info.get("spread_bucket", "unknown"),
                "segment_tier_matchup": segment_info.get("tier_matchup", "unknown"),
            }
        )

    pred_spreads_arr = np.array(pred_spreads, dtype=float)
    pred_totals_arr = np.array(pred_totals, dtype=float)
    actual_spreads_arr = np.array(actual_spreads, dtype=float)
    actual_totals_arr = np.array(actual_totals, dtype=float)

    accuracy = calculate_accuracy_metrics(
        pred_spreads_arr, actual_spreads_arr, pred_totals_arr, actual_totals_arr
    )
    calibration = calculate_calibration_metrics(
        actual_spreads_arr,
        np.array(spread_ci_50_lower),
        np.array(spread_ci_50_upper),
        np.array(spread_ci_80_lower),
        np.array(spread_ci_80_upper),
        np.array(spread_ci_95_lower),
        np.array(spread_ci_95_upper),
        actual_totals_arr,
        np.array(total_ci_50_lower),
        np.array(total_ci_50_upper),
        np.array(total_ci_80_lower),
        np.array(total_ci_80_upper),
        np.array(total_ci_95_lower),
        np.array(total_ci_95_upper),
    )

    betting = simulate_betting(
        np.array(bet_pred_spreads, dtype=float),
        np.array(bet_market_spreads, dtype=float),
        np.array(bet_actual_spreads, dtype=float),
        np.array(bet_pred_totals, dtype=float),
        np.array(bet_market_totals, dtype=float),
        np.array(bet_actual_totals, dtype=float),
        edge_threshold=config.edge_threshold,
    )

    clv_metrics = None
    mean_spread_clv = float(np.mean(clv_spreads)) if clv_spreads else 0.0
    mean_total_clv = float(np.mean(clv_totals)) if clv_totals else 0.0
    clv_positive_rate = float(np.mean(np.array(clv_spreads) > 0)) if clv_spreads else 0.0

    if clv_market_spreads and clv_market_totals:
        clv_metrics = calculate_clv_metrics(
            np.array(clv_pred_spreads, dtype=float),
            np.array(clv_market_spreads, dtype=float),
            np.array(clv_closing_spreads, dtype=float),
            clv_spread_sides,
            np.array(clv_pred_totals, dtype=float),
            np.array(clv_market_totals, dtype=float),
            np.array(clv_closing_totals, dtype=float),
            clv_total_sides,
        )
        mean_spread_clv = clv_metrics.mean_spread_clv
        mean_total_clv = clv_metrics.mean_total_clv
        clv_positive_rate = clv_metrics.spread_clv_positive_rate

    roi_values = []
    if betting.n_spread_bets > 0:
        roi_values.append(betting.spread_roi)
    if betting.n_total_bets > 0:
        roi_values.append(betting.total_roi)
    simulated_roi = float(np.mean(roi_values)) if roi_values else 0.0

    summary = BacktestSummary(
        run_id=run_id,
        start_date=config.start_date,
        end_date=config.end_date,
        total_games=accuracy.n_games,
        games_with_lines=len(clv_market_spreads),
        spread_mae=accuracy.spread_mae,
        spread_rmse=accuracy.spread_rmse,
        total_mae=accuracy.total_mae,
        total_rmse=accuracy.total_rmse,
        spread_50_coverage=calibration.spread_50_coverage,
        spread_80_coverage=calibration.spread_80_coverage,
        spread_95_coverage=calibration.spread_95_coverage,
        mean_spread_clv=mean_spread_clv,
        mean_total_clv=mean_total_clv,
        clv_positive_rate=clv_positive_rate,
        spread_wins=betting.spread_wins,
        spread_losses=betting.spread_losses,
        spread_pushes=betting.spread_pushes,
        total_wins=betting.total_wins,
        total_losses=betting.total_losses,
        total_pushes=betting.total_pushes,
        simulated_roi=simulated_roi,
    )

    segment_rows = []
    if segment_records:
        segment_df = pd.DataFrame(segment_records)
        segment_cols = {
            "conference_game": "segment_conference",
            "season_timing": "segment_timing",
            "spread_bucket": "segment_spread_bucket",
            "tier_matchup": "segment_tier_matchup",
        }

        for segment_type, col in segment_cols.items():
            for segment_value, seg_df in segment_df.groupby(col):
                metrics = summarize_segment(seg_df, config.edge_threshold)
                metrics.segment_type = segment_type
                metrics.segment_value = str(segment_value)
                segment_rows.append(
                    (
                        run_id,
                        metrics.segment_type,
                        metrics.segment_value,
                        metrics.total_games,
                        metrics.market_spread_count,
                        metrics.market_total_count,
                        metrics.closing_spread_count,
                        metrics.closing_total_count,
                        metrics.spread_mae,
                        metrics.spread_rmse,
                        metrics.total_mae,
                        metrics.total_rmse,
                        metrics.spread_50_coverage,
                        metrics.spread_80_coverage,
                        metrics.spread_95_coverage,
                        metrics.total_50_coverage,
                        metrics.total_80_coverage,
                        metrics.total_95_coverage,
                        metrics.mean_spread_clv,
                        metrics.mean_total_clv,
                        metrics.clv_positive_rate,
                        metrics.spread_wins,
                        metrics.spread_losses,
                        metrics.spread_pushes,
                        metrics.total_wins,
                        metrics.total_losses,
                        metrics.total_pushes,
                        metrics.simulated_roi,
                    )
                )

    if save_to_db:
        with get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO backtest_predictions (
                    run_id, game_id, prediction_timestamp, model_version,
                    proj_home_score, proj_away_score, proj_spread, proj_total,
                    proj_possessions, home_win_prob,
                    spread_ci_50_lower, spread_ci_50_upper,
                    spread_ci_80_lower, spread_ci_80_upper,
                    spread_ci_95_lower, spread_ci_95_upper,
                    total_ci_50_lower, total_ci_50_upper,
                    total_ci_80_lower, total_ci_80_upper,
                    total_ci_95_lower, total_ci_95_upper,
                    market_spread, edge_vs_market_spread,
                    market_total, edge_vs_market_total,
                    closing_spread, closing_total,
                    spread_clv, total_clv,
                    recommended_side, recommended_units, confidence_rating
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                backtest_rows,
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO backtest_runs (
                    run_id, start_date, end_date, model_version, edge_threshold,
                    total_games, games_with_lines,
                    spread_mae, spread_rmse, total_mae, total_rmse,
                    spread_50_coverage, spread_80_coverage, spread_95_coverage,
                    mean_spread_clv, mean_total_clv, clv_positive_rate,
                    spread_wins, spread_losses, spread_pushes,
                    total_wins, total_losses, total_pushes,
                    simulated_roi
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.run_id,
                    summary.start_date.isoformat(),
                    summary.end_date.isoformat(),
                    config.model_version,
                    config.edge_threshold,
                    summary.total_games,
                    summary.games_with_lines,
                    summary.spread_mae,
                    summary.spread_rmse,
                    summary.total_mae,
                    summary.total_rmse,
                    summary.spread_50_coverage,
                    summary.spread_80_coverage,
                    summary.spread_95_coverage,
                    summary.mean_spread_clv,
                    summary.mean_total_clv,
                    summary.clv_positive_rate,
                    summary.spread_wins,
                    summary.spread_losses,
                    summary.spread_pushes,
                    summary.total_wins,
                    summary.total_losses,
                    summary.total_pushes,
                    summary.simulated_roi,
                ),
            )

            if segment_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO backtest_segments (
                        run_id, segment_type, segment_value,
                        total_games, market_spread_count, market_total_count,
                        closing_spread_count, closing_total_count,
                        spread_mae, spread_rmse, total_mae, total_rmse,
                        spread_50_coverage, spread_80_coverage, spread_95_coverage,
                        total_50_coverage, total_80_coverage, total_95_coverage,
                        mean_spread_clv, mean_total_clv, clv_positive_rate,
                        spread_wins, spread_losses, spread_pushes,
                        total_wins, total_losses, total_pushes,
                        simulated_roi
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    segment_rows,
                )

            if clv_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO clv_reports (
                        game_id, prediction_timestamp, bet_timestamp,
                        our_spread, our_total,
                        market_spread_at_bet, market_total_at_bet,
                        closing_spread, closing_total,
                        spread_clv, total_clv,
                        actual_spread, actual_total,
                        spread_bet_side, spread_bet_won,
                        total_bet_side, total_bet_won
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    clv_rows,
                )

    logger.info(
        "Backtest complete",
        run_id=summary.run_id,
        games=summary.total_games,
        spread_mae=summary.spread_mae,
        total_mae=summary.total_mae,
    )

    return summary
