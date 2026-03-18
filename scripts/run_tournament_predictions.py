"""
March Madness tournament prediction pipeline.

Loads the live NCAA bracket, ensures tournament teams have ratings, prices all
known games, then simulates the full bracket with dynamic matchup repricing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import structlog

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packages.common.database import get_connection, init_database
from packages.common.season import infer_season_year, season_start_date
from packages.features.conference_hca import get_conference_hca_map
from packages.features.kenpom_ratings import (
    TeamRatings,
    calculate_adjusted_ratings,
    save_ratings_to_db,
)
from packages.ingest.espn_enhanced import ingest_team_stats_for_date_range
from packages.ingest.espn_api import fetch_schedule, save_games_to_db
from packages.ingest.espn_pickcenter import ingest_pickcenter_for_games
from packages.ingest.tournament_bracket import (
    ROUND_NAMES,
    load_2026_bracket,
    load_bracket_from_db,
)
from packages.models.enhanced_predictor import create_enhanced_predictor
from packages.models.tournament_predictor import (
    TournamentPredictor,
    TournamentPrediction,
    save_tournament_prediction,
)
from packages.simulation import BracketSimulator, GamePrediction, save_simulation_results

logger = structlog.get_logger()


def sync_tournament_schedule(start_date: date, end_date: date) -> None:
    """Save scheduled tournament games from free ESPN data into the games table."""
    current = start_date
    while current <= end_date:
        games = fetch_schedule(current)
        if games:
            save_games_to_db(games, skip_fk_errors=True)
        current = current + timedelta(days=1)


def sync_tournament_pickcenter_lines(start_date: date, end_date: date) -> int:
    """Ingest free ESPN pickcenter lines for tournament games in date window."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT game_id
            FROM games
            WHERE CAST(game_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND status = 'scheduled'
            ORDER BY game_date, game_id
            """,
            (start_date.isoformat(), end_date.isoformat()),
        ).fetchall()

    game_ids = [int(row[0]) for row in rows]
    if not game_ids:
        return 0
    return ingest_pickcenter_for_games(game_ids)


def _market_spread_from_snapshot(
    spread_home: Optional[float],
    spread_away: Optional[float],
) -> Optional[float]:
    if spread_home is not None:
        return -float(spread_home)
    if spread_away is not None:
        return float(spread_away)
    return None


def lookup_market_lines(
    slot: dict,
    team_a_id: int,
    team_b_id: int,
) -> tuple[Optional[float], Optional[float]]:
    """Return market spread in team-a orientation plus total, if available."""
    slot_date = slot.get("game_date")
    if slot_date is None:
        return None, None
    if isinstance(slot_date, str):
        slot_date = date.fromisoformat(slot_date)

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT game_id, home_team_id, away_team_id
            FROM games
            WHERE CAST(game_date AS DATE) = CAST(? AS DATE)
              AND ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
            LIMIT 1
            """,
            (slot_date.isoformat(), team_a_id, team_b_id, team_b_id, team_a_id),
        ).fetchone()

        if row is None:
            return None, None

        game_id = int(row[0])
        home_team_id = int(row[1])

        split_row = conn.execute(
            """
            SELECT spread_line_home, total_line
            FROM betting_splits
            WHERE game_id = ?
            ORDER BY snapshot_timestamp DESC
            LIMIT 1
            """,
            (game_id,),
        ).fetchone()
        if split_row is not None and (split_row[0] is not None or split_row[1] is not None):
            market_spread = float(split_row[0]) if split_row[0] is not None else None
            market_total = float(split_row[1]) if split_row[1] is not None else None
            if market_spread is not None and home_team_id != team_a_id:
                market_spread = -market_spread
            return market_spread, market_total

        line_row = conn.execute(
            """
            SELECT spread_home, spread_away, total_line
            FROM line_snapshots
            WHERE game_id = ?
            ORDER BY snapshot_timestamp DESC
            LIMIT 1
            """,
            (game_id,),
        ).fetchone()

    if line_row is None:
        return None, None

    market_spread = _market_spread_from_snapshot(line_row[0], line_row[1])
    if market_spread is not None and home_team_id != team_a_id:
        market_spread = -market_spread
    market_total = float(line_row[2]) if line_row[2] is not None else None
    return market_spread, market_total


def _build_feeds_into_map(bracket_slots: list[dict]) -> dict[int, dict[str, int]]:
    feeds_into: dict[int, dict[str, int]] = {}
    for slot in bracket_slots:
        next_slot_id = slot.get("next_slot_id")
        position = str(slot.get("victor_game_position") or "").strip().lower()
        if next_slot_id is None or position not in {"top", "bottom"}:
            continue
        feeds_into.setdefault(int(next_slot_id), {})[position.title()] = int(slot["slot_id"])
    return feeds_into


def _resolve_slot_teams(
    slot: dict,
    winners: dict[int, int],
    feeds_into: dict[int, dict[str, int]],
) -> tuple[Optional[int], Optional[int]]:
    team_a_id = int(slot["team_a_id"]) if slot.get("team_a_id") is not None else None
    team_b_id = int(slot["team_b_id"]) if slot.get("team_b_id") is not None else None
    source_map = feeds_into.get(int(slot["slot_id"]), {})

    top_source = source_map.get("Top")
    bottom_source = source_map.get("Bottom")
    if top_source is not None:
        team_a_id = winners.get(top_source, team_a_id)
    if bottom_source is not None:
        team_b_id = winners.get(bottom_source, team_b_id)

    return team_a_id, team_b_id


def _round_name(round_value: int) -> str:
    return ROUND_NAMES.get(round_value, f"Round {round_value}")


def _game_card(
    slot: dict,
    prediction: TournamentPrediction,
    sim_prediction: GamePrediction,
    winner_id: int,
    team_names: dict[int, str],
) -> dict:
    team_a_is_higher = int(sim_prediction.team_a_id) == int(prediction.higher_seed_team_id)
    team_a_score = prediction.proj_higher_score if team_a_is_higher else prediction.proj_lower_score
    team_b_score = prediction.proj_lower_score if team_a_is_higher else prediction.proj_higher_score
    team_a_p10 = prediction.higher_score_p10 if team_a_is_higher else prediction.lower_score_p10
    team_a_p90 = prediction.higher_score_p90 if team_a_is_higher else prediction.lower_score_p90
    team_b_p10 = prediction.lower_score_p10 if team_a_is_higher else prediction.higher_score_p10
    team_b_p90 = prediction.lower_score_p90 if team_a_is_higher else prediction.higher_score_p90
    winner_name = team_names.get(winner_id, f"Team {winner_id}")

    return {
        "slot_id": int(slot["slot_id"]),
        "round": int(slot["round"]),
        "round_name": _round_name(int(slot["round"])),
        "game_in_round": int(slot.get("game_in_round") or 0),
        "team_a_id": int(sim_prediction.team_a_id),
        "team_a_name": sim_prediction.team_a_name,
        "team_a_seed": int(sim_prediction.team_a_seed),
        "team_a_score": round(float(team_a_score), 1),
        "team_a_score_p10": round(float(team_a_p10), 1),
        "team_a_score_p90": round(float(team_a_p90), 1),
        "team_a_win_prob": round(float(sim_prediction.team_a_win_prob), 3),
        "team_b_id": int(sim_prediction.team_b_id),
        "team_b_name": sim_prediction.team_b_name,
        "team_b_seed": int(sim_prediction.team_b_seed),
        "team_b_score": round(float(team_b_score), 1),
        "team_b_score_p10": round(float(team_b_p10), 1),
        "team_b_score_p90": round(float(team_b_p90), 1),
        "team_b_win_prob": round(float(sim_prediction.team_b_win_prob), 3),
        "projected_total": round(float(prediction.proj_total), 1),
        "projected_spread_team_a": round(float(sim_prediction.proj_spread), 1),
        "projected_possessions": round(float(prediction.proj_possessions), 1),
        "upset_prob": round(float(prediction.upset_prob), 3),
        "confidence_rating": prediction.confidence_rating,
        "market_spread_team_a": prediction.market_spread
        if team_a_is_higher
        else (-prediction.market_spread if prediction.market_spread is not None else None),
        "market_total": prediction.market_total,
        "edge_vs_market_spread": prediction.edge_vs_market_spread
        if team_a_is_higher
        else (
            -prediction.edge_vs_market_spread
            if prediction.edge_vs_market_spread is not None
            else None
        ),
        "edge_vs_market_total": prediction.edge_vs_market_total,
        "recommended_side": prediction.recommended_side,
        "recommended_units": prediction.recommended_units,
        "likely_winner_id": int(winner_id),
        "likely_winner_name": winner_name,
        "likely_winner_prob": round(
            float(
                sim_prediction.team_a_win_prob
                if winner_id == sim_prediction.team_a_id
                else sim_prediction.team_b_win_prob
            ),
            3,
        ),
    }


def build_best_bracket_report(
    bracket_slots: list[dict],
    ratings: dict[int, TeamRatings],
    seed_map: dict[int, int],
    tournament_predictor: TournamentPredictor,
    team_names: dict[int, str],
) -> dict:
    """Build a deterministic best-bracket path by repricing each round."""
    sorted_slots = sorted(
        bracket_slots, key=lambda slot: (int(slot["round"]), int(slot["slot_id"]))
    )
    feeds_into = _build_feeds_into_map(bracket_slots)
    winners: dict[int, int] = {}
    game_cards: list[dict] = []

    for slot in sorted_slots:
        team_a_id, team_b_id = _resolve_slot_teams(slot, winners, feeds_into)
        if team_a_id is None or team_b_id is None:
            continue

        market_spread, market_total = lookup_market_lines(slot, team_a_id, team_b_id)
        prediction, sim_prediction = create_matchup_prediction(
            slot=slot,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            ratings=ratings,
            seed_map=seed_map,
            tournament_predictor=tournament_predictor,
            team_names=team_names,
            market_spread_team_a=market_spread,
            market_total=market_total,
        )
        winner_id = (
            team_a_id
            if sim_prediction.team_a_win_prob >= sim_prediction.team_b_win_prob
            else team_b_id
        )
        winners[int(slot["slot_id"])] = winner_id
        game_cards.append(_game_card(slot, prediction, sim_prediction, winner_id, team_names))

    champion = game_cards[-1]["likely_winner_name"] if game_cards else None
    return {
        "champion": champion,
        "games": game_cards,
    }


def export_tournament_artifacts(results: dict) -> dict[str, str]:
    """Write tournament game cards, bracket path, and advancement probabilities to disk."""
    year = int(results["year"])
    output_dir = Path("predictions") / f"tournament_{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_bracket = results["best_bracket"]
    simulation = results["simulation"]
    team_names = results["team_names"]

    best_bracket_path = output_dir / "best_bracket.json"
    best_bracket_path.write_text(json.dumps(best_bracket, indent=2), encoding="utf-8")

    game_cards_path = output_dir / "game_cards.json"
    game_cards_path.write_text(json.dumps(best_bracket["games"], indent=2), encoding="utf-8")

    filled_bracket_csv = output_dir / "filled_bracket.csv"
    with filled_bracket_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slot_id",
                "round",
                "round_name",
                "game_in_round",
                "team_a_seed",
                "team_a_name",
                "team_a_score",
                "team_a_score_p10",
                "team_a_score_p90",
                "team_b_seed",
                "team_b_name",
                "team_b_score",
                "team_b_score_p10",
                "team_b_score_p90",
                "likely_winner_name",
                "likely_winner_prob",
                "projected_total",
                "projected_spread_team_a",
                "projected_possessions",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(best_bracket["games"])

    advancement_csv = output_dir / "advancement_probabilities.csv"
    with advancement_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "team_id",
                "team_name",
                "round_of_64",
                "round_of_32",
                "sweet_16",
                "elite_8",
                "final_four",
                "championship",
                "champion",
                "expected_wins",
            ],
        )
        writer.writeheader()
        for team_id, advancement in sorted(
            simulation.team_advancement.items(),
            key=lambda item: simulation.championship_probs.get(item[0], 0.0),
            reverse=True,
        ):
            writer.writerow(
                {
                    "team_id": team_id,
                    "team_name": team_names.get(team_id, str(team_id)),
                    "round_of_64": round(advancement.get(1, 0.0), 6),
                    "round_of_32": round(advancement.get(2, 0.0), 6),
                    "sweet_16": round(advancement.get(3, 0.0), 6),
                    "elite_8": round(advancement.get(4, 0.0), 6),
                    "final_four": round(advancement.get(5, 0.0), 6),
                    "championship": round(advancement.get(6, 0.0), 6),
                    "champion": round(simulation.championship_probs.get(team_id, 0.0), 6),
                    "expected_wins": round(simulation.team_expected_wins.get(team_id, 0.0), 6),
                }
            )

    md_path = output_dir / "best_bracket.md"
    lines = [
        f"# {year} Best Bracket",
        "",
        f"Projected champion: **{best_bracket['champion']}**",
        "",
    ]
    current_round = None
    for game in best_bracket["games"]:
        if game["round"] != current_round:
            current_round = game["round"]
            lines.extend([f"## {game['round_name']}", ""])
        lines.append(
            f"- {game['team_a_name']} ({game['team_a_seed']}) {game['team_a_score']:.1f} "
            f"[{game['team_a_score_p10']:.1f}-{game['team_a_score_p90']:.1f}] vs "
            f"{game['team_b_name']} ({game['team_b_seed']}) {game['team_b_score']:.1f} "
            f"[{game['team_b_score_p10']:.1f}-{game['team_b_score_p90']:.1f}] -> "
            f"{game['likely_winner_name']} ({game['likely_winner_prob']:.1%})"
        )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "best_bracket_json": str(best_bracket_path),
        "game_cards_json": str(game_cards_path),
        "filled_bracket_csv": str(filled_bracket_csv),
        "advancement_csv": str(advancement_csv),
        "best_bracket_md": str(md_path),
    }


def load_latest_ratings() -> dict[int, TeamRatings]:
    """Load the latest persisted team ratings from DuckDB."""
    with get_connection() as conn:
        latest = conn.execute("SELECT MAX(as_of_date) FROM team_strengths").fetchone()
        latest_date = latest[0] if latest else None
        if latest_date is None:
            return {}

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
            WHERE as_of_date = ?
            """,
            (latest_date,),
        ).fetchall()

    ratings: dict[int, TeamRatings] = {}
    for row in rows:
        ratings[int(row[0])] = TeamRatings(
            team_id=int(row[0]),
            adj_off=float(row[2]),
            adj_def=float(row[3]),
            adj_tempo=float(row[4]),
            adj_em=float(row[5]),
            adj_efg=float(row[6]),
            adj_tov=float(row[7]),
            adj_orb=float(row[8]),
            adj_ftr=float(row[9]),
            adj_efg_def=float(row[10]),
            adj_tov_def=float(row[11]),
            adj_drb=float(row[12]),
            adj_ftr_def=float(row[13]),
            games_played=int(row[14]),
            sos_off=float(row[15]),
            sos_def=float(row[16]),
            as_of_date=date.fromisoformat(row[1]) if isinstance(row[1], str) else row[1],
            home_off_delta=float(row[17] or 0.0),
            home_def_delta=float(row[18] or 0.0),
            away_off_delta=float(row[19] or 0.0),
            away_def_delta=float(row[20] or 0.0),
            home_games_played=int(row[21] or 0),
            away_games_played=int(row[22] or 0),
            off_std=float(row[23] or 0.0),
            def_std=float(row[24] or 0.0),
            tempo_std=float(row[25] or 0.0),
        )

    return ratings


def latest_ratings_metadata() -> tuple[Optional[date], int]:
    """Return latest rating snapshot date and team count."""
    with get_connection() as conn:
        latest = conn.execute("SELECT MAX(as_of_date) FROM team_strengths").fetchone()
        latest_date_raw = latest[0] if latest else None
        if latest_date_raw is None:
            return None, 0

        latest_date = (
            date.fromisoformat(latest_date_raw)
            if isinstance(latest_date_raw, str)
            else latest_date_raw
        )
        count_row = conn.execute(
            "SELECT COUNT(*) FROM team_strengths WHERE as_of_date = ?",
            (latest_date.isoformat(),),
        ).fetchone()
        return latest_date, int(count_row[0] if count_row else 0)


def load_team_names() -> dict[int, str]:
    """Load team ID to display-name mapping."""
    with get_connection() as conn:
        rows = conn.execute("SELECT team_id, name FROM teams").fetchall()
    return {int(row[0]): str(row[1]) for row in rows}


def build_tournament_ratings(as_of_date: date) -> dict[int, TeamRatings]:
    """Recalculate ratings from team_game_stats for the tournament date."""
    with get_connection() as conn:
        team_stats = conn.execute("SELECT * FROM team_game_stats").fetchdf()
        if team_stats.empty:
            raise ValueError("team_game_stats is empty - run boxscore ingestion first")

        conf_df = conn.execute(
            "SELECT team_id, conference_id, conference_name FROM team_conference_ids"
        ).fetchdf()

    hca_map = get_conference_hca_map(conf_df) if not conf_df.empty else None
    ratings = calculate_adjusted_ratings(
        team_stats=team_stats,
        as_of_date=as_of_date,
        use_recency_weights=True,
        conference_hca=hca_map,
    )
    save_ratings_to_db(ratings)
    logger.info("Tournament ratings rebuilt", teams=len(ratings), as_of_date=as_of_date.isoformat())
    return ratings


def ensure_team_stats_current(target_date: date) -> None:
    """Backfill team_game_stats through the day before the tournament if needed."""
    with get_connection() as conn:
        latest = conn.execute("SELECT MAX(CAST(game_date AS DATE)) FROM team_game_stats").fetchone()
        latest_date_raw = latest[0] if latest else None

    latest_date = (
        date.fromisoformat(latest_date_raw) if isinstance(latest_date_raw, str) else latest_date_raw
    )
    if latest_date is not None and latest_date >= target_date:
        return

    start_date = latest_date + timedelta(days=1) if latest_date else season_start_date(target_date)
    logger.info(
        "Refreshing team_game_stats for tournament model",
        start=start_date.isoformat(),
        end=target_date.isoformat(),
    )
    ingest_team_stats_for_date_range(
        start_date=start_date,
        end_date=target_date,
        season=infer_season_year(target_date),
    )


def build_seed_map(bracket_slots: list[dict]) -> dict[int, int]:
    """Map resolved tournament team IDs to seeds."""
    seed_map: dict[int, int] = {}
    for slot in bracket_slots:
        if slot.get("team_a_id") is not None and slot.get("seed_a") is not None:
            seed_map[int(slot["team_a_id"])] = int(slot["seed_a"])
        if slot.get("team_b_id") is not None and slot.get("seed_b") is not None:
            seed_map[int(slot["team_b_id"])] = int(slot["seed_b"])
    return seed_map


def ensure_tournament_team_ratings(
    bracket_slots: list[dict],
    tournament_start: date,
) -> dict[int, TeamRatings]:
    """Ensure every resolved tournament team has a rating."""
    needed_team_ids = {
        int(team_id)
        for slot in bracket_slots
        for team_id in (slot.get("team_a_id"), slot.get("team_b_id"))
        if team_id is not None
    }

    as_of_date = tournament_start - timedelta(days=1)
    ensure_team_stats_current(as_of_date)

    latest_date, latest_count = latest_ratings_metadata()
    with get_connection() as conn:
        team_master_row = conn.execute("SELECT COUNT(*) FROM teams").fetchone()
        team_master_count = int(team_master_row[0]) if team_master_row is not None else 0

    ratings = load_latest_ratings()
    missing_ids = sorted(team_id for team_id in needed_team_ids if team_id not in ratings)
    needs_rebuild = (
        latest_date != as_of_date or latest_count > int(team_master_count) + 5 or bool(missing_ids)
    )

    if needs_rebuild:
        logger.info(
            "Rebuilding tournament ratings snapshot",
            latest_date=latest_date.isoformat() if latest_date else None,
            latest_count=latest_count,
            team_master_count=int(team_master_count),
            missing_teams=len(missing_ids),
            target_date=as_of_date.isoformat(),
        )
        ratings = build_tournament_ratings(as_of_date)
        missing_ids = sorted(team_id for team_id in needed_team_ids if team_id not in ratings)

    if missing_ids:
        raise ValueError(f"Missing ratings for tournament teams: {missing_ids}")

    return ratings


def create_matchup_prediction(
    slot: dict,
    team_a_id: int,
    team_b_id: int,
    ratings: dict[int, TeamRatings],
    seed_map: dict[int, int],
    tournament_predictor: TournamentPredictor,
    team_names: dict[int, str],
    market_spread_team_a: Optional[float] = None,
    market_total: Optional[float] = None,
) -> tuple[TournamentPrediction, GamePrediction]:
    """Price a single tournament matchup in current slot orientation."""
    seed_a = seed_map.get(team_a_id, int(slot.get("seed_a") or 16))
    seed_b = seed_map.get(team_b_id, int(slot.get("seed_b") or 16))

    if seed_a <= seed_b:
        higher_seed = seed_a
        lower_seed = seed_b
        higher_id = team_a_id
        lower_id = team_b_id
        team_a_is_higher = True
    else:
        higher_seed = seed_b
        lower_seed = seed_a
        higher_id = team_b_id
        lower_id = team_a_id
        team_a_is_higher = False

    market_spread_higher_seed = None
    if market_spread_team_a is not None:
        market_spread_higher_seed = (
            market_spread_team_a if team_a_is_higher else -market_spread_team_a
        )

    prediction = tournament_predictor.predict_game(
        higher_seed=higher_seed,
        lower_seed=lower_seed,
        higher_seed_ratings=ratings[higher_id],
        lower_seed_ratings=ratings[lower_id],
        slot_id=int(slot["slot_id"]),
        year=int(slot["year"]),
        game_round=int(slot["round"]),
        higher_seed_team_id=higher_id,
        lower_seed_team_id=lower_id,
        higher_seed_name=team_names.get(higher_id, f"Team {higher_id}"),
        lower_seed_name=team_names.get(lower_id, f"Team {lower_id}"),
        market_spread=market_spread_higher_seed,
        market_total=market_total,
    )

    team_a_win_prob = (
        prediction.higher_seed_win_prob
        if team_a_is_higher
        else 1.0 - prediction.higher_seed_win_prob
    )
    game_prediction = GamePrediction(
        slot_id=int(slot["slot_id"]),
        round=int(slot["round"]),
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        team_a_seed=seed_a,
        team_b_seed=seed_b,
        team_a_name=team_names.get(team_a_id, f"Team {team_a_id}"),
        team_b_name=team_names.get(team_b_id, f"Team {team_b_id}"),
        team_a_win_prob=team_a_win_prob,
        team_b_win_prob=1.0 - team_a_win_prob,
        proj_spread=prediction.proj_spread if team_a_is_higher else -prediction.proj_spread,
        proj_total=prediction.proj_total,
        upset_prob=prediction.upset_prob,
    )

    return prediction, game_prediction


def run_tournament_predictions(
    year: int = 2026,
    num_simulations: int = 100_000,
    use_gpu: bool = True,
    use_seed_priors: bool = True,
    use_template: bool = False,
) -> dict:
    """Run the full live tournament pipeline."""
    logger.info("Starting tournament prediction pipeline", year=year)
    init_database()

    load_2026_bracket(year=year, use_template=use_template)
    bracket_slots = load_bracket_from_db(year)
    if not bracket_slots:
        raise RuntimeError("No tournament bracket slots found after load")

    tournament_start = min(
        date.fromisoformat(slot["game_date"])
        if isinstance(slot["game_date"], str)
        else slot["game_date"]
        for slot in bracket_slots
        if slot.get("game_date")
    )
    sync_tournament_schedule(tournament_start, tournament_start + timedelta(days=3))
    sync_tournament_pickcenter_lines(tournament_start, tournament_start + timedelta(days=3))

    ratings = ensure_tournament_team_ratings(bracket_slots, tournament_start)
    team_names = load_team_names()
    seed_map = build_seed_map(bracket_slots)

    downstream_sources = {
        int(slot["next_slot_id"]) for slot in bracket_slots if slot.get("next_slot_id") is not None
    }
    unresolved_current = []
    for slot in bracket_slots:
        round_value = int(slot["round"])
        missing_seeded_side = (
            slot.get("seed_a") is not None and slot.get("team_a_id") is None
        ) or (slot.get("seed_b") is not None and slot.get("team_b_id") is None)
        if not missing_seeded_side:
            continue
        if round_value == 0:
            unresolved_current.append(slot["slot_id"])
        elif round_value == 1 and int(slot["slot_id"]) not in downstream_sources:
            unresolved_current.append(slot["slot_id"])

    if unresolved_current:
        raise RuntimeError(f"Unresolved current tournament teams in slots: {unresolved_current}")

    base_predictor = create_enhanced_predictor()
    tournament_predictor = TournamentPredictor(
        base_predictor=base_predictor,
        use_seed_priors=use_seed_priors,
        seed_prior_weight=0.15,
    )

    known_predictions: list[TournamentPrediction] = []
    game_predictions: dict[int, GamePrediction] = {}
    games_with_market = 0

    with get_connection() as conn:
        conn.execute("DELETE FROM tournament_predictions WHERE year = ?", (year,))
        for slot in bracket_slots:
            team_a_id = slot.get("team_a_id")
            team_b_id = slot.get("team_b_id")
            if team_a_id is None or team_b_id is None:
                continue

            market_spread, market_total = lookup_market_lines(
                slot=slot,
                team_a_id=int(team_a_id),
                team_b_id=int(team_b_id),
            )
            if market_spread is not None or market_total is not None:
                games_with_market += 1

            prediction, sim_prediction = create_matchup_prediction(
                slot=slot,
                team_a_id=int(team_a_id),
                team_b_id=int(team_b_id),
                ratings=ratings,
                seed_map=seed_map,
                tournament_predictor=tournament_predictor,
                team_names=team_names,
                market_spread_team_a=market_spread,
                market_total=market_total,
            )
            save_tournament_prediction(prediction, conn)
            known_predictions.append(prediction)
            game_predictions[int(slot["slot_id"])] = sim_prediction

    logger.info(
        "Known tournament games priced",
        count=len(known_predictions),
        games_with_market=games_with_market,
    )
    if games_with_market == 0:
        logger.warning(
            "No market lines found for priced tournament games; running model-only tournament pricing"
        )

    def matchup_predictor(team_a_id: int, team_b_id: int, slot_info: dict) -> GamePrediction:
        market_spread, market_total = lookup_market_lines(
            slot=slot_info,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
        )
        _, game_prediction = create_matchup_prediction(
            slot=slot_info,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            ratings=ratings,
            seed_map=seed_map,
            tournament_predictor=tournament_predictor,
            team_names=team_names,
            market_spread_team_a=market_spread,
            market_total=market_total,
        )
        return game_prediction

    simulator = BracketSimulator(num_simulations=num_simulations, use_gpu=use_gpu)
    sim_result = simulator.simulate(
        bracket_slots=bracket_slots,
        game_predictions=game_predictions,
        year=year,
        matchup_predictor=matchup_predictor,
        team_names=team_names,
    )
    save_simulation_results(sim_result, team_names)

    best_bracket = build_best_bracket_report(
        bracket_slots=bracket_slots,
        ratings=ratings,
        seed_map=seed_map,
        tournament_predictor=tournament_predictor,
        team_names=team_names,
    )

    results = {
        "year": year,
        "num_games_predicted": len(known_predictions),
        "num_simulations": num_simulations,
        "gpu_used": sim_result.gpu_used,
        "runtime_ms": sim_result.total_runtime_ms,
        "predictions": known_predictions,
        "simulation": sim_result,
        "team_names": team_names,
        "best_bracket": best_bracket,
    }
    results["artifacts"] = export_tournament_artifacts(results)
    return results


def print_predictions(results: dict) -> None:
    """Print a concise tournament summary."""
    predictions = results["predictions"]
    sim = results["simulation"]
    best_bracket = results["best_bracket"]

    print(f"\n{'=' * 80}")
    print(f"  {results['year']} MARCH MADNESS TOURNAMENT PREDICTIONS")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"  GPU Acceleration: {'Yes' if sim.gpu_used else 'No'} ({results['num_simulations']:,} simulations)"
    )
    print(f"{'=' * 80}\n")

    current_round = -1
    for prediction in sorted(predictions, key=lambda pred: (pred.game_round, pred.slot_id)):
        if prediction.game_round != current_round:
            current_round = prediction.game_round
            round_name = ROUND_NAMES.get(current_round, f"Round {current_round}")
            print(f"\n{'-' * 60}")
            print(f"  {round_name.upper()}")
            print(f"{'-' * 60}")

        print(
            f"\n  #{prediction.higher_seed} {prediction.higher_seed_name} vs "
            f"#{prediction.lower_seed} {prediction.lower_seed_name}"
        )
        print(
            f"  Projected: {prediction.higher_seed_name} {prediction.proj_higher_score:.1f} - "
            f"{prediction.lower_seed_name} {prediction.proj_lower_score:.1f}"
        )
        print(
            f"  Win Prob: {prediction.higher_seed_win_prob:.1%} | "
            f"Total: {prediction.proj_total:.1f}"
        )
        print(
            f"  Score Band: {prediction.higher_seed_name} {prediction.higher_score_p10:.1f}-{prediction.higher_score_p90:.1f} | "
            f"{prediction.lower_seed_name} {prediction.lower_score_p10:.1f}-{prediction.lower_score_p90:.1f}"
        )

    if best_bracket.get("champion"):
        print(f"\nProjected champion in best bracket: {best_bracket['champion']}")

    print(f"\n{'=' * 80}")
    print("  TOP 15 CHAMPIONSHIP PROBABILITIES")
    print(f"{'=' * 80}")
    for rank, (_team_id, team_name, probability) in enumerate(sim.top_champions[:15], start=1):
        print(f"  {rank:2d}. {team_name:<28} {probability:>7.1%}")
    print(f"{'=' * 80}\n")

    artifacts = results.get("artifacts", {})
    if artifacts:
        print("Artifacts:")
        print(f"  Best bracket: {artifacts['best_bracket_json']}")
        print(f"  Game cards: {artifacts['game_cards_json']}")
        print(f"  Filled bracket CSV: {artifacts['filled_bracket_csv']}")
        print(f"  Advancement probs: {artifacts['advancement_csv']}")
        print(f"  Markdown bracket: {artifacts['best_bracket_md']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tournament prediction pipeline")
    parser.add_argument(
        "--sims", type=int, default=100_000, help="Number of Monte Carlo simulations"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument(
        "--no-seed-priors",
        action="store_true",
        help="Disable historical seed priors in tournament pricing",
    )
    parser.add_argument(
        "--use-template",
        action="store_true",
        help="Use the local template bracket instead of live NCAA data",
    )
    args = parser.parse_args()

    results = run_tournament_predictions(
        year=2026,
        num_simulations=args.sims,
        use_gpu=not args.no_gpu,
        use_seed_priors=not args.no_seed_priors,
        use_template=args.use_template,
    )
    print_predictions(results)


if __name__ == "__main__":
    main()
