"""
Scheduled job worker for NCAAB predictions.

Uses APScheduler to run:
- Daily schedule ingestion (6am ET)
- Odds snapshots (every 4 hours)
- Prediction generation (6:30am ET)
- Closing line capture (after games start)

Run with: python -m apps.worker.scheduler
"""

from datetime import date, datetime, timedelta
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import structlog

from packages.common.config import get_settings
from packages.common.database import get_connection, init_database
from packages.common.schemas import FeatureRow
from packages.eval.metrics import calculate_clv
from packages.eval.guardrails import apply_min_games_guardrail
from packages.eval.quality_report import generate_quality_report, write_quality_report
from packages.ingest.espn_api import (
    fetch_schedule,
    fetch_team_conferences,
    fetch_teams,
    save_games_to_db,
    save_team_conferences_to_db,
    save_teams_to_db,
)
from packages.ingest.odds_api import fetch_ncaab_odds, save_odds_snapshot
from packages.features.conference_hca import get_conference_hca_map
from packages.common.season import infer_season_year, season_start_date
from packages.features.kenpom_ratings import TeamRatings, calculate_adjusted_ratings, save_ratings_to_db
from packages.ingest.espn_enhanced import ingest_team_stats_for_date_range
from packages.models.enhanced_predictor import EnhancedPredictor, create_enhanced_predictor

logger = structlog.get_logger()

# Initialize scheduler
scheduler = BlockingScheduler(timezone="America/New_York")


def job_ingest_schedule():
    """
    Ingest today's game schedule from ESPN.

    Runs daily at 6:00 AM ET.
    """
    logger.info("Starting schedule ingestion job")

    try:
        # Fetch today and tomorrow's games
        today = date.today()
        tomorrow = today + timedelta(days=1)

        for target_date in [today, tomorrow]:
            games = fetch_schedule(target_date)
            if games:
                saved = save_games_to_db(games, skip_fk_errors=True)
                logger.info(
                    "Schedule ingested",
                    date=target_date.isoformat(),
                    fetched=len(games),
                    saved=saved,
                )
            team_confs = fetch_team_conferences(target_date)
            if team_confs:
                saved_confs = save_team_conferences_to_db(team_confs)
                logger.info(
                    "Team conferences ingested",
                    date=target_date.isoformat(),
                    saved=saved_confs,
                )

    except Exception as e:
        logger.error("Schedule ingestion failed", error=str(e))


def job_ingest_results(days_back: int = 3):
    """
    Refresh recent game results (scores + status).

    Runs daily to backfill finals for the last few days.
    """
    logger.info("Starting results refresh job", days_back=days_back)

    try:
        today = date.today()
        for offset in range(1, days_back + 1):
            target_date = today - timedelta(days=offset)
            games = fetch_schedule(target_date)
            if games:
                saved = save_games_to_db(games, skip_fk_errors=True)
                logger.info(
                    "Results refreshed",
                    date=target_date.isoformat(),
                    fetched=len(games),
                    saved=saved,
                )
    except Exception as e:
        logger.error("Results refresh failed", error=str(e))


def job_ingest_boxscores(days_back: int = 7):
    """
    Refresh team-game stats for recent completed games.
    """
    logger.info("Starting boxscore ingestion job", days_back=days_back)

    try:
        today = date.today()
        end_date = today - timedelta(days=1)
        if end_date < season_start_date(today):
            logger.info("No completed games yet for current season")
            return

        season_start = season_start_date(today)
        start_date = end_date - timedelta(days=days_back)

        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT MIN(CAST(game_date AS DATE))
                FROM team_game_stats
                WHERE CAST(game_date AS DATE) >= ?
                """,
                (season_start.isoformat(),),
            ).fetchone()

        min_date = row[0] if row else None
        if isinstance(min_date, str):
            min_date = date.fromisoformat(min_date)

        if min_date is None or min_date > season_start:
            logger.info(
                "Backfilling team stats from season start",
                season_start=season_start.isoformat(),
            )
            start_date = season_start

        if end_date < start_date:
            logger.info("No date range to ingest for boxscores")
            return

        season = infer_season_year(end_date)
        stats_df = ingest_team_stats_for_date_range(start_date, end_date, season=season)
        logger.info("Boxscore ingestion complete", rows=len(stats_df))
    except Exception as e:
        logger.error("Boxscore ingestion failed", error=str(e))


def job_update_ratings():
    """
    Recalculate adjusted ratings from team_game_stats.
    """
    logger.info("Starting ratings update job")

    try:
        today = date.today()

        with get_connection() as conn:
            team_stats = conn.execute("SELECT * FROM team_game_stats").fetchdf()
            if team_stats.empty:
                logger.warning("No team_game_stats found - run boxscore ingestion first")
                return

            team_confs = conn.execute(
                """
                SELECT team_id, conference_id, conference_name
                FROM team_conference_ids
                """
            ).fetchdf()

        hca_map = get_conference_hca_map(team_confs) if not team_confs.empty else None
        ratings = calculate_adjusted_ratings(
            team_stats=team_stats,
            as_of_date=today,
            use_recency_weights=True,
            conference_hca=hca_map,
        )
        save_ratings_to_db(ratings)

        logger.info("Ratings update complete", teams=len(ratings), as_of_date=today.isoformat())
    except Exception as e:
        logger.error("Ratings update failed", error=str(e))


def job_ingest_teams():
    """
    Ingest D1 teams from ESPN.

    Runs weekly on Sunday at 5:00 AM ET.
    """
    logger.info("Starting teams ingestion job")

    try:
        teams = fetch_teams()
        saved = save_teams_to_db(teams)
        logger.info("Teams ingested", count=saved)

    except Exception as e:
        logger.error("Teams ingestion failed", error=str(e))


def job_ingest_odds():
    """
    Fetch and save odds snapshots.

    Runs every 4 hours.
    """
    settings = get_settings()
    if not settings.odds_api_key:
        logger.warning("No ODDS_API_KEY configured, skipping odds ingestion")
        return

    logger.info("Starting odds ingestion job")

    try:
        snapshots = fetch_ncaab_odds()
        if snapshots:
            save_odds_snapshot(snapshots)
            logger.info("Odds snapshot saved", count=len(snapshots))
        else:
            logger.info("No odds data available")

    except Exception as e:
        logger.error("Odds ingestion failed", error=str(e))


def job_generate_predictions():
    """
    Generate predictions for today's games.

    Runs daily at 6:30 AM ET.
    """
    logger.info("Starting prediction generation job")

    try:
        predictor = create_enhanced_predictor()
        today = date.today()

        with get_connection() as conn:
            # Get today's games that don't have predictions yet
            rows = conn.execute(
                """
                SELECT
                    g.game_id, g.home_team_id, g.away_team_id,
                    g.neutral_site, g.home_team_name, g.away_team_name,
                    bs.spread_line_home, bs.total_line,
                    ls.spread_home, ls.spread_away, ls.total_line
                FROM games g
                LEFT JOIN (
                    SELECT
                        game_id,
                        spread_line_home,
                        total_line,
                        ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY snapshot_timestamp DESC) AS rn
                    FROM betting_splits
                ) bs ON g.game_id = bs.game_id AND bs.rn = 1
                LEFT JOIN (
                    SELECT
                        game_id,
                        AVG(spread_home) AS spread_home,
                        AVG(spread_away) AS spread_away,
                        AVG(total_line) AS total_line
                    FROM line_snapshots
                    GROUP BY game_id, snapshot_timestamp
                    QUALIFY snapshot_timestamp = MAX(snapshot_timestamp)
                        OVER (PARTITION BY game_id)
                ) ls ON g.game_id = ls.game_id
                WHERE g.game_date = ?
                    AND g.status = 'scheduled'
                """,
                (today.isoformat(),),
            ).fetchall()

            if not rows:
                logger.info("No games to predict")
                return

            # Conference mapping (team_id -> conference_id)
            conf_rows = conn.execute(
                "SELECT team_id, conference_id FROM team_conference_ids"
            ).fetchall()
            team_confs = {row[0]: row[1] for row in conf_rows}

            # Rest days lookup (team_id -> last game date)
            last_game_rows = conn.execute(
                """
                SELECT team_id, MAX(game_date) AS last_game_date
                FROM (
                    SELECT home_team_id AS team_id, game_date
                    FROM games
                    WHERE game_date < ? AND status = 'final'
                    UNION ALL
                    SELECT away_team_id AS team_id, game_date
                    FROM games
                    WHERE game_date < ? AND status = 'final'
                )
                GROUP BY team_id
                """,
                (today, today),
            ).fetchall()
            last_game_dates = {row[0]: row[1] for row in last_game_rows}

            # Get team strengths
            team_strengths = _get_team_strengths(conn)

            predictions_made = 0
            for row in rows:
                game_id = row[0]
                home_team_id = row[1]
                away_team_id = row[2]
                is_neutral = bool(row[3])
                home_name = row[4]
                away_name = row[5]
                splits_spread_home = row[6]
                splits_total = row[7]
                spread_home = row[8]
                spread_away = row[9]
                line_total = row[10]
                home_conf_id = team_confs.get(home_team_id)

                home_last = last_game_dates.get(home_team_id)
                away_last = last_game_dates.get(away_team_id)
                home_rest_days = (today - home_last).days if home_last else 7
                away_rest_days = (today - away_last).days if away_last else 7

                market_spread = None
                market_total = None
                if splits_spread_home is not None:
                    # DraftKings splits stored in model convention (positive = home favored)
                    market_spread = splits_spread_home
                    market_total = splits_total
                else:
                    if spread_home is not None:
                        market_spread = -spread_home
                    elif spread_away is not None:
                        market_spread = spread_away
                    market_total = line_total

                # Get team stats - SKIP if either team has no real ratings
                # Per CLAUDE.md: "Never use fallbacks, mock data, or placeholder values"
                home_stats = team_strengths.get(home_team_id)
                away_stats = team_strengths.get(away_team_id)

                if home_stats is None or away_stats is None:
                    missing = []
                    if home_stats is None:
                        missing.append(f"home:{home_name}({home_team_id})")
                    if away_stats is None:
                        missing.append(f"away:{away_name}({away_team_id})")
                    logger.warning(
                        "Skipping game - missing team ratings (no fallbacks allowed)",
                        game_id=game_id,
                        missing_teams=missing,
                    )
                    continue

                # Generate prediction
                pred = predictor.predict_game(
                    home_ratings=home_stats,
                    away_ratings=away_stats,
                    game_id=game_id,
                    is_neutral=is_neutral,
                    home_rest_days=home_rest_days,
                    away_rest_days=away_rest_days,
                    home_conference_id=home_conf_id,
                    market_spread=market_spread,
                    market_total=market_total,
                )

                # Convert to prediction row and save
                pred_row = predictor.to_prediction_row(
                    prediction=pred,
                    market_spread=market_spread,
                    market_total=market_total,
                )
                pred_row = apply_min_games_guardrail(
                    pred_row,
                    home_games_played=home_stats.games_played,
                    away_games_played=away_stats.games_played,
                )

                _save_prediction(conn, pred_row)
                predictions_made += 1

                logger.debug(
                    "Prediction generated",
                    game_id=game_id,
                    matchup=f"{away_name} @ {home_name}",
                    spread=pred.spread,
                    total=pred.total,
                )

            logger.info("Predictions generated", count=predictions_made)

    except Exception as e:
        logger.error("Prediction generation failed", error=str(e))


def job_capture_closing_lines():
    """
    Generate CLV reports for completed games.

    Uses latest predictions and line snapshots.
    """
    logger.info("Generating CLV reports for completed games")

    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                WITH latest_predictions AS (
                    SELECT *
                    FROM (
                        SELECT
                            *,
                            ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY prediction_timestamp DESC) AS rn
                        FROM predictions
                    )
                    WHERE rn = 1
                )
                SELECT
                    g.game_id, g.game_date, g.game_datetime,
                    g.home_score, g.away_score,
                    p.prediction_timestamp,
                    p.proj_spread, p.proj_total
                FROM games g
                JOIN latest_predictions p ON g.game_id = p.game_id
                LEFT JOIN clv_reports c
                    ON g.game_id = c.game_id AND p.prediction_timestamp = c.prediction_timestamp
                WHERE g.status = 'final'
                  AND g.home_score IS NOT NULL
                  AND g.away_score IS NOT NULL
                  AND c.game_id IS NULL
                """
            ).fetchall()

            if not rows:
                logger.info("No new games for CLV reports")
                return

            def _market_spread_from_snapshot(spread_home, spread_away):
                if spread_home is not None:
                    return -float(spread_home)
                if spread_away is not None:
                    return float(spread_away)
                return None

            def _fetch_line_snapshot(game_id, as_of_ts):
                row = conn.execute(
                    """
                    SELECT spread_home, spread_away, total_line
                    FROM line_snapshots
                    WHERE game_id = ?
                      AND snapshot_timestamp <= ?
                    ORDER BY snapshot_timestamp DESC
                    LIMIT 1
                    """,
                    (game_id, as_of_ts),
                ).fetchone()
                if not row:
                    return None, None
                return _market_spread_from_snapshot(row[0], row[1]), row[2]

            def _fetch_splits_snapshot(game_id, as_of_ts):
                row = conn.execute(
                    """
                    SELECT spread_line_home, total_line
                    FROM betting_splits
                    WHERE game_id = ?
                      AND snapshot_timestamp <= ?
                    ORDER BY snapshot_timestamp DESC
                    LIMIT 1
                    """,
                    (game_id, as_of_ts),
                ).fetchone()
                if not row:
                    return None, None
                return row[0], row[1]

            def _resolve_market_line(game_id, as_of_ts):
                spread, total = _fetch_line_snapshot(game_id, as_of_ts)
                if spread is None and total is None:
                    spread, total = _fetch_splits_snapshot(game_id, as_of_ts)
                return spread, total

            inserts = []

            for row in rows:
                game_id = row[0]
                game_date = row[1]
                game_datetime = row[2]
                home_score = row[3]
                away_score = row[4]
                pred_ts = row[5]
                pred_spread = row[6]
                pred_total = row[7]

                if isinstance(pred_ts, str):
                    pred_ts = datetime.fromisoformat(pred_ts.replace("Z", "+00:00"))
                if isinstance(game_datetime, str):
                    game_datetime = datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))

                close_ts = game_datetime or datetime.combine(
                    game_date if hasattr(game_date, "year") else date.fromisoformat(game_date),
                    datetime.max.time(),
                )

                market_spread, market_total = _resolve_market_line(game_id, pred_ts)
                closing_spread, closing_total = _resolve_market_line(game_id, close_ts)

                if market_spread is None or market_total is None:
                    continue
                if closing_spread is None or closing_total is None:
                    continue

                spread_side = "home" if (pred_spread - market_spread) > 0 else "away"
                total_side = "over" if (pred_total - market_total) > 0 else "under"

                spread_clv = calculate_clv(market_spread, closing_spread, spread_side)
                total_clv = calculate_clv(market_total, closing_total, total_side)

                actual_spread = home_score - away_score
                actual_total = home_score + away_score

                spread_won = None
                if actual_spread != market_spread:
                    spread_won = actual_spread > market_spread if spread_side == "home" else actual_spread < market_spread

                total_won = None
                if actual_total != market_total:
                    total_won = actual_total > market_total if total_side == "over" else actual_total < market_total

                inserts.append(
                    (
                        game_id,
                        pred_ts.isoformat(),
                        pred_ts.isoformat(),
                        pred_spread,
                        pred_total,
                        market_spread,
                        market_total,
                        closing_spread,
                        closing_total,
                        spread_clv,
                        total_clv,
                        actual_spread,
                        actual_total,
                        spread_side,
                        spread_won,
                        total_side,
                        total_won,
                    )
                )

            if inserts:
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
                    inserts,
                )

            logger.info("CLV reports generated", count=len(inserts))

    except Exception as e:
        logger.error("CLV report generation failed", error=str(e))


def job_quality_report():
    """
    Generate a daily data-quality report for today's slate.
    """
    logger.info("Generating daily quality report")

    try:
        report = generate_quality_report(date.today())
        path = write_quality_report(report)
        logger.info("Quality report generated", path=str(path))
    except Exception as e:
        logger.error("Quality report generation failed", error=str(e))


def _get_team_strengths(conn) -> dict[int, TeamRatings]:
    """Get latest team strength ratings from database."""
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
        WHERE (team_id, as_of_date) IN (
            SELECT team_id, MAX(as_of_date)
            FROM team_strengths
            GROUP BY team_id
        )
        """
    ).fetchall()

    strengths: dict[int, TeamRatings] = {}
    for row in rows:
        strengths[row[0]] = TeamRatings(
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

    return strengths


def _save_prediction(conn, pred_row):
    """Save prediction to database."""
    conn.execute(
        """
        INSERT OR REPLACE INTO predictions (
            game_id, prediction_timestamp, model_version,
            proj_home_score, proj_away_score,
            proj_spread, proj_total, proj_possessions,
            home_win_prob,
            spread_ci_50_lower, spread_ci_50_upper,
            spread_ci_80_lower, spread_ci_80_upper,
            spread_ci_95_lower, spread_ci_95_upper,
            total_ci_50_lower, total_ci_50_upper,
            total_ci_80_lower, total_ci_80_upper,
            total_ci_95_lower, total_ci_95_upper,
            market_spread, edge_vs_market_spread,
            market_total, edge_vs_market_total,
            recommended_side, recommended_units, confidence_rating
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pred_row.game_id,
            pred_row.prediction_timestamp.isoformat(),
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
            pred_row.market_spread,
            pred_row.edge_vs_market_spread,
            pred_row.market_total,
            pred_row.edge_vs_market_total,
            pred_row.recommended_side,
            pred_row.recommended_units,
            pred_row.confidence_rating,
        ),
    )


def run_manual_job(job_name: str):
    """Run a job manually for testing."""
    jobs = {
        "schedule": job_ingest_schedule,
        "results": job_ingest_results,
        "boxscores": job_ingest_boxscores,
        "ratings": job_update_ratings,
        "teams": job_ingest_teams,
        "odds": job_ingest_odds,
        "predictions": job_generate_predictions,
        "closing": job_capture_closing_lines,
        "report": job_quality_report,
    }

    if job_name not in jobs:
        logger.error("Unknown job", name=job_name, available=list(jobs.keys()))
        return

    logger.info("Running manual job", name=job_name)
    jobs[job_name]()


def setup_scheduler():
    """Configure all scheduled jobs."""
    # Daily schedule ingestion at 6:00 AM ET
    scheduler.add_job(
        job_ingest_schedule,
        CronTrigger(hour=6, minute=0),
        id="ingest_schedule",
        name="Ingest daily schedule",
        replace_existing=True,
    )

    # Refresh results for recent days at 6:05 AM ET
    scheduler.add_job(
        job_ingest_results,
        CronTrigger(hour=6, minute=5),
        id="ingest_results",
        name="Refresh recent results",
        replace_existing=True,
    )

    # Refresh boxscores at 6:10 AM ET
    scheduler.add_job(
        job_ingest_boxscores,
        CronTrigger(hour=6, minute=10),
        id="ingest_boxscores",
        name="Refresh recent boxscores",
        replace_existing=True,
    )

    # Update ratings at 6:20 AM ET
    scheduler.add_job(
        job_update_ratings,
        CronTrigger(hour=6, minute=20),
        id="update_ratings",
        name="Update team ratings",
        replace_existing=True,
    )

    # Weekly teams refresh on Sunday at 5:00 AM ET
    scheduler.add_job(
        job_ingest_teams,
        CronTrigger(day_of_week="sun", hour=5, minute=0),
        id="ingest_teams",
        name="Refresh teams",
        replace_existing=True,
    )

    # Odds snapshots every 4 hours
    scheduler.add_job(
        job_ingest_odds,
        IntervalTrigger(hours=4),
        id="ingest_odds",
        name="Fetch odds snapshot",
        replace_existing=True,
    )

    # Daily predictions at 6:30 AM ET
    scheduler.add_job(
        job_generate_predictions,
        CronTrigger(hour=6, minute=30),
        id="generate_predictions",
        name="Generate daily predictions",
        replace_existing=True,
    )

    # Closing line capture every 30 minutes
    scheduler.add_job(
        job_capture_closing_lines,
        IntervalTrigger(minutes=30),
        id="generate_clv",
        name="Generate CLV reports",
        replace_existing=True,
    )

    # Daily quality report at 7:00 AM ET
    scheduler.add_job(
        job_quality_report,
        CronTrigger(hour=7, minute=0),
        id="quality_report",
        name="Generate quality report",
        replace_existing=True,
    )

    logger.info(
        "Scheduler configured",
        jobs=[job.id for job in scheduler.get_jobs()],
    )


def main():
    """Main entry point for scheduler."""
    import sys

    logger.info("Starting NCAAB scheduler worker")

    # Initialize database
    init_database()

    # Check for manual job run
    if len(sys.argv) > 1:
        job_name = sys.argv[1]
        run_manual_job(job_name)
        return

    # Setup and start scheduler
    setup_scheduler()

    try:
        logger.info("Starting scheduler")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
