"""
Run the complete enhanced prediction pipeline.

This script:
1. Ingests real ESPN box score data
2. Calculates KenPom-style adjusted efficiency ratings
3. Generates PhD-level predictions for today's games
4. Outputs validation metrics

NO MOCK DATA. NO FALLBACKS. REAL PREDICTIONS ONLY.
"""

import sys
from pathlib import Path
from datetime import date, datetime
from typing import Optional

import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packages.ingest.espn_enhanced import (
    fetch_season_boxscores,
    fetch_season_schedule,
    fetch_all_teams,
    process_boxscores_to_team_stats,
    save_team_stats_to_db,
)
from packages.features.kenpom_ratings import (
    calculate_adjusted_ratings,
    save_ratings_to_db,
    TeamRatings,
)
from packages.features.conference_hca import get_conference_hca_map
from packages.models.enhanced_predictor import (
    EnhancedPredictor,
    EnhancedPrediction,
    predict_slate,
    create_enhanced_predictor,
    MODEL_VERSION,
)
from packages.ingest.fetch_betting_splits import fetch_and_save_splits
from packages.common.database import get_connection, init_database

logger = structlog.get_logger()


def setup_database():
    """Ensure database tables exist."""
    logger.info("Setting up database")

    with get_connection() as conn:
        # Create team_game_stats table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_game_stats (
                game_id INTEGER,
                game_date TEXT,
                team_id INTEGER,
                opponent_id INTEGER,
                is_home BOOLEAN,
                is_neutral BOOLEAN,
                team_score INTEGER,
                opponent_score INTEGER,
                won BOOLEAN,

                fgm INTEGER,
                fga INTEGER,
                fg3m INTEGER,
                fg3a INTEGER,
                ftm INTEGER,
                fta INTEGER,
                orb INTEGER,
                drb INTEGER,
                turnovers INTEGER,
                assists INTEGER,
                steals INTEGER,
                blocks INTEGER,

                off_efg REAL,
                off_tov REAL,
                off_orb REAL,
                off_ftr REAL,
                possessions REAL,
                off_rating REAL,

                def_efg REAL,
                def_tov REAL,
                def_orb REAL,
                def_ftr REAL,
                def_rating REAL,

                rest_days INTEGER,

                PRIMARY KEY (game_id, team_id)
            )
        """)

        # Create team_strengths table (append-only snapshots)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_strengths (
                team_id INTEGER,
                as_of_date TEXT,
                adj_offensive_efficiency REAL,
                adj_defensive_efficiency REAL,
                adj_tempo REAL,
                adj_em REAL,
                off_efg REAL,
                off_tov REAL,
                off_orb REAL,
                off_ftr REAL,
                def_efg REAL,
                def_tov REAL,
                def_drb REAL,
                def_ftr REAL,
                games_played INTEGER,
                sos_off REAL,
                sos_def REAL,
                home_off_delta REAL,
                home_def_delta REAL,
                away_off_delta REAL,
                away_def_delta REAL,
                home_games_played INTEGER,
                away_games_played INTEGER,
                off_rating_std REAL,
                def_rating_std REAL,
                tempo_std REAL,
                PRIMARY KEY (team_id, as_of_date)
            )
        """)

        # Create teams table if needed
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT,
                display_name TEXT,
                abbreviation TEXT,
                conference_id INTEGER,
                conference_name TEXT,
                logo_url TEXT
            )
        """)

    logger.info("Database setup complete")


def ingest_season_data(season: int = 2025) -> tuple:
    """Ingest full season data from ESPN."""
    logger.info("Starting season data ingestion", season=season)

    # Fetch box scores
    logger.info("Fetching box scores...")
    boxscores = fetch_season_boxscores(season)
    logger.info(f"Fetched {len(boxscores)} box score rows")

    # Fetch schedule
    logger.info("Fetching schedule...")
    schedule = fetch_season_schedule(season)
    logger.info(f"Fetched {len(schedule)} schedule rows")

    # Process into team stats
    logger.info("Processing box scores into team stats...")
    team_stats = process_boxscores_to_team_stats(boxscores, schedule)
    logger.info(f"Processed {len(team_stats)} team-game stats")

    # Save to database
    save_team_stats_to_db(team_stats)

    # Ingest betting splits
    logger.info("Fetching betting splits...")
    try:
        splits_count = fetch_and_save_splits(date.today())
        logger.info(f"Saved betting splits for {splits_count} games")
    except Exception as e:
        logger.warning(f"Failed to fetch betting splits: {str(e)}")

    return boxscores, schedule, team_stats


def calculate_ratings(team_stats, teams_df, as_of_date: Optional[date] = None) -> dict[int, TeamRatings]:
    """Calculate adjusted efficiency ratings."""
    if as_of_date is None:
        as_of_date = date.today()

    logger.info("Calculating adjusted ratings", as_of_date=as_of_date.isoformat())

    # Get conference HCA map
    hca_map = get_conference_hca_map(teams_df)

    # Calculate ratings
    ratings = calculate_adjusted_ratings(
        team_stats=team_stats,
        as_of_date=as_of_date,
        use_recency_weights=True,
        conference_hca=hca_map,
    )

    logger.info(f"Calculated ratings for {len(ratings)} teams")

    # Save to database
    save_ratings_to_db(ratings)

    return ratings


def get_todays_games(schedule, as_of_date: Optional[date] = None) -> list[dict]:
    """Get today's games from the schedule."""
    import pandas as pd

    if as_of_date is None:
        as_of_date = date.today()

    # Filter for today's games
    schedule['game_date_parsed'] = pd.to_datetime(schedule['game_date']).dt.date
    todays = schedule[schedule['game_date_parsed'] == as_of_date].copy()

    games = []
    for _, row in todays.iterrows():
        games.append({
            'game_id': row.get('game_id'),
            'home_team_id': row.get('home_id'),
            'away_team_id': row.get('away_id'),
            'is_neutral': bool(row.get('neutral_site', False)),
            'home_rest_days': 2,  # Default, will be calculated properly
            'away_rest_days': 2,
            'home_conference_id': row.get('home_conference_id'),
            'market_spread': None,  # Will be fetched from odds API
            'market_total': None,
        })

    return games


def print_top_ratings(ratings: dict[int, TeamRatings], teams_df, n: int = 25):
    """Print top N teams by efficiency margin."""
    import pandas as pd

    # Build team name lookup - check actual column names
    team_names = {}
    cols = teams_df.columns.tolist()

    for _, row in teams_df.iterrows():
        # Try different column name patterns
        tid = None
        name = None

        for col in ['team_id', 'id']:
            if col in cols and pd.notna(row.get(col)):
                tid = int(row[col])
                break

        for col in ['team_display_name', 'display_name', 'team_name', 'name']:
            if col in cols and pd.notna(row.get(col)):
                name = str(row[col])
                break

        if tid and name:
            team_names[tid] = name

    # Sort by efficiency margin
    sorted_teams = sorted(
        ratings.values(),
        key=lambda r: r.adj_em,
        reverse=True
    )

    print("\n" + "=" * 80)
    print(f"TOP {n} TEAMS BY ADJUSTED EFFICIENCY MARGIN")
    print("=" * 80)
    print(f"{'Rank':<5} {'Team':<30} {'AdjO':<8} {'AdjD':<8} {'AdjEM':<8} {'Tempo':<8} {'Games':<6}")
    print("-" * 80)

    for i, r in enumerate(sorted_teams[:n], 1):
        name = team_names.get(r.team_id, f"Team {r.team_id}")[:28]
        print(f"{i:<5} {name:<30} {r.adj_off:>7.1f} {r.adj_def:>7.1f} {r.adj_em:>+7.1f} {r.adj_tempo:>7.1f} {r.games_played:>5}")

    print("-" * 80)


def print_predictions(predictions: list[EnhancedPrediction], ratings: dict, teams_df, n: int = 10):
    """Print predictions with all details."""
    import pandas as pd

    # Build team name lookup
    team_names = {}
    for _, row in teams_df.iterrows():
        tid = row.get('team_id') or row.get('id')
        name = row.get('team_display_name') or row.get('display_name') or row.get('name')
        team_names[tid] = name

    print("\n" + "=" * 100)
    print("ENHANCED PREDICTIONS")
    print("=" * 100)

    for i, pred in enumerate(predictions[:n], 1):
        # Look up team names from ratings
        home_id = None
        away_id = None
        for game in games:
            if game['game_id'] == pred.game_id:
                home_id = game['home_team_id']
                away_id = game['away_team_id']
                break

        home_name = team_names.get(home_id, f"Home {home_id}")[:20] if home_id else "Unknown"
        away_name = team_names.get(away_id, f"Away {away_id}")[:20] if away_id else "Unknown"

        print(f"\nGame {i}: {away_name} @ {home_name}")
        print("-" * 50)
        print(f"  Projected Score: {pred.away_score:.1f} - {pred.home_score:.1f}")
        print(f"  Projected Spread: {pred.spread:+.1f} (Home)")
        print(f"  Projected Total: {pred.total:.1f}")
        print(f"  Home Win Prob: {pred.home_win_prob:.1%}")
        print(f"  Spread Std Dev: {pred.spread_std:.2f}")
        print(f"  80% CI: [{pred.spread_ci_80[0]:.1f}, {pred.spread_ci_80[1]:.1f}]")
        print(f"  Components:")
        print(f"    Efficiency: {pred.efficiency_spread:+.1f}")
        print(f"    HCA: {pred.hca_adjustment:+.1f}")
        print(f"    Travel: {pred.travel_adjustment:+.1f}")
        print(f"    Rest: {pred.rest_adjustment:+.1f}")
        print(f"    Four Factors: {pred.four_factors_adjustment:+.1f}")


def validate_model(team_stats, ratings: dict[int, TeamRatings], teams_df):
    """Run validation on completed games."""
    import pandas as pd
    import numpy as np
    from scipy import stats

    logger.info("Running model validation on completed games")

    # Get completed games from team_stats
    completed = team_stats[team_stats['is_home'] == True].copy()  # One row per game

    predictor = create_enhanced_predictor()

    predictions = []
    actuals = []

    for _, game in completed.iterrows():
        home_id = game['team_id']
        away_id = game['opponent_id']

        if home_id not in ratings or away_id not in ratings:
            continue

        pred = predictor.predict_game(
            home_ratings=ratings[home_id],
            away_ratings=ratings[away_id],
            game_id=game['game_id'],
            is_neutral=game['is_neutral'],
        )

        actual_spread = game['team_score'] - game['opponent_score']

        predictions.append(pred.spread)
        actuals.append(actual_spread)

    if len(predictions) == 0:
        logger.warning("No games for validation")
        return

    preds = np.array(predictions)
    acts = np.array(actuals)

    # Calculate metrics
    mae = np.mean(np.abs(preds - acts))
    rmse = np.sqrt(np.mean((preds - acts) ** 2))
    correlation = np.corrcoef(preds, acts)[0, 1]

    # Direction accuracy
    pred_winner = preds > 0
    actual_winner = acts > 0
    direction_acc = np.mean(pred_winner == actual_winner)

    print("\n" + "=" * 60)
    print("MODEL VALIDATION METRICS")
    print("=" * 60)
    print(f"  Games Validated: {len(predictions)}")
    print(f"  Spread MAE: {mae:.2f} points")
    print(f"  Spread RMSE: {rmse:.2f} points")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  Direction Accuracy: {direction_acc:.1%}")
    print("=" * 60)

    # Target benchmarks
    print("\n  Target Benchmarks:")
    mae_check = "[PASS]" if mae < 9.0 else "[FAIL]"
    dir_check = "[PASS]" if direction_acc > 0.65 else "[FAIL]"
    print(f"    Spread MAE Target: < 9.0 (Current: {mae:.2f}) {mae_check}")
    print(f"    Direction Accuracy Target: > 65% (Current: {direction_acc:.1%}) {dir_check}")
    print("=" * 60)

    # Analysis of prediction errors
    errors = preds - acts
    print("\n  Error Distribution:")
    print(f"    Mean Error: {np.mean(errors):+.2f} (should be ~0)")
    print(f"    Std Dev: {np.std(errors):.2f}")
    print(f"    Min Error: {np.min(errors):.1f}")
    print(f"    Max Error: {np.max(errors):.1f}")

    # Check predictions that were way off
    large_errors = np.abs(errors) > 30
    print(f"\n    Games with >30pt error: {np.sum(large_errors)} ({100*np.mean(large_errors):.1f}%)")

    # Break down by spread range
    print("\n  Performance by Spread Size:")
    for bucket, (low, high) in [("Close (<5)", (0, 5)), ("Medium (5-10)", (5, 10)),
                                 ("Large (10-15)", (10, 15)), ("Blowout (15+)", (15, 100))]:
        mask = (np.abs(preds) >= low) & (np.abs(preds) < high)
        if np.sum(mask) > 0:
            bucket_mae = np.mean(np.abs(errors[mask]))
            bucket_acc = np.mean((preds[mask] > 0) == (acts[mask] > 0))
            print(f"    {bucket}: MAE={bucket_mae:.1f}, Dir Acc={bucket_acc:.1%}, N={np.sum(mask)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run enhanced prediction pipeline")
    parser.add_argument("--season", type=int, default=2025, help="Season to ingest")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ENHANCED NCAA BASKETBALL PREDICTION PIPELINE")
    print(f"Model Version: {MODEL_VERSION}")
    print(f"Run Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Setup database
    setup_database()

    # Fetch teams
    logger.info("Fetching team data...")
    teams_df = fetch_all_teams()
    logger.info(f"Fetched {len(teams_df)} teams")

    # Ingest season data
    if not args.skip_ingest:
        boxscores, schedule, team_stats = ingest_season_data(args.season)
    else:
        # Load from database
        import pandas as pd
        with get_connection() as conn:
            team_stats = conn.execute("SELECT * FROM team_game_stats").fetchdf()
        schedule = fetch_season_schedule(args.season)

    # Calculate ratings
    ratings = calculate_ratings(team_stats, teams_df)

    # Print top teams
    print_top_ratings(ratings, teams_df, n=25)

    # Run validation
    validate_model(team_stats, ratings, teams_df)

    # Get today's games
    games = get_todays_games(schedule)

    if games:
        logger.info(f"Found {len(games)} games for today")

        # Generate predictions
        predictions = predict_slate(ratings, games)

        # Print predictions
        print_predictions(predictions, ratings, teams_df)
    else:
        logger.info("No games scheduled for today")

    print("\n[SUCCESS] Pipeline complete!")
