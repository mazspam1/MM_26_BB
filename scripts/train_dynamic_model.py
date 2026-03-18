"""
Script to train and verify the new Dynamic State-Space Model.

This script:
1. Fetches 2025 season data.
2. Formats it for the GPU Bayesian model.
3. Fits the model to learn weekly team evolution.
4. Reports learned parameters (Volatility, Rest Advantage).
5. Demonstrates a prediction.
"""

import sys
import os
import pandas as pd
import numpy as np
import structlog

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.models.gpu_bayes import HierarchicalNCAABModel, GPUModelConfig
from packages.ingest.espn_enhanced import ingest_full_season, fetch_season_schedule

logger = structlog.get_logger()

def prepare_training_data(season=2025):
    """
    Load data and format it for the model.
    Merges home and away rows into single game records.
    """
    logger.info("Fetching training data", season=season)
    
    # 1. Get Team Stats (includes scores and rest days)
    # This might take a moment if not cached
    stats_df = ingest_full_season(season)
    
    if stats_df.empty:
        logger.error("No data found for season", season=season)
        return [], {}, {}

    # 2. Get Schedule (for neutral site info if needed, though stats_df has it)
    # stats_df has 'is_neutral', 'is_home'.
    
    # Filter to only completed games
    games_df = stats_df[stats_df['team_score'] > 0].copy()
    
    # 3. Create Game Records (Home vs Away)
    # We have 2 rows per game. We need to pair them.
    # Easiest way: Split into Home/Neutral_Home and Away/Neutral_Away
    
    # For neutral games, ESPN designates one as "home" (bottom of bracket).
    # Our ingest preserves this logic in 'is_home'.
    
    home_games = games_df[games_df['is_home'] == True].set_index('game_id')
    away_games = games_df[games_df['is_home'] == False].set_index('game_id')
    
    # Join on game_id
    merged = home_games.join(away_games, lsuffix='_home', rsuffix='_away', how='inner')
    
    # DEBUG: Check columns and types
    print("\nDEBUG: Merged DataFrame info:")
    print(merged[['team_id_home', 'team_id_away']].dtypes)
    print(merged[['team_id_home', 'team_id_away']].head())
    
    # Force int type to avoid object/dict issues
    merged['team_id_home'] = pd.to_numeric(merged['team_id_home'], errors='coerce').fillna(-1).astype(int)
    merged['team_id_away'] = pd.to_numeric(merged['team_id_away'], errors='coerce').fillna(-1).astype(int)
    
    training_data = []
    
    # Mappings
    # Filter out invalid IDs (-1)
    valid_ids = set(merged['team_id_home']) | set(merged['team_id_away'])
    valid_ids.discard(-1)
    
    all_teams = valid_ids
    team_id_map = {tid: idx for idx, tid in enumerate(all_teams)}
    
    # Mock conference map for now (would fetch from DB in production)
    # The model handles unknown conferences gracefully (assigns to generic pool)
    conference_map = {tid: 0 for tid in all_teams}

    logger.info("Formatting games for model", count=len(merged))
    
    for game_id, row in merged.iterrows():
        # Calculate rest difference
        # rest_days is in the DF. 
        # Rest Advantage = Home Rest - Away Rest
        rest_diff = row['rest_days_home'] - row['rest_days_away']
        
        # Cap rest diff at +/- 5 days to avoid outliers from season openers
        rest_diff = max(min(rest_diff, 5), -5)
        
        training_data.append({
            "game_id": game_id,
            "game_date": row['game_date_home'], # Date object
            "home_team_id": row['team_id_home'],
            "away_team_id": row['team_id_away'],
            "home_score": row['team_score_home'],
            "away_score": row['team_score_away'],
            "is_neutral": row['is_neutral_home'],
            "rest_diff": int(rest_diff)
        })
        
    return training_data, team_id_map, conference_map

def main():
    # 1. Get Data
    games_data, team_map, conf_map = prepare_training_data(2025)
    
    if not games_data:
        print("No games found. Aborting.")
        return

    # 2. Configure Model
    config = GPUModelConfig(
        num_warmup=500,     # Lower for quick test
        num_samples=1000,   # Lower for quick test
        num_chains=1,       # 1 chain for speed
        heteroscedastic=True
    )
    
    model = HierarchicalNCAABModel(config)
    
    # 3. Fit Model
    print(f"\nTraining on {len(games_data)} games...")
    model.fit(games_data, team_map, conf_map)
    
    # 4. Analyze Learned Parameters
    samples = model.posterior_samples
    
    print("\n" + "="*50)
    print("PHD MODEL INSIGHTS")
    print("="*50)
    
    # Rest Factor
    rest_factor = float(np.mean(samples['rest_factor']))
    print(f"\n1. LEARNED REST ADVANTAGE:")
    print(f"   The model learned that 1 day of rest advantage is worth {rest_factor:.2f} points.")
    print(f"   (Old static model assumed 0.30)")
    
    # Volatility (Drift)
    sigma_off = float(np.mean(samples['sigma_drift_off']))
    print(f"\n2. TEAM VOLATILITY (Evolution per Week):")
    print(f"   Teams change their offensive rating by ~{sigma_off:.2f} points per week.")
    print(f"   (This allows the model to catch 'hot' teams automatically)")
    
    # HCA
    hca = float(np.mean(samples['hca_global']))
    print(f"\n3. HOME COURT ADVANTAGE:")
    print(f"   Global HCA is {hca:.2f} points.")
    
    # 5. Top 5 Teams (Current Form)
    ratings = model.get_team_ratings()
    sorted_teams = sorted(ratings.items(), key=lambda x: x[1]['adj_em'], reverse=True)
    
    print(f"\n4. TOP 5 TEAMS (Current Form - Final Week):")
    print(f"   {'Team ID':<10} {'AdjEM':<10} {'Off':<10} {'Def':<10}")
    print("-" * 45)
    for tid, r in sorted_teams[:5]:
        print(f"   {tid:<10} {r['adj_em']:<10.1f} {r['adj_off']:<10.1f} {r['adj_def']:<10.1f}")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
