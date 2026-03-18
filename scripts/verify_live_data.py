
import sys
import os
from datetime import date
import pandas as pd

# Add project root to path
sys.path.insert(0, os.getcwd())

try:
    from apps.dashboard.app import fetch_slate
    from packages.common.config import get_settings
    print("Successfully imported modules.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def verify_live_data():
    print("--- Verifying Live Data Integration ---")
    today = date.today()
    print(f"Target Date: {today}")
    
    # 1. Fetch Slate
    print("Fetching slate via dashboard logic...")
    try:
        df = fetch_slate(today)
    except Exception as e:
        print(f"CRITICAL ERROR fetching slate: {e}")
        return

    if df.empty:
        print(f"WARNING: No games found for {today}. This might be expected if no games are scheduled.")
        # Try a known date if today is empty? 
        # For verification, we just report what we found.
    else:
        print(f"SUCCESS: Found {len(df)} games.")
        
        # 2. Verify Data Quality
        print("\nData Quality Checks:")
        cols = df.columns.tolist()
        print(f"Columns found: {cols}")
        
        # Check for Vegas data
        if 'market_spread' in cols:
            market_count = df['market_spread'].count()
            print(f"Market Spreads Available: {market_count}/{len(df)}")
            if market_count > 0:
                print("Confirmed: Real-time Vegas odds are present.")
            else:
                print("Warning: Market spread column exists but is empty.")
        else:
            print("FAILED: 'market_spread' column missing.")

        # Check for Predictions
        if 'proj_spread' in cols:
             print("Confirmed: Model predictions (proj_spread) are present.")
        
        # Check for Betting Splits
        if 'spread_favored_handle_pct' in cols:
             count = df['spread_favored_handle_pct'].count()
             print(f"Betting Splits Available: {count}/{len(df)}")
             if count > 0:
                 print("Confirmed: DraftKings betting splits are loaded.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_live_data()
