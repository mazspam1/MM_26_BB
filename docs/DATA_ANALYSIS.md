# Dashboard Data Analysis: Real vs Mock/Hardcoded

## Overview
This document breaks down what data in the dashboard is **REAL** (from database/API) vs **MOCK/HARDCODED**.

---

## ✅ REAL DATA (From Database/API)

### 1. **Header Metrics** (Top 4 Cards)
- **Games Today** (`games-count`): ✅ **REAL** - Fetched from `/slate` endpoint, comes from `data.games_count` which is a COUNT query from the `games` table
- **Predictions** (`predictions-count`): ✅ **REAL** - Calculated from `allGames.filter(g => g.proj_spread !== null).length` - counts games with actual predictions
- **Model Version** (`model-version`): ✅ **REAL** - From `/health` endpoint, comes from `settings.model_version` (default: "v0.1.0" in config, but can be overridden)
- **Teams Tracked** (`teams-count`): ✅ **REAL** - From `/health` endpoint, comes from `SELECT COUNT(*) FROM teams` query

### 2. **Games Table Data**
All data comes from `/slate` endpoint which queries the database:

- **Time**: ✅ **REAL** - From `game_datetime` in `games` table
- **Matchup** (Away/Home teams): ✅ **REAL** - From `away_team_name` and `home_team_name` in `games` table
- **Model Spread** (`proj_spread`): ✅ **REAL** - From `predictions.proj_spread` table (or 0.0 if placeholder)
- **Vegas Spread** (`market_spread`): ?o. **REAL** - From latest `betting_splits.spread_line_home` via `/slate` (model convention: positive = home favored)
- **Edge** (`edge_vs_spread`): ?o. **REAL** - Computed in API: `proj_spread - market_spread` (model convention)
- **Model O/U** (`proj_total`): ✅ **REAL** - From `predictions.proj_total` table
- **Vegas O/U** (`market_total`): ?o. **REAL** - From latest `betting_splits.total_line` via `/slate` (NULL if missing)
- **Spread Handle**: ?s??,? **REAL** - From latest `betting_splits` snapshot (requires scraper to run)
- **O/U Handle**: ?s??,? **REAL** - From latest `betting_splits` snapshot (requires scraper to run)
- **Play** (recommendation): ✅ **REAL** - Calculated from `edge_vs_spread` and `market_spread` values

### 3. **Top Picks Panel**
- **All data**: ✅ **REAL** - Derived from the same `allGames` array from `/slate` endpoint
- **Filtering**: Sorted by `absEdge` (absolute edge value)
- **Confidence rating**: From `confidence_rating` field in `predictions` table
- **Recommended units**: From `recommended_units` field in `predictions` table

---

## ⚠️ PLACEHOLDER/MOCK DATA

### 1. **Placeholder Predictions**
When a game exists in the `games` table but has NO prediction yet, the API creates a placeholder:

```python
def _create_placeholder_prediction():
    proj_home_score=0.0,      # ❌ MOCK - Hardcoded to 0.0
    proj_away_score=0.0,      # ❌ MOCK - Hardcoded to 0.0
    proj_spread=0.0,           # ❌ MOCK - Hardcoded to 0.0
    proj_total=0.0,            # ❌ MOCK - Hardcoded to 0.0
    home_win_prob=0.5,         # ❌ MOCK - Hardcoded to 50%
    spread_ci_50=(0.0, 0.0),   # ❌ MOCK - Hardcoded zeros
    spread_ci_80=(0.0, 0.0),   # ❌ MOCK - Hardcoded zeros
    spread_ci_95=(0.0, 0.0),   # ❌ MOCK - Hardcoded zeros
    total_ci_50=(0.0, 0.0),    # ❌ MOCK - Hardcoded zeros
    total_ci_80=(0.0, 0.0),    # ❌ MOCK - Hardcoded zeros
    total_ci_95=(0.0, 0.0),    # ❌ MOCK - Hardcoded zeros
    model_version="pending",    # ❌ MOCK - Hardcoded string
```

**Impact**: Games without predictions will show:
- Spread: `+0.0` or `-0.0`
- Total: `0` or `--`
- Edge: `--` (no edge calculated)
- Play: `NO EDGE`

### 2. **Default Values in Dashboard**
- **Win Probability fallback**: If `home_win_prob` is null, defaults to `50` (line 936)
- **Play text fallback**: If no edge or no vegas spread, shows `'NO EDGE'` (line 962)
- **Empty state messages**: Hardcoded strings like "No games found", "Loading games..."

### 3. **API Health Endpoint Fallbacks**
If database connection fails:
- `teams_count = 0` (fallback)
- `games_count = 0` (fallback)
- `status = "degraded"` (fallback)

---

## 🔍 DATA SOURCES BREAKDOWN

### Database Tables Used:
1. **`games`** - ✅ Real game data (teams, dates, times, scores)
2. **`predictions`** - ✅ Real model predictions (if exists, otherwise placeholder)
3. **`teams`** - ✅ Real team data
4. **`betting_splits`** - ⚠️ Real table but likely empty (needs scraper execution)

### API Endpoints:
- **`GET /health`**: Returns real database counts and connection status
- **`GET /slate?target_date=YYYY-MM-DD`**: Returns real games and predictions from database

### Hardcoded Values:
- **Model version**: `"v0.1.0"` in `packages/common/config.py` (can be overridden via env)
- **API base URL**: `'http://localhost:2500'` in dashboard HTML (line 874)
- **Placeholder prediction values**: All zeros and 0.5 for win prob

---

## 📊 SUMMARY

### Real Data Sources:
- ✅ Game schedules (teams, dates, times)
- ✅ Team names and IDs
- ✅ Model predictions (when they exist in `predictions` table)
- ✅ Market spreads/totals (when they exist in `predictions` table)
- ✅ Edge calculations (derived from real model vs market data)
- ✅ Database counts (teams, games)
- ✅ Model version (from config)

### Mock/Placeholder Data:
- ❌ Predictions for games without model output (all zeros)
- ❌ Betting splits (table exists but empty - needs scraper)
- ❌ Fallback values when data is missing (0.5 for win prob, "NO EDGE" for plays)

### Missing Data Indicators:
- `--` shown when: `null`, `undefined`, or empty
- `0.0` spread/total: Likely a placeholder (game has no prediction)
- `"pending"` model version: Placeholder prediction

---

## 🚨 KEY FINDINGS

1. **Betting Splits are EMPTY**: The `betting_splits` table exists but has no data. The scraper needs to be run: `python packages/ingest/fetch_betting_splits.py`

2. **Placeholder Predictions**: Games without predictions show all zeros. This is intentional - they're waiting for the prediction pipeline to run.

3. **All Game Data is Real**: The games themselves (matchups, dates, times) are real from the database.

4. **Model Predictions are Real**: When they exist in the `predictions` table, all values are real model outputs.

5. **No Hardcoded Game Data**: Unlike some mock dashboards, there are no fake games or hardcoded matchups - everything comes from the database.

---

## 🔧 TO GET FULLY REAL DATA

1. **Run prediction pipeline** to populate `predictions` table (eliminates placeholder predictions)
2. **Run betting splits scraper** to populate `betting_splits` table: `python packages/ingest/fetch_betting_splits.py`
3. **Ensure games are ingested** from ESPN/SportsDataVerse into `games` table
4. **Ensure market odds are fetched** and stored in `predictions.market_spread` and `predictions.market_total`

