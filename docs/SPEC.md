# 2026 March Madness Modeling System Spec

## Goal

Build a real-data, free-data-only NCAA men's tournament modeling system that produces:

- game-by-game projected scores for every round
- win probability, upset probability, spread, total, and score intervals
- full bracket advancement probabilities through the champion
- Monte Carlo bracket paths that propagate actual winners into later rounds
- honest calibration and backtests, with exact-score output treated as a derived estimate rather than the main target

The right standard is not "pick every exact score." The right standard is:

1. strong win probability calibration
2. low margin and total error
3. realistic uncertainty bands
4. bracket simulations that update matchup odds after each upset
5. out-of-sample performance that holds up against market closes and tournament results

## Current Repo Audit

The repo already has useful building blocks, but it is not yet at the level needed for a scary-good 2026 tournament system.

### What is already solid

- `packages/features/kenpom_ratings.py` has a legitimate opponent-adjusted efficiency framework with recency weighting and convergence logic.
- `packages/models/enhanced_predictor.py` already blends adjusted efficiency, Four Factors, travel, rest, and market anchoring hooks.
- `packages/models/tournament_predictor.py` separates tournament logic from regular-season prediction logic.
- `packages/simulation/bracket_simulator.py` already has bracket-DAG simulation scaffolding.
- `pytest -q` passes: 67 tests passed on this machine.

### What must be fixed before trusting tournament outputs

1. Real bracket source is not wired end-to-end.
   - `packages/ingest/tournament_bracket.py` still falls back to `BRACKET_2026_TEMPLATE`.
   - `scripts/run_march_madness.py` hardcodes a 2026 bracket dictionary.

2. Bracket propagation is not fully model-correct after upsets.
   - `packages/simulation/bracket_simulator.py` currently keeps the original win probability when later-round teams change.
   - This is the biggest simulation flaw. Every future matchup must be repriced from the actual surviving teams.

3. The backtest pipeline is not production-ready yet.
   - `python -m scripts.run_backtest --days 7` fails with a DuckDB DATE/VARCHAR mismatch in `packages/eval/backtest.py`.
   - If backtests are broken, model quality claims are not yet trustworthy.

4. The data store is incomplete for top-tier evaluation.
   - `scripts/check_db_state.py` reports 362 teams, 1400 team ratings, 2021 games, but no tournament tables.
   - Box-score-derived `team_game_stats` ingestion exists in code, but the current database state is not yet verified as full season-to-date and tournament-ready.

5. Current calibration still suggests heavy model correction.
   - `data/model_calibration.json` shows large total adjustments, which is a sign the pre-calibration model is still missing structure.

6. The rating universe needs D1 filtering.
   - A December backtest run built ratings for 673 to 700 teams, which strongly suggests non-D1 teams are entering the training set.
   - For March Madness, the core rating engine must operate on a clean Division I universe with explicit handling for non-D1 opponents.

## What "Top Tier" Actually Means

For this project, top tier means a three-engine system:

1. Rating engine
   - dynamic team offense, defense, tempo, and volatility
   - opponent-adjusted and recency-weighted

2. Matchup pricing engine
   - predicts margin, total, win probability, and score distribution for one game
   - uses tournament context and style interactions

3. Bracket simulation engine
   - simulates the entire field by repricing each future matchup from sampled team strengths
   - outputs round advancement, Final Four, title, and most likely path distributions

Exact score should be derived from predicted margin and total, or from possessions plus team PPP, not modeled as a naive single-regression target.

## Free Data Stack

Everything below can be done without paid APIs.

This repo's source policy is strict:

- runtime production data must come from free sources only
- paid KenPom, paid odds history, and paid injury feeds are not required inputs
- if we want KenPom-like information, we derive it from our own opponent-adjusted box-score model
- market anchoring is optional and only activates when free odds data is actually present

### Required feeds

- NCAA March Madness Live / NCAA.com:
  - official bracket
  - seeds
  - First Four routing
  - round structure
  - sites and dates
- ESPN / SportsDataverse / hoopR:
  - schedules
  - scores
  - team metadata
  - box scores
  - play-by-play where available
- The Odds API free tier when available:
  - pregame spreads
  - totals
  - moneylines
  - timestamps
- ESPN team pages and schedule data:
  - roster continuity proxies
  - recent form context
  - availability notes when free/publicly visible

### Five source buckets

1. Official bracket and tournament structure
   - NCAA March Madness Live is the source of truth for slot graph, seeds, and play-in destinations.

2. Official game and box data
   - ESPN plus SportsDataverse supply the free structured game, team, and box-score feeds used by the rating engine.

3. Self-derived advanced ratings
   - We do not depend on paid KenPom APIs.
   - Our adjusted-efficiency model is the free replacement for the core neutral-floor strength estimate.

4. Free market prices
   - The Odds API free tier is the primary market source.
   - If no free line data is present, the model runs in pure-model mode instead of fabricating market anchors.

5. Historical training and backtesting
   - Historical games, scores, and box scores come from free ESPN/SportsDataverse ingestion already stored locally.

### Local reference tables we should own

- team identity crosswalks
- arena coordinates
- timezone per team and site
- tournament site coordinates
- coach continuity and roster continuity features
- manually curated injury and rotation overrides for March only

## Modeling Blueprint

## 1. Team Strength Engine

Primary latent state per team on each date:

- offensive strength
- defensive strength
- tempo
- volatility
- three-point reliance and opponent three-point volatility exposure
- offensive rebound pressure
- turnover creation and turnover resistance
- foul-drawing and foul-avoidance profile

Recommended implementation:

- dynamic Bayesian hierarchical model in JAX/NumPyro
- weekly or daily latent-state updates
- partial pooling by conference and season phase
- explicit neutral-site handling

Output per team-date:

- `adj_off`
- `adj_def`
- `adj_tempo`
- `adj_em`
- `off_std`
- `def_std`
- `tempo_std`
- shot-profile factors
- recent-form deltas

## 2. Matchup Pricing Engine

Predict these directly for every game:

- expected possessions
- expected margin
- expected total
- win probability
- cover probability
- score intervals

Recommended stack:

1. Bayesian base model
   - possessions and PPP decomposition
   - interpretable
   - uncertainty-native

2. Gradient boosting residual model
   - XGBoost or CatBoost on engineered matchup features
   - corrects nonlinear interactions the base model misses

3. Calibration layer
   - isotonic or beta calibration for win probability
   - conformal intervals for margin and total

### Tournament-specific features

- neutral-site compression
- round number
- days since previous game
- travel distance to site
- timezone change to site
- seed difference
- seed as public-bias proxy, not talent proxy
- roster continuity
- coach tournament experience
- late-game foul inflation tendency
- three-point attempt rate and opponent 3PA allowed
- turnover pressure mismatch
- offensive rebound mismatch

### Score generation

Preferred approach:

1. predict `margin`
2. predict `total`
3. derive scores:

```text
home_score = (total + margin) / 2
away_score = (total - margin) / 2
```

Also support a possession-based path:

```text
score = possessions * points_per_possession
```

That lets us generate full score distributions during simulation.

## 3. Bracket Simulation Engine

This must be posterior-aware, not static.

### Correct simulation logic

For each simulation draw:

1. sample team latent strengths and game noise
2. simulate every Round of 64 game
3. take actual simulated winners into Round of 32
4. recalculate the next matchup from those two surviving teams
5. repeat until champion

### Non-negotiable rule

Never reuse a precomputed Round of 32 or Sweet 16 win probability if one or both teams changed due to an upset.

### Outputs

- champion probability
- title game probability
- Final Four probability
- Elite Eight probability
- expected wins
- most common bracket paths
- per-game most likely score and median score band

## GPU Plan For Your 5090

Use the GPU where it actually matters.

### Good GPU use

- JAX/NumPyro Bayesian sampling for dynamic team-strength models
- GPU XGBoost or CatBoost for matchup residual models
- vectorized Monte Carlo tournament simulation
- posterior predictive sampling for score distributions

### Keep on CPU

- DuckDB queries
- Polars or pandas feature engineering
- ingestion and schema validation
- ordinary backtest orchestration

### Recommended split

- 9950X3D: ETL, DuckDB, feature pipelines, Optuna orchestration, backtests
- RTX 5090: Bayesian fitting, boosted-tree training on large feature matrices, simulation at scale
- 128 GB RAM: in-memory multi-season feature store, cached posterior samples, large simulation buffers

## Data and Validation Rules

### Data rules

- no template bracket in live mode
- no hardcoded team lists in live mode
- no silent fallbacks for missing team IDs or missing lines
- log every data source timestamp and load version
- keep append-only odds snapshots

### Validation rules

- strict Pydantic contracts for all ingest and prediction rows
- unit tests for each feature builder
- integration tests for one full tournament slate
- rolling-origin backtests across multiple seasons
- separate regular-season and tournament evaluation

## Success Metrics

Bracket score alone is not enough.

Primary metrics:

- win-probability log loss
- Brier score
- spread MAE
- total MAE
- team-score MAE
- calibration by decile
- champion and Final Four probability calibration
- CLV vs close when market data exists

Secondary metrics:

- exact-score bucket hit rate
- upset detection recall and precision
- round-by-round bracket accuracy
- seed-bias residual analysis

## Immediate Repo Work Plan

### Phase 0 - Repair trust and data integrity

1. Replace template bracket loading with real NCAA/ESPN bracket ingest.
2. Persist tournament tables and slot graph for 2026.
3. Fix the backtest date casting bug in `packages/eval/backtest.py`.
4. Verify `team_game_stats` is fully populated season-to-date.
5. Add one command that reproduces the full tournament pipeline from raw data.

### Phase 1 - Make the current model tournament-safe

1. Reprice every simulated future matchup from actual advancing teams.
2. Add tournament-site travel and timezone calculations against site coordinates, not home arenas.
3. Remove hardcoded bracket assumptions from scripts.
4. Add first-four aware bracket propagation.

### Phase 2 - Upgrade team strength

1. Fit a dynamic Bayesian offense/defense/tempo model on full season box score data.
2. Add uncertainty on team states, not just on final spread/total.
3. Separate early season, conference play, and postseason regimes.

### Phase 3 - Add matchup ensemble

1. Train boosted-tree models for margin and total residuals.
2. Add shot-profile, rebounding, turnover, and foul-pressure features.
3. Calibrate probabilities and intervals out of sample.

### Phase 4 - Tournament simulation at scale

1. Generate at least 100,000 tournament simulations from posterior draws.
2. Save bracket-path outputs and advancement probabilities.
3. Produce one canonical `predictions/tournament_2026/` artifact set.

### Phase 5 - Final selection and lock

1. Freeze the model version before the Round of 64.
2. Export per-game projections and full bracket probabilities.
3. Track actual results and post-mortem every round.

## Definition Of Done

The system is ready for 2026 tournament deployment when all of the following are true:

- the bracket is loaded from a real source, not a template
- all 2025-26 season team-game stats are in DuckDB
- rolling backtests run cleanly across prior seasons
- simulation reprices every future matchup dynamically
- every game output includes score, margin, total, win probability, and uncertainty bands
- every team output includes advancement probabilities through champion
- one command can regenerate the entire tournament package locally

## First Build Slice

The fastest high-value slice is:

1. ingest real 2026 bracket into tournament tables
2. verify full season box-score ingestion
3. fix backtest so we can measure quality honestly
4. repair bracket simulator to reprice future matchups
5. then train and compare a stronger matchup model

That is the shortest path from "interesting project" to "serious tournament system."
