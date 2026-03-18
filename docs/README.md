# CBB Lines

PhD-grade NCAA Basketball spread/total prediction system.

Free sources only. No paid APIs are part of the production plan.

## Features

- Daily predictions for all D1 games
- Uncertainty quantification (50/80/95% confidence intervals)
- Closing Line Value (CLV) backtesting
- Market edge detection
- FastAPI service + Streamlit dashboard
- Rolling-origin backtests with run tracking
- Persisted calibration + market anchoring
- Segment diagnostics (coverage drift, line availability, CLV by segment)
- Daily data-quality reports and guardrails

## Quick Start

```powershell
# Setup
.\start.ps1 setup

# Ingest recent games + boxscores
.\start.ps1 ingest

# Update ratings
.\start.ps1 ratings

# Generate predictions
.\start.ps1 predict

# Run backtest (latest 30 days)
.\start.ps1 backtest

# Run an uncalibrated backtest when refitting model calibration
python -m scripts.run_backtest --start 2025-11-03 --end 2025-12-23 --uncalibrated

# Fit calibration from latest backtest
.\start.ps1 calibrate

# Or fit on the most recent 500 backtest predictions
python -m scripts.fit_calibration --max-samples 500

# Generate daily quality report
.\start.ps1 report

# Run API
.\start.ps1 api

# Run Dashboard
.\start.ps1 dashboard
```

## Data Sources

- **NCAA March Madness Live data**: official bracket structure, seeds, play-in routing, sites
- **ESPN / sportsdataverse-py**: schedules, box scores, play-by-play, team metadata
- **The Odds API**: spreads, totals, moneylines on the free tier when configured

Advanced paid rating feeds are intentionally out of scope. We build our own adjusted-efficiency engine from free box-score data.

## Model

KenPom/Torvik-style tempo-based Bayesian hierarchical model:
- Adjusted offensive/defensive efficiency
- Hierarchical home court advantage
- Four Factors integration
- Market-aware priors (anchoring)
- Calibrated prediction intervals
