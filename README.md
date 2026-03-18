# March Madness 2026 - NCAA Basketball Prediction System

A research-grade NCAA men's basketball spread, total, and tournament prediction system. Features KenPom-style opponent-adjusted efficiency ratings, Four Factors analysis, Monte Carlo bracket simulation (GPU-accelerated with CuPy on NVIDIA GPUs), and an interactive March Madness bracket dashboard.

## Requirements

- **Python 3.11+**
- **Windows** (PowerShell 5.1+ for `start.ps1` launcher)
- **Optional**: NVIDIA GPU with CUDA 12.8+ for GPU-accelerated Monte Carlo simulations (CuPy, JAX, NumPyro)

### Python Dependencies

All dependencies are managed via `pyproject.toml`. Key packages:

| Category | Packages |
|---|---|
| Data Ingestion | `sportsdataverse`, `httpx` |
| Data Processing | `pandas`, `polars`, `duckdb`, `pyarrow` |
| Validation | `pydantic`, `pydantic-settings` |
| Bayesian Modeling | `pymc`, `arviz`, `scipy`, `numpy` |
| Uncertainty | `mapie`, `scikit-learn` |
| API | `fastapi`, `uvicorn`, `apscheduler` |
| Dashboard | `streamlit`, `plotly` |
| Logging | `structlog` |

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/mazspam1/MM_26_BB.git
cd MM_26_BB
```

### 2. Setup (creates venv + installs dependencies)

```powershell
.\start.ps1 setup
```

This will:
- Create a `.venv` virtual environment
- Install all Python dependencies via `pip install -e ".[dev]"`
- Copy `.env.example` to `.env` if `.env` doesn't exist

### 3. Configure environment

Edit `.env` with your settings:

```env
# The Odds API (free tier: 500 requests/month)
# Get your key at: https://the-odds-api.com/
ODDS_API_KEY=your_api_key_here

# Database path (DuckDB, auto-created)
DATABASE_PATH=data/cbb_lines.duckdb

ENVIRONMENT=development
MODEL_CALIBRATION_PATH=data/model_calibration.json
MIN_GAMES_PLAYED=8
QUALITY_REPORT_DIR=data/reports
```

> **Note**: The system uses free ESPN data for scores and box scores. The Odds API key is optional (free tier available) and only needed for market odds/edge calculations.

### 4. Run

```powershell
# Full daily pipeline: ingest data + calculate ratings + predict + launch dashboard
.\start.ps1 full

# Or run individual steps:
.\start.ps1 ingest      # Fetch schedule, box scores, odds
.\start.ps1 ratings     # Calculate KenPom-style ratings
.\start.ps1 predict     # Generate spread/total predictions
.\start.ps1 dashboard   # Launch dashboard UI (http://localhost:2501)
```

### 5. March Madness Bracket

```powershell
# Run tournament Monte Carlo simulation
.\start.ps1 tournament

# Render interactive bracket (opens in browser)
.\start.ps1 bracket
```

## `start.ps1` Command Reference

| Command | Description |
|---|---|
| `full` | Full daily run: ingest + ratings + predict + dashboard (default) |
| `quick` | Predict only + dashboard (needs existing data) |
| `pipeline` | PhD-grade pipeline (Bayesian + RAPM) |
| `setup` | Install dependencies and create venv |
| `ingest` | Fetch schedule, box scores, odds |
| `ratings` | Calculate KenPom-style adjusted efficiency ratings |
| `predict` | Generate spread/total predictions |
| `splits` | Fetch DraftKings betting splits |
| `backtest` | Run 30-day rolling backtest |
| `calibrate` | Fit model calibration |
| `report` | Generate data quality report |
| `tournament` | Run March Madness Monte Carlo simulation |
| `bracket` | Render visual bracket (opens in browser) |
| `api` | Start FastAPI server (port 2500) |
| `dashboard` | Start bracket dashboard (port 2502) |
| `worker` | Start background scheduler |
| `test` | Run pytest test suite |
| `stop` | Stop all services |
| `status` | Show running services |
| `help` | Show help |

### Options

```powershell
.\start.ps1 full -Date 2026-03-17    # Run for a specific date
.\start.ps1 full -NoBrowser           # Don't auto-open browser
.\start.ps1 full -NoGPU              # Disable GPU acceleration
```

## Services

| Service | URL |
|---|---|
| Dashboard | http://localhost:2502 |
| API | http://localhost:2500 |
| API Docs | http://localhost:2500/docs |

## Project Structure

```
NCAAB/
  apps/
    api/              # FastAPI endpoints
    dashboard/        # Bracket dashboard + Streamlit UI
    worker/           # Background scheduler
  packages/
    ingest/           # ESPN data connectors
    features/         # KenPom ratings, Four Factors, conference HCA
    models/           # Enhanced predictor, Bayesian core
    simulation/       # Monte Carlo bracket simulator (GPU support)
    common/           # Database, schemas, logging
  scripts/            # CLI entry points (pipeline, tournament, etc.)
  data/               # DuckDB database + model artifacts (gitignored)
  predictions/        # Generated predictions + bracket HTML
  tests/              # pytest suite
  docs/               # Specs and model cards
```

## GPU Acceleration

For NVIDIA GPUs (RTX 3000+ recommended, tested on RTX 5090):

```powershell
pip install -e ".[gpu]"
```

This installs JAX with CUDA 12, NumPyro, and CuPy for GPU-accelerated Monte Carlo tournament simulations.

## License

MIT
