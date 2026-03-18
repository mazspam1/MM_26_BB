# ============================================================
#  CBB Lines - NCAAB Prediction System  (PhD-Grade v2.0)
#  Pure ASCII - works everywhere
# ============================================================

param(
    [Parameter(Position = 0)]
    [ValidateSet("full", "quick", "pipeline", "setup", "api", "dashboard", "worker",
                 "ingest", "ratings", "predict", "splits", "backtest", "calibrate", "report",
                 "tournament", "bracket", "test", "stop", "status", "help")]
    [string]$Command = "full",

    [Parameter()]
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),

    [Parameter()]
    [switch]$NoBrowser,

    [Parameter()]
    [switch]$NoGPU
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Ports = @(2500, 2501, 2502)

# Activate virtual environment
$VenvPath = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPath) {
    . $VenvPath
}
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

function Write-Step  { param([string]$m) Write-Host "  [$((Get-Date).ToString('HH:mm:ss'))] $m" -ForegroundColor Cyan }
function Write-OK    { param([string]$m) Write-Host "  [OK] $m" -ForegroundColor Green }
function Write-Warn  { param([string]$m) Write-Host "  [!!] $m" -ForegroundColor Yellow }
function Write-Err   { param([string]$m) Write-Host "  [XX] $m" -ForegroundColor Red }
function Write-Header { param([string]$m)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor DarkCyan
    Write-Host "  $m" -ForegroundColor White
    Write-Host ("=" * 60) -ForegroundColor DarkCyan
    Write-Host ""
}

function Test-PortOpen {
    param([int]$Port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", $Port)
        $tcp.Close()
        return $true
    } catch {
        return $false
    }
}

function Stop-All {
    param([switch]$Silent)
    if (-not $Silent) { Write-Step "Stopping all services..." }
    Get-Job | Where-Object { $_.State -eq 'Running' } | Stop-Job -ErrorAction SilentlyContinue
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
    foreach ($port in $Ports) {
        $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        foreach ($conn in $conns) {
            try {
                $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
                if ($proc -and $proc.ProcessName -match 'python|uvicorn') {
                    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                }
            } catch {}
        }
    }
    if (-not $Silent) { Write-OK "All services stopped" }
}

function Show-Help {
    Write-Host @"

  CBB Lines - NCAAB Prediction System (PhD-Grade v2.0)
  ====================================================

  MAIN COMMANDS
    full        Full daily run: ingest + ratings + predict + dashboard
    quick       Predict only + dashboard (needs existing data)
    pipeline    PhD-grade pipeline (Bayesian + Injuries + RAPM)
    setup       Install dependencies

  INDIVIDUAL STEPS
    ingest      Fetch schedule, box scores, odds
    ratings     Calculate KenPom-style ratings
    predict     Generate predictions
    splits      Fetch DraftKings betting splits
    backtest    Run 30-day backtest
    calibrate   Fit model calibration
    report      Data quality report

  SERVICES
    api         Start FastAPI server (port 2500)
    dashboard   Start Streamlit dashboard (port 2501)
    worker      Start background scheduler
    stop        Kill all services
    status      Show running services

  OTHER
    tournament  March Madness simulation
    bracket     Render visual bracket
    test        Run test suite
    help        Show this help

  URLS
    Dashboard:  http://localhost:2501
    API:        http://localhost:2500
    API Docs:   http://localhost:2500/docs

"@ -ForegroundColor Gray
}

function Install-Dependencies {
    Write-Header "SETUP"
    if (-not (Test-Path (Join-Path $ProjectRoot ".venv"))) {
        Write-Step "Creating virtual environment..."
        python -m venv .venv
        . (Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1")
    }
    Write-Step "Installing packages..."
    pip install -e ".[dev]" 2>&1 | Out-Null
    if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
        Copy-Item ".env.example" ".env"
        Write-Warn "Created .env - add API keys if needed"
    }
    Write-OK "Setup complete"
}

function Invoke-Ingest {
    param([string]$TargetDate)
    Write-Header "DATA INGESTION ($TargetDate)"
    Write-Step "Fetching schedule + box scores..."
    & $PythonExe scripts/ingest_recent.py --end $TargetDate --days-back 7
    Write-OK "Ingestion complete"
}

function Invoke-Ratings {
    Write-Header "CALCULATING RATINGS"
    & $PythonExe -c "
import sys; sys.path.insert(0, '.')
from datetime import date
from packages.features.kenpom_ratings import calculate_adjusted_ratings, save_ratings_to_db
from packages.common.database import get_connection
import pandas as pd
print('  Loading team game stats...')
with get_connection() as conn:
    try:
        team_stats = pd.read_sql('SELECT * FROM team_game_stats', conn)
    except:
        team_stats = pd.read_sql('SELECT bs.team_id, g.game_date, CASE WHEN bs.team_id = g.home_team_id THEN TRUE ELSE FALSE END as is_home, g.neutral_site as is_neutral, CASE WHEN bs.team_id = g.home_team_id THEN g.away_team_id ELSE g.home_team_id END as opponent_id, (bs.points / GREATEST(bs.field_goals_attempted - bs.offensive_rebounds + bs.turnovers + 0.475 * bs.free_throws_attempted, 1)) * 100 as off_rating, (obs.points / GREATEST(obs.field_goals_attempted - obs.offensive_rebounds + obs.turnovers + 0.475 * obs.free_throws_attempted, 1)) * 100 as def_rating, (bs.field_goals_attempted - bs.offensive_rebounds + bs.turnovers + 0.475 * bs.free_throws_attempted) as possessions, (bs.field_goals_made + 0.5 * bs.three_pointers_made) / GREATEST(bs.field_goals_attempted, 1) as off_efg, bs.turnovers / GREATEST(bs.field_goals_attempted - bs.offensive_rebounds + bs.turnovers + 0.475 * bs.free_throws_attempted, 1) as off_tov, bs.offensive_rebounds / GREATEST(bs.offensive_rebounds + obs.defensive_rebounds, 1) as off_orb, bs.free_throws_attempted / GREATEST(bs.field_goals_attempted, 1) as off_ftr, (obs.field_goals_made + 0.5 * obs.three_pointers_made) / GREATEST(obs.field_goals_attempted, 1) as def_efg, obs.turnovers / GREATEST(obs.field_goals_attempted - obs.offensive_rebounds + obs.turnovers + 0.475 * obs.free_throws_attempted, 1) as def_tov, obs.offensive_rebounds / GREATEST(obs.offensive_rebounds + bs.defensive_rebounds, 1) as def_orb, obs.free_throws_attempted / GREATEST(obs.field_goals_attempted, 1) as def_ftr FROM box_scores bs JOIN games g ON bs.game_id = g.game_id JOIN box_scores obs ON obs.game_id = g.game_id AND obs.team_id != bs.team_id WHERE g.status = final ORDER BY g.game_date DESC LIMIT 10000', conn)
if len(team_stats) == 0:
    print('  No game data found. Run: start.ps1 ingest')
    exit(1)
print(f'  Found {len(team_stats)} team-game records')
ratings = calculate_adjusted_ratings(team_stats, as_of_date=date.today())
if ratings:
    save_ratings_to_db(ratings)
    print(f'  Calculated ratings for {len(ratings)} teams')
    sorted_r = sorted(ratings.values(), key=lambda r: r.adj_em, reverse=True)
    print()
    print('  TOP 10 BY ADJUSTED EFFICIENCY MARGIN')
    print('  ' + '-' * 55)
    for i, r in enumerate(sorted_r[:10], 1):
        print(f'  {i:2}. Team {r.team_id:5} | EM: {r.adj_em:+6.1f} | Off: {r.adj_off:5.1f} | Def: {r.adj_def:5.1f}')
else:
    print('  ERROR: No ratings calculated')
    exit(1)
"
    Write-OK "Ratings updated"
}

function Invoke-Predict {
    param([string]$TargetDate)
    Write-Header "GENERATING PREDICTIONS ($TargetDate)"
    $pipelineScript = Join-Path $ProjectRoot "scripts\daily_pipeline.py"
    $legacyScript = Join-Path $ProjectRoot "scripts\run_predictions.py"
    if (Test-Path $pipelineScript) {
        Write-Step "Running PhD-grade pipeline..."
        & $PythonExe $pipelineScript --date $TargetDate --no-fetch
    } else {
        Write-Step "Running legacy predictor..."
        & $PythonExe $legacyScript --date $TargetDate
    }
    Write-OK "Predictions generated"
}

function Invoke-Pipeline {
    param([string]$TargetDate)
    Write-Header "PHD-GRADE PIPELINE ($TargetDate)"
    $pipelineScript = Join-Path $ProjectRoot "scripts\daily_pipeline.py"
    if (Test-Path $pipelineScript) {
        & $PythonExe $pipelineScript --date $TargetDate
    } else {
        Write-Err "scripts/daily_pipeline.py not found"
        exit 1
    }
    Write-OK "Pipeline complete"
}

function Invoke-Backtest {
    Write-Header "BACKTEST (30 days)"
    & $PythonExe scripts/run_backtest.py --days 30
    Write-OK "Backtest complete"
}

function Invoke-Calibrate {
    Write-Header "CALIBRATION"
    & $PythonExe scripts/fit_calibration.py
    Write-OK "Calibration complete"
}

function Invoke-Splits {
    Write-Header "BETTING SPLITS"
    & $PythonExe -c "
import sys; sys.path.insert(0, '.')
from datetime import date
from packages.ingest.draftkings_splits import DraftKingsSplitsScraper, save_betting_splits_to_db
from packages.common.database import get_connection
scraper = DraftKingsSplitsScraper()
try:
    spread_splits = scraper.fetch_spread_splits(days=1)
    total_splits = scraper.fetch_total_splits(days=1)
    print(f'  Spread splits: {len(spread_splits)} games')
    print(f'  Total splits: {len(total_splits)} games')
    today = date.today().isoformat()
    with get_connection() as conn:
        games = conn.execute('SELECT game_id, home_team_name, away_team_name FROM games WHERE game_date = ?', (today,)).fetchall()
    updated = 0
    for game_id, home_name, away_name in games:
        hl = home_name.lower(); al = away_name.lower()
        for s in spread_splits + total_splits:
            sh = s.get('home_team', '').lower(); sa = s.get('away_team', '').lower()
            if (hl.split()[-1] in sh or sh in hl) and (al.split()[-1] in sa or sa in al):
                save_betting_splits_to_db(game_id, s)
                updated += 1; break
    print(f'  Updated {updated} games')
finally:
    scraper.close()
"
    Write-OK "Splits fetched"
}

function Invoke-Report {
    Write-Header "QUALITY REPORT"
    & $PythonExe scripts/generate_quality_report.py --date $Date
    Write-OK "Report generated"
}

function Invoke-Tests {
    Write-Header "TEST SUITE"
    & $PythonExe -m pytest tests/ -v --tb=short --ignore=tests/unit/test_bracket_simulator.py
    Write-OK "Tests complete"
}

function Start-API {
    Write-Header "API SERVER (port 2500)"
    Write-Host "  API:  http://localhost:2500" -ForegroundColor Green
    Write-Host "  Docs: http://localhost:2500/docs" -ForegroundColor Gray
    Write-Host ""
    Set-Location $ProjectRoot
    & $PythonExe -m uvicorn apps.api.main:app --host 0.0.0.0 --port 2500 --reload
}

function Start-Dashboard {
    Write-Header "DASHBOARD (port 2501)"
    Write-Host "  Dashboard: http://localhost:2501" -ForegroundColor Green
    Write-Host ""
    Set-Location $ProjectRoot
    $streamlitApp = Join-Path $ProjectRoot "apps\dashboard\app.py"
    if (Test-Path $streamlitApp) {
        & $PythonExe -m streamlit run $streamlitApp --server.port 2501 --server.headless true
    } else {
        & $PythonExe -m http.server 2501 --directory apps/dashboard
    }
}

function Start-Worker {
    Write-Header "BACKGROUND SCHEDULER"
    Set-Location $ProjectRoot
    & $PythonExe -m apps.worker.scheduler
}

function Start-Full {
    Write-Header "CBB LINES - FULL DAILY RUN"
    Write-Host "  Date: $Date" -ForegroundColor White
    Write-Host "  Model: v2.0.0-phd (Bayesian + Conformal + RAPM + Injuries)" -ForegroundColor DarkGray
    Write-Host ""

    Stop-All -Silent

    Write-Step "[1/5] Ingesting data..."
    try { Invoke-Ingest -TargetDate $Date } catch { Write-Warn "Ingest issues: $_" }

    Write-Step "[2/5] Calculating ratings..."
    try { Invoke-Ratings } catch { Write-Warn "Ratings issues: $_" }

    Write-Step "[3/5] Generating predictions..."
    try { Invoke-Predict -TargetDate $Date } catch { Write-Warn "Prediction issues: $_" }

    Write-Step "[4/5] Fetching betting splits..."
    try { Invoke-Splits } catch { Write-Warn "Splits failed (non-critical): $_" }

    Write-Step "[5/5] Starting services..."
    Set-Location $ProjectRoot

    Start-Job -Name "API" -ScriptBlock {
        param($root, $python)
        Set-Location $root
        & $python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 2500
    } -ArgumentList $ProjectRoot, $PythonExe | Out-Null

    Start-Sleep -Seconds 2

    $streamlitApp = Join-Path $ProjectRoot "apps\dashboard\app.py"
    $useStreamlit = Test-Path $streamlitApp
    Start-Job -Name "Dashboard" -ScriptBlock {
        param($root, $python, $useStreamlit, $appPath)
        Set-Location $root
        if ($useStreamlit) {
            & $python -m streamlit run $appPath --server.port 2501 --server.headless true
        } else {
            & $python -m http.server 2501 --directory apps/dashboard
        }
    } -ArgumentList $ProjectRoot, $PythonExe, $useStreamlit, $streamlitApp | Out-Null

    Start-Sleep -Seconds 4

    Write-Host ""
    Write-Host "  +====================================================+" -ForegroundColor Green
    Write-Host "  |              SERVICES RUNNING                      |" -ForegroundColor Green
    Write-Host "  +====================================================+" -ForegroundColor Green

    foreach ($port in @(2500, 2501)) {
        $svc = if ($port -eq 2500) { "API Server" } else { "Dashboard" }
        $running = Test-PortOpen -Port $port
        $status = if ($running) { "RUNNING" } else { "starting..." }
        $color = if ($running) { "Green" } else { "Yellow" }
        Write-Host "  |  $svc".PadRight(25) -NoNewline -ForegroundColor White
        Write-Host $status.PadRight(12) -NoNewline -ForegroundColor $color
        Write-Host "http://localhost:$port".PadRight(20) -NoNewline -ForegroundColor Gray
        Write-Host " |" -ForegroundColor Green
    }

    Write-Host "  +====================================================+" -ForegroundColor Green
    Write-Host "  |  API Docs: http://localhost:2500/docs              |" -ForegroundColor Green
    Write-Host "  +====================================================+" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Press Ctrl+C to stop | Or run: start.ps1 stop" -ForegroundColor DarkGray
    Write-Host ""

    if (-not $NoBrowser) {
        Start-Sleep -Seconds 2
        Start-Process "http://localhost:2501"
    }

    try {
        while ($true) {
            $failed = Get-Job | Where-Object { $_.State -eq 'Failed' }
            if ($failed) {
                Write-Err "A service crashed"
                $failed | ForEach-Object { Receive-Job -Job $_ -ErrorAction SilentlyContinue }
                break
            }
            Start-Sleep -Seconds 5
        }
    } finally {
        Stop-All -Silent
    }
}

function Start-Quick {
    Write-Header "QUICK PREDICT"
    Stop-All -Silent
    Invoke-Predict -TargetDate $Date
    Set-Location $ProjectRoot
    $streamlitApp = Join-Path $ProjectRoot "apps\dashboard\app.py"
    $useStreamlit = Test-Path $streamlitApp
    Start-Job -Name "Dashboard" -ScriptBlock {
        param($root, $python, $useStreamlit, $appPath)
        Set-Location $root
        if ($useStreamlit) {
            & $python -m streamlit run $appPath --server.port 2501 --server.headless true
        } else {
            & $python -m http.server 2501 --directory apps/dashboard
        }
    } -ArgumentList $ProjectRoot, $PythonExe, $useStreamlit, $streamlitApp | Out-Null
    Start-Job -Name "API" -ScriptBlock {
        param($root, $python)
        Set-Location $root
        & $python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 2500
    } -ArgumentList $ProjectRoot, $PythonExe | Out-Null
    Start-Sleep -Seconds 4
    Write-OK "Dashboard: http://localhost:2501"
    Write-OK "API: http://localhost:2500"
    if (-not $NoBrowser) { Start-Process "http://localhost:2501" }
    try { while ($true) { Start-Sleep -Seconds 5 } }
    finally { Stop-All -Silent }
}

function Show-Status {
    Write-Header "SERVICE STATUS"
    foreach ($port in $Ports) {
        $svc = switch ($port) { 2500 { "API Server" } 2501 { "Dashboard" } 2502 { "Bracket API" } }
        if (Test-PortOpen -Port $port) {
            Write-OK "$svc running on port $port (http://localhost:$port)"
        } else {
            Write-Host "  [--] $svc NOT running on port $port" -ForegroundColor DarkGray
        }
    }
    $jobs = Get-Job -ErrorAction SilentlyContinue
    if ($jobs) {
        Write-Host ""
        Write-Host "  Background Jobs:" -ForegroundColor White
        $jobs | ForEach-Object {
            Write-Host "    $($_.Name): $($_.State)"
        }
    }
}

function Run-Tournament {
    Write-Header "TOURNAMENT"
    Set-Location $ProjectRoot
    & $PythonExe scripts/run_tournament_predictions.py
}

function Run-Bracket {
    Write-Header "RENDER BRACKET"
    Set-Location $ProjectRoot
    & $PythonExe scripts/render_bracket.py
    if (Test-Path "predictions/tournament_2026/bracket.html") {
        Start-Process "predictions/tournament_2026/bracket.html"
    }
}

# MAIN DISPATCH
switch ($Command) {
    "full"         { Start-Full }
    "quick"        { Start-Quick }
    "pipeline"     { Invoke-Pipeline -TargetDate $Date }
    "setup"        { Install-Dependencies }
    "api"          { Start-API }
    "dashboard"    { Start-Dashboard }
    "worker"       { Start-Worker }
    "ingest"       { Invoke-Ingest -TargetDate $Date }
    "ratings"      { Invoke-Ratings }
    "predict"      { Invoke-Predict -TargetDate $Date }
    "splits"       { Invoke-Splits }
    "backtest"     { Invoke-Backtest }
    "calibrate"    { Invoke-Calibrate }
    "report"       { Invoke-Report }
    "test"         { Invoke-Tests }
    "tournament"   { Run-Tournament }
    "bracket"      { Run-Bracket }
    "stop"         { Stop-All }
    "status"       { Show-Status }
    "help"         { Show-Help }
    default        { Start-Full }
}
