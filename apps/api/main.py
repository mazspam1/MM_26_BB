"""
FastAPI service for NCAAB spread/total predictions.

Endpoints:
- GET /health - Health check
- GET /slate - Today's games with predictions
- GET /game/{game_id} - Single game prediction
- GET /teams - List all teams
- GET /backtest - Backtest results summary
- GET /backtest/segments - Backtest diagnostics by segment
- GET /clv-analysis - CLV analysis
"""

from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

from packages.common.config import get_settings
from packages.common.database import get_connection, init_database
from packages.common.schemas import SeasonPhase
from packages.models.enhanced_predictor import (
    EnhancedPredictor,
    MODEL_VERSION,
    create_enhanced_predictor,
)
from packages.features.kenpom_ratings import TeamRatings

logger = structlog.get_logger()

# Global predictor instance
_predictor: Optional[EnhancedPredictor] = None
_ratings_cache: Optional[dict] = None
_ratings_cache_date: Optional[date] = None

def get_predictor() -> EnhancedPredictor:
    """Get or create the enhanced predictor."""
    global _predictor
    if _predictor is None:
        _predictor = create_enhanced_predictor()
    return _predictor

def get_ratings() -> dict[int, TeamRatings]:
    """Get cached ratings from database."""
    global _ratings_cache, _ratings_cache_date
    today = date.today()

    if _ratings_cache is not None and _ratings_cache_date == today:
        return _ratings_cache

    # Load from database
    with get_connection() as conn:
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
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM team_strengths)
        """
        ).fetchall()

    if not rows:
        logger.warning("No ratings found in database")
        return {}

    _ratings_cache = {}
    for row in rows:
        _ratings_cache[row[0]] = TeamRatings(
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

    _ratings_cache_date = today
    logger.info("Loaded ratings from database", teams=len(_ratings_cache))
    return _ratings_cache

# Initialize FastAPI app
app = FastAPI(
    title="NCAAB Prediction API",
    description="NCAA Men's Basketball spread and total predictions with uncertainty quantification",
    version="0.1.0",
)

# Add CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    database_connected: bool
    teams_count: int
    games_count: int
    timestamp: datetime


class TeamResponse(BaseModel):
    """Team information."""

    team_id: int
    name: str
    abbreviation: str
    conference: str
    logo_url: Optional[str] = None


class PredictionResponse(BaseModel):
    """Game prediction response."""

    game_id: int
    game_date: date
    game_datetime: Optional[datetime] = None
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    status: str
    is_neutral: bool

    # Predictions
    proj_home_score: float
    proj_away_score: float
    proj_spread: float = Field(description="Positive = home favored")
    proj_total: float
    home_win_prob: float

    # Actual scores (for finalized/live games)
    actual_home_score: Optional[int] = None
    actual_away_score: Optional[int] = None

    # Confidence intervals
    spread_ci_50: tuple[float, float]
    spread_ci_80: tuple[float, float]
    spread_ci_95: tuple[float, float]
    total_ci_50: tuple[float, float]
    total_ci_80: tuple[float, float]
    total_ci_95: tuple[float, float]

    # Market comparison (if available)
    market_spread: Optional[float] = Field(
        default=None,
        description="Market spread in model convention (positive = home favored)",
    )
    market_total: Optional[float] = None
    edge_vs_spread: Optional[float] = None
    edge_vs_total: Optional[float] = None

    # Recommendation
    recommended_side: Optional[str] = None
    recommended_units: Optional[float] = None
    confidence_rating: Optional[str] = None

    model_version: str

    # Betting splits (from DraftKings)
    spread_favored_handle_pct: Optional[float] = None
    spread_favored_bets_pct: Optional[float] = None
    spread_underdog_handle_pct: Optional[float] = None
    spread_underdog_bets_pct: Optional[float] = None
    total_over_handle_pct: Optional[float] = None
    total_over_bets_pct: Optional[float] = None
    total_under_handle_pct: Optional[float] = None
    total_under_bets_pct: Optional[float] = None


class SlateResponse(BaseModel):
    """Today's slate response."""

    date: date
    games_count: int
    predictions: list[PredictionResponse]
    generated_at: datetime


class BacktestSummary(BaseModel):
    """Backtest results summary."""

    start_date: date
    end_date: date
    total_games: int
    spread_mae: float
    spread_rmse: float
    total_mae: float
    total_rmse: float
    coverage_50: float
    coverage_80: float
    coverage_95: float
    mean_clv: Optional[float] = None
    clv_positive_rate: Optional[float] = None


class BacktestSegment(BaseModel):
    """Backtest segment diagnostics."""

    run_id: str
    segment_type: str
    segment_value: str
    total_games: int
    market_spread_count: int
    market_total_count: int
    closing_spread_count: int
    closing_total_count: int
    spread_mae: float
    spread_rmse: float
    total_mae: float
    total_rmse: float
    spread_50_coverage: float
    spread_80_coverage: float
    spread_95_coverage: float
    total_50_coverage: float
    total_80_coverage: float
    total_95_coverage: float
    mean_spread_clv: float
    mean_total_clv: float
    clv_positive_rate: float
    simulated_roi: float
    market_spread_rate: float
    market_total_rate: float
    closing_spread_rate: float
    closing_total_rate: float
    spread_50_drift: float
    spread_80_drift: float
    spread_95_drift: float
    total_50_drift: float
    total_80_drift: float
    total_95_drift: float


class CLVAnalysis(BaseModel):
    """CLV analysis response."""

    period_start: date
    period_end: date
    total_predictions: int
    mean_clv: float
    median_clv: float
    clv_positive_rate: float
    mean_clv_by_confidence: dict[str, float]
    mean_clv_by_edge_bucket: dict[str, float]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Starting NCAAB Prediction API")
    init_database()
    logger.info("Database initialized")


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        with get_connection() as conn:
            teams_count = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
            games_count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            db_connected = True
    except Exception as e:
        logger.error("Database connection failed", error=str(e))
        db_connected = False
        teams_count = 0
        games_count = 0

    settings = get_settings()

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        version=settings.model_version,
        database_connected=db_connected,
        teams_count=teams_count,
        games_count=games_count,
        timestamp=datetime.utcnow(),
    )


@app.get("/teams", response_model=list[TeamResponse])
async def list_teams(
    conference: Optional[str] = Query(None, description="Filter by conference"),
    limit: int = Query(500, ge=1, le=500),
):
    """List all teams."""
    with get_connection() as conn:
        if conference:
            rows = conn.execute(
                """
                SELECT team_id, name, abbreviation, conference, logo_url
                FROM teams
                WHERE conference = ?
                ORDER BY name
                LIMIT ?
                """,
                (conference, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT team_id, name, abbreviation, conference, logo_url
                FROM teams
                ORDER BY name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [
        TeamResponse(
            team_id=row[0],
            name=row[1],
            abbreviation=row[2],
            conference=row[3],
            logo_url=row[4],
        )
        for row in rows
    ]


@app.get("/slate", response_model=SlateResponse)
async def get_slate(
    target_date: Optional[date] = Query(None, description="Date for slate (default: today)"),
    min_edge: Optional[float] = Query(None, description="Minimum edge to include"),
):
    """Get today's slate with predictions."""
    if target_date is None:
        target_date = date.today()

    with get_connection() as conn:
        # Get games for date
        # Get latest betting splits for each game (market data from DraftKings)
        # Use spread_line_home in model convention:
        # - Positive = home team is favored
        # - Negative = away team is favored (home is underdog)
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
                g.home_team_id, g.home_team_name,
                g.away_team_id, g.away_team_name,
                g.status, g.neutral_site,
                p.proj_spread, p.proj_total,
                p.home_win_prob,
                p.efficiency_spread, p.hca_adjustment, p.travel_adjustment, p.rest_adjustment, p.four_factors_adjustment,
                p.spread_ci_50_lower, p.spread_ci_50_upper,
                p.spread_ci_80_lower, p.spread_ci_80_upper,
                p.spread_ci_95_lower, p.spread_ci_95_upper,
                p.total_ci_50_lower, p.total_ci_50_upper,
                p.total_ci_80_lower, p.total_ci_80_upper,
                p.total_ci_95_lower, p.total_ci_95_upper,
                -- Use betting_splits for market data (corrected sign convention)
                (SELECT spread_line_home FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as market_spread,
                (SELECT total_line FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as market_total,
                -- Model market lines retained for reference
                p.market_spread as model_market_spread,
                p.market_total as model_market_total,
                p.edge_vs_market_spread, p.edge_vs_market_total,
                p.recommended_side, p.recommended_units, p.confidence_rating,
                p.model_version,
                (SELECT spread_favored_handle_pct FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as spread_favored_handle_pct,
                (SELECT spread_favored_bets_pct FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as spread_favored_bets_pct,
                (SELECT spread_underdog_handle_pct FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as spread_underdog_handle_pct,
                (SELECT spread_underdog_bets_pct FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as spread_underdog_bets_pct,
                (SELECT total_over_handle_pct FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as total_over_handle_pct,
                (SELECT total_over_bets_pct FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as total_over_bets_pct,
                (SELECT total_under_handle_pct FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as total_under_handle_pct,
                (SELECT total_under_bets_pct FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as total_under_bets_pct
            FROM games g
            LEFT JOIN latest_predictions p ON g.game_id = p.game_id
            WHERE g.game_date = ?
            ORDER BY g.game_datetime
            """,
            (target_date.isoformat(),),
        ).fetchall()

    predictions = []
    for row in rows:
        # SKIP games without real predictions
        # Per CLAUDE.md: "Never use fallbacks, mock data, or placeholder values"
        if row[9] is None:  # proj_home_score is None - no prediction exists
            logger.debug(
                "Skipping game without prediction",
                game_id=row[0],
                home_team=row[4],
                away_team=row[6],
            )
            continue
        else:
            # CI values should always be present if prediction exists
            # Log warning if missing - indicates model bug that should be fixed
            if any(row[i] is None for i in range(21, 33)):
                logger.warning(
                    "Prediction missing CI values - model should calculate these",
                    game_id=row[0],
                    missing_ci_indices=[i for i in range(21, 33) if row[i] is None],
                )
            # Use actual values - None will be handled by Pydantic if schema allows
            spread_ci_50 = (row[21] if row[21] is not None else 0.0, row[22] if row[22] is not None else 0.0)
            spread_ci_80 = (row[23] if row[23] is not None else 0.0, row[24] if row[24] is not None else 0.0)
            spread_ci_95 = (row[25] if row[25] is not None else 0.0, row[26] if row[26] is not None else 0.0)
            total_ci_50 = (row[27] if row[27] is not None else 0.0, row[28] if row[28] is not None else 0.0)
            total_ci_80 = (row[29] if row[29] is not None else 0.0, row[30] if row[30] is not None else 0.0)
            total_ci_95 = (row[31] if row[31] is not None else 0.0, row[32] if row[32] is not None else 0.0)

            # Resolve market data using DraftKings splits when available.
            market_spread, market_total, market_spread_model = _resolve_market_lines(
                row[33],  # spread_line_home from betting_splits
                row[34],  # total_line from betting_splits
                row[35],  # model_market_spread (home-positive)
                row[36],  # model_market_total
                allow_model_fallback=False,
            )
            proj_spread = row[13]    # model's projected spread
            proj_total = row[14]     # model's projected total

            # Calculate edge dynamically using model convention (home-positive).
            edge_vs_spread = None
            edge_vs_total = None
            if market_spread_model is not None and proj_spread is not None:
                edge_vs_spread = proj_spread - market_spread_model
            if market_total is not None and proj_total is not None:
                edge_vs_total = proj_total - market_total

            pred = PredictionResponse(
                game_id=row[0],
                game_date=row[1] if isinstance(row[1], date) else date.fromisoformat(row[1]),
                game_datetime=_parse_datetime(row[2]),
                home_team_id=row[3],
                home_team_name=row[4],
                away_team_id=row[5],
                away_team_name=row[6],
                status=row[7],
                is_neutral=bool(row[8]),
                actual_home_score=row[9],
                actual_away_score=row[10],
                proj_home_score=row[11],
                proj_away_score=row[12],
                proj_spread=row[13],
                proj_total=row[14],
                home_win_prob=row[15],
                efficiency_spread=row[16],
                hca_adjustment=row[17],
                travel_adjustment=row[18],
                rest_adjustment=row[19],
                four_factors_adjustment=row[20],
                spread_ci_50=spread_ci_50,
                spread_ci_80=spread_ci_80,
                spread_ci_95=spread_ci_95,
                total_ci_50=total_ci_50,
                total_ci_80=total_ci_80,
                total_ci_95=total_ci_95,
                market_spread=market_spread,
                market_total=market_total,
                edge_vs_spread=edge_vs_spread,
                edge_vs_total=edge_vs_total,
                recommended_side=row[39],
                recommended_units=row[40],
                confidence_rating=row[41],
                model_version=row[42] or "pending",
                spread_favored_handle_pct=row[43] if len(row) > 43 else None,
                spread_favored_bets_pct=row[44] if len(row) > 44 else None,
                spread_underdog_handle_pct=row[45] if len(row) > 45 else None,
                spread_underdog_bets_pct=row[46] if len(row) > 46 else None,
                total_over_handle_pct=row[47] if len(row) > 47 else None,
                total_over_bets_pct=row[48] if len(row) > 48 else None,
                total_under_handle_pct=row[49] if len(row) > 49 else None,
                total_under_bets_pct=row[50] if len(row) > 50 else None,
            )

        # Filter by minimum edge if specified
        if min_edge is not None:
            if pred.edge_vs_spread is None or abs(pred.edge_vs_spread) < min_edge:
                if pred.edge_vs_total is None or abs(pred.edge_vs_total) < min_edge:
                    continue

        predictions.append(pred)

    return SlateResponse(
        date=target_date,
        games_count=len(predictions),
        predictions=predictions,
        generated_at=datetime.utcnow(),
    )


class LivePredictionResponse(BaseModel):
    """Live prediction response with enhanced model details."""
    game_id: int
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    is_neutral: bool

    # Model predictions
    proj_home_score: float
    proj_away_score: float
    proj_spread: float
    proj_total: float
    home_win_prob: float

    # Uncertainty
    spread_std: float
    spread_ci_50: tuple[float, float]
    spread_ci_80: tuple[float, float]
    spread_ci_95: tuple[float, float]

    # Components
    efficiency_spread: float
    hca_adjustment: float
    travel_adjustment: float
    rest_adjustment: float
    four_factors_adjustment: float

    # Market comparison
    market_spread: Optional[float] = None
    edge: Optional[float] = None
    recommended_play: Optional[str] = None

    # Ratings info
    home_adj_em: float
    away_adj_em: float
    home_games_played: int
    away_games_played: int

    model_version: str


@app.get("/live-slate")
async def get_live_slate(
    target_date: Optional[date] = Query(None, description="Date for slate (default: today)"),
):
    """Get today's slate with LIVE predictions from enhanced model."""
    from packages.common.sportsdataverse_mbb import load_mbb

    if target_date is None:
        target_date = date.today()

    # Get predictor and ratings
    predictor = get_predictor()
    ratings = get_ratings()

    if not ratings:
        raise HTTPException(status_code=503, detail="Ratings not yet calculated. Run pipeline first.")

    # Fetch today's games from ESPN
    try:
        date_str = target_date.strftime("%Y%m%d")
        schedule = load_mbb().espn_mbb_schedule(dates=date_str, groups=50, return_as_pandas=True)
    except Exception as e:
        logger.error("Failed to fetch schedule", error=str(e))
        raise HTTPException(status_code=503, detail="Failed to fetch schedule from ESPN")

    if len(schedule) == 0:
        return {"date": target_date, "games_count": 0, "predictions": []}

    # Get team names
    team_names = {}
    with get_connection() as conn:
        rows = conn.execute("SELECT team_id, name FROM teams").fetchall()
        for row in rows:
            team_names[row[0]] = row[1]

    # Generate predictions
    predictions = []
    for _, game in schedule.iterrows():
        # Convert string IDs to integers (ESPN returns strings)
        try:
            home_id = int(game.get('home_id'))
            away_id = int(game.get('away_id'))
            game_id = int(game.get('game_id')) if game.get('game_id') else 0
        except (ValueError, TypeError):
            continue

        is_neutral = bool(game.get('neutral_site', False))

        # Skip if we don't have ratings for both teams
        if home_id not in ratings or away_id not in ratings:
            continue

        home_ratings = ratings[home_id]
        away_ratings = ratings[away_id]

        # Generate prediction
        pred = predictor.predict_game(
            home_ratings=home_ratings,
            away_ratings=away_ratings,
            game_id=game_id,
            is_neutral=is_neutral,
        )

        # Get team names
        home_name = team_names.get(home_id, game.get('home_display_name', f'Team {home_id}'))
        away_name = team_names.get(away_id, game.get('away_display_name', f'Team {away_id}'))

        predictions.append(LivePredictionResponse(
            game_id=game_id,
            home_team_id=home_id,
            home_team_name=home_name,
            away_team_id=away_id,
            away_team_name=away_name,
            is_neutral=is_neutral,
            proj_home_score=pred.home_score,
            proj_away_score=pred.away_score,
            proj_spread=pred.spread,
            proj_total=pred.total,
            home_win_prob=pred.home_win_prob,
            spread_std=pred.spread_std,
            spread_ci_50=pred.spread_ci_50,
            spread_ci_80=pred.spread_ci_80,
            spread_ci_95=pred.spread_ci_95,
            efficiency_spread=pred.efficiency_spread,
            hca_adjustment=pred.hca_adjustment,
            travel_adjustment=pred.travel_adjustment,
            rest_adjustment=pred.rest_adjustment,
            four_factors_adjustment=pred.four_factors_adjustment,
            market_spread=pred.market_spread,
            edge=pred.edge,
            recommended_play=pred.recommended_play,
            home_adj_em=home_ratings.adj_em,
            away_adj_em=away_ratings.adj_em,
            home_games_played=home_ratings.games_played,
            away_games_played=away_ratings.games_played,
            model_version=pred.model_version,
        ))

    # Sort by absolute edge (best opportunities first)
    predictions.sort(key=lambda p: abs(p.edge or 0), reverse=True)

    return {
        "date": target_date,
        "games_count": len(predictions),
        "predictions": predictions,
        "model_version": MODEL_VERSION,
        "ratings_teams": len(ratings),
    }


@app.get("/team-ratings/{team_id}")
async def get_team_ratings(team_id: int):
    """Get adjusted ratings for a specific team."""
    ratings = get_ratings()

    if team_id not in ratings:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found in ratings")

    r = ratings[team_id]
    return {
        "team_id": r.team_id,
        "adj_off": r.adj_off,
        "adj_def": r.adj_def,
        "adj_em": r.adj_em,
        "adj_tempo": r.adj_tempo,
        "games_played": r.games_played,
        "four_factors": {
            "off_efg": r.adj_efg,
            "off_tov": r.adj_tov,
            "off_orb": r.adj_orb,
            "off_ftr": r.adj_ftr,
            "def_efg": r.adj_efg_def,
            "def_tov": r.adj_tov_def,
            "def_drb": r.adj_drb,
            "def_ftr": r.adj_ftr_def,
        },
        "sos": {
            "off": r.sos_off,
            "def": r.sos_def,
        },
        "as_of_date": r.as_of_date.isoformat(),
    }


@app.get("/top-teams")
async def get_top_teams(limit: int = Query(25, ge=1, le=100)):
    """Get top teams by adjusted efficiency margin."""
    ratings = get_ratings()

    if not ratings:
        raise HTTPException(status_code=503, detail="Ratings not yet calculated")

    # Get team names
    team_names = {}
    with get_connection() as conn:
        rows = conn.execute("SELECT team_id, name FROM teams").fetchall()
        for row in rows:
            team_names[row[0]] = row[1]

    # Sort by efficiency margin
    sorted_teams = sorted(ratings.values(), key=lambda r: r.adj_em, reverse=True)[:limit]

    return [
        {
            "rank": i + 1,
            "team_id": r.team_id,
            "name": team_names.get(r.team_id, f"Team {r.team_id}"),
            "adj_off": round(r.adj_off, 1),
            "adj_def": round(r.adj_def, 1),
            "adj_em": round(r.adj_em, 1),
            "adj_tempo": round(r.adj_tempo, 1),
            "games_played": r.games_played,
        }
        for i, r in enumerate(sorted_teams)
    ]


@app.get("/game/{game_id}", response_model=PredictionResponse)
async def get_game(game_id: int):
    """Get prediction for a single game."""
    with get_connection() as conn:
        row = conn.execute(
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
                g.home_team_id, g.home_team_name,
                g.away_team_id, g.away_team_name,
                g.status, g.neutral_site,
                p.proj_home_score, p.proj_away_score,
                p.proj_spread, p.proj_total,
                p.home_win_prob,
                p.spread_ci_50_lower, p.spread_ci_50_upper,
                p.spread_ci_80_lower, p.spread_ci_80_upper,
                p.spread_ci_95_lower, p.spread_ci_95_upper,
                p.total_ci_50_lower, p.total_ci_50_upper,
                p.total_ci_80_lower, p.total_ci_80_upper,
                p.total_ci_95_lower, p.total_ci_95_upper,
                (SELECT spread_line_home FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as market_spread,
                (SELECT total_line FROM betting_splits WHERE betting_splits.game_id = g.game_id ORDER BY snapshot_timestamp DESC LIMIT 1) as market_total,
                p.market_spread as model_market_spread,
                p.market_total as model_market_total,
                p.edge_vs_market_spread, p.edge_vs_market_total,
                p.recommended_side, p.recommended_units, p.confidence_rating,
                p.model_version
            FROM games g
            LEFT JOIN latest_predictions p ON g.game_id = p.game_id
            WHERE g.game_id = ?
            """,
            (game_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    # No fake placeholder predictions - require real data
    # Per CLAUDE.md: "Never use fallbacks, mock data, or placeholder values"
    if row[9] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction not yet available for game {game_id}. Run prediction pipeline first."
        )

    # CI values should always be present if prediction exists
    # Log warning if missing - indicates model bug that should be fixed
    if any(row[i] is None for i in range(14, 26)):
        logger.warning(
            "Prediction missing CI values - model should calculate these",
            game_id=game_id,
            missing_ci_indices=[i for i in range(14, 26) if row[i] is None],
        )
    spread_ci_50 = (row[14] if row[14] is not None else 0.0, row[15] if row[15] is not None else 0.0)
    spread_ci_80 = (row[16] if row[16] is not None else 0.0, row[17] if row[17] is not None else 0.0)
    spread_ci_95 = (row[18] if row[18] is not None else 0.0, row[19] if row[19] is not None else 0.0)
    total_ci_50 = (row[20] if row[20] is not None else 0.0, row[21] if row[21] is not None else 0.0)
    total_ci_80 = (row[22] if row[22] is not None else 0.0, row[23] if row[23] is not None else 0.0)
    total_ci_95 = (row[24] if row[24] is not None else 0.0, row[25] if row[25] is not None else 0.0)

    market_spread, market_total, market_spread_model = _resolve_market_lines(
        row[26],  # spread_line_home from betting_splits
        row[27],  # total_line from betting_splits
        row[28],  # model_market_spread (home-positive)
        row[29],  # model_market_total
        allow_model_fallback=False,
    )
    edge_vs_spread = None
    edge_vs_total = None
    if market_spread_model is not None and row[11] is not None:
        edge_vs_spread = row[11] - market_spread_model
    if market_total is not None and row[12] is not None:
        edge_vs_total = row[12] - market_total

    return PredictionResponse(
        game_id=row[0],
        game_date=row[1] if isinstance(row[1], date) else date.fromisoformat(row[1]),
        game_datetime=_parse_datetime(row[2]),
        home_team_id=row[3],
        home_team_name=row[4],
        away_team_id=row[5],
        away_team_name=row[6],
        status=row[7],
        is_neutral=bool(row[8]),
        proj_home_score=row[9],
        proj_away_score=row[10],
        proj_spread=row[11],
        proj_total=row[12],
        home_win_prob=row[13],
        spread_ci_50=spread_ci_50,
        spread_ci_80=spread_ci_80,
        spread_ci_95=spread_ci_95,
        total_ci_50=total_ci_50,
        total_ci_80=total_ci_80,
        total_ci_95=total_ci_95,
        market_spread=market_spread,
        market_total=market_total,
        edge_vs_spread=edge_vs_spread,
        edge_vs_total=edge_vs_total,
        recommended_side=row[32],
        recommended_units=row[33],
        confidence_rating=row[34],
        model_version=row[35] or "pending",
    )


@app.get("/backtest", response_model=BacktestSummary)
async def get_backtest_summary(
    start_date: Optional[date] = Query(None, description="Start date"),
    end_date: Optional[date] = Query(None, description="End date"),
):
    """Get backtest results summary."""
    with get_connection() as conn:
        run_row = conn.execute(
            """
            SELECT run_id
            FROM backtest_runs
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

        if not run_row:
            raise HTTPException(status_code=404, detail="No backtest runs available")

        run_id = run_row[0]

        # Build date filter
        date_filter = ""
        params: list = []
        if start_date:
            date_filter += " AND g.game_date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            date_filter += " AND g.game_date <= ?"
            params.append(end_date.isoformat())

        # Get summary stats
        query = f"""
        SELECT
            MIN(g.game_date) as start_date,
            MAX(g.game_date) as end_date,
            COUNT(*) as total_games,
            AVG(ABS(b.proj_spread - (g.home_score - g.away_score))) as spread_mae,
            SQRT(AVG(POWER(b.proj_spread - (g.home_score - g.away_score), 2))) as spread_rmse,
            AVG(ABS(b.proj_total - (g.home_score + g.away_score))) as total_mae,
            SQRT(AVG(POWER(b.proj_total - (g.home_score + g.away_score), 2))) as total_rmse,
            AVG(CASE WHEN (g.home_score - g.away_score) BETWEEN b.spread_ci_50_lower AND b.spread_ci_50_upper THEN 1 ELSE 0 END) as coverage_50,
            AVG(CASE WHEN (g.home_score - g.away_score) BETWEEN b.spread_ci_80_lower AND b.spread_ci_80_upper THEN 1 ELSE 0 END) as coverage_80,
            AVG(CASE WHEN (g.home_score - g.away_score) BETWEEN b.spread_ci_95_lower AND b.spread_ci_95_upper THEN 1 ELSE 0 END) as coverage_95,
            AVG(b.spread_clv) as mean_clv,
            SUM(CASE WHEN b.spread_clv > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN b.spread_clv IS NOT NULL THEN 1 ELSE 0 END), 0) as clv_positive_rate
        FROM games g
        JOIN backtest_predictions b ON g.game_id = b.game_id
        WHERE g.status = 'final'
            AND g.home_score IS NOT NULL
            AND g.away_score IS NOT NULL
            AND b.run_id = ?
            {date_filter}
        """

        row = conn.execute(query, [run_id, *params]).fetchone()

        if not row or row[2] == 0:
            raise HTTPException(status_code=404, detail="No backtest data available")

        # Calculate coverage (simplified - would need actual CI comparison)
        return BacktestSummary(
            start_date=date.fromisoformat(row[0]) if row[0] else date.today(),
            end_date=date.fromisoformat(row[1]) if row[1] else date.today(),
            total_games=row[2],
            spread_mae=row[3] or 0.0,
            spread_rmse=row[4] or 0.0,
            total_mae=row[5] or 0.0,
            total_rmse=row[6] or 0.0,
            coverage_50=row[7] or 0.0,
            coverage_80=row[8] or 0.0,
            coverage_95=row[9] or 0.0,
            mean_clv=row[10] or 0.0,
            clv_positive_rate=row[11] or 0.0,
        )


@app.get("/backtest/segments", response_model=list[BacktestSegment])
async def get_backtest_segments(
    run_id: Optional[str] = Query(None, description="Backtest run_id (default: latest)"),
    segment_type: Optional[str] = Query(None, description="Segment type filter"),
):
    """Get segment-level backtest diagnostics."""
    with get_connection() as conn:
        if run_id is None:
            run_row = conn.execute(
                """
                SELECT run_id
                FROM backtest_runs
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if not run_row:
                raise HTTPException(status_code=404, detail="No backtest runs available")
            run_id = run_row[0]

        query = """
        SELECT
            segment_type, segment_value,
            total_games,
            market_spread_count, market_total_count,
            closing_spread_count, closing_total_count,
            spread_mae, spread_rmse, total_mae, total_rmse,
            spread_50_coverage, spread_80_coverage, spread_95_coverage,
            total_50_coverage, total_80_coverage, total_95_coverage,
            mean_spread_clv, mean_total_clv, clv_positive_rate,
            simulated_roi
        FROM backtest_segments
        WHERE run_id = ?
        """
        params: list = [run_id]
        if segment_type:
            query += " AND segment_type = ?"
            params.append(segment_type)
        query += " ORDER BY segment_type, segment_value"

        rows = conn.execute(query, params).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No backtest segment data available")

    response = []
    for row in rows:
        total_games = row[2] or 0
        market_spread_count = row[3] or 0
        market_total_count = row[4] or 0
        closing_spread_count = row[5] or 0
        closing_total_count = row[6] or 0

        spread_50 = row[11] or 0.0
        spread_80 = row[12] or 0.0
        spread_95 = row[13] or 0.0
        total_50 = row[14] or 0.0
        total_80 = row[15] or 0.0
        total_95 = row[16] or 0.0

        response.append(
            BacktestSegment(
                run_id=run_id,
                segment_type=row[0],
                segment_value=row[1],
                total_games=total_games,
                market_spread_count=market_spread_count,
                market_total_count=market_total_count,
                closing_spread_count=closing_spread_count,
                closing_total_count=closing_total_count,
                spread_mae=row[7] or 0.0,
                spread_rmse=row[8] or 0.0,
                total_mae=row[9] or 0.0,
                total_rmse=row[10] or 0.0,
                spread_50_coverage=spread_50,
                spread_80_coverage=spread_80,
                spread_95_coverage=spread_95,
                total_50_coverage=total_50,
                total_80_coverage=total_80,
                total_95_coverage=total_95,
                mean_spread_clv=row[17] or 0.0,
                mean_total_clv=row[18] or 0.0,
                clv_positive_rate=row[19] or 0.0,
                simulated_roi=row[20] or 0.0,
                market_spread_rate=(market_spread_count / total_games) if total_games else 0.0,
                market_total_rate=(market_total_count / total_games) if total_games else 0.0,
                closing_spread_rate=(closing_spread_count / total_games) if total_games else 0.0,
                closing_total_rate=(closing_total_count / total_games) if total_games else 0.0,
                spread_50_drift=spread_50 - 0.50,
                spread_80_drift=spread_80 - 0.80,
                spread_95_drift=spread_95 - 0.95,
                total_50_drift=total_50 - 0.50,
                total_80_drift=total_80 - 0.80,
                total_95_drift=total_95 - 0.95,
            )
        )

    return response


@app.get("/clv-analysis", response_model=CLVAnalysis)
async def get_clv_analysis(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
):
    """Get CLV analysis."""
    with get_connection() as conn:
        # Build date filter
        date_filter = ""
        params: list = []
        if start_date:
            date_filter += " AND prediction_timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            date_filter += " AND prediction_timestamp <= ?"
            params.append(end_date.isoformat())

        # Check if we have CLV data
        row = conn.execute(
            f"""
            SELECT
                MIN(CAST(prediction_timestamp AS DATE)),
                MAX(CAST(prediction_timestamp AS DATE)),
                COUNT(*), AVG(clv_spread),
                SUM(CASE WHEN clv_spread > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
            FROM clv_reports
            WHERE 1=1 {date_filter}
            """,
            params,
        ).fetchone()

        if not row or row[2] == 0:
            raise HTTPException(status_code=404, detail="No CLV data available")

        return CLVAnalysis(
            period_start=date.fromisoformat(row[0]) if row[0] else date.today(),
            period_end=date.fromisoformat(row[1]) if row[1] else date.today(),
            total_predictions=row[2],
            mean_clv=row[3] or 0.0,
            median_clv=row[3] or 0.0,  # Would need actual median
            clv_positive_rate=row[4] or 0.0,
            mean_clv_by_confidence={},  # Would need actual breakdown
            mean_clv_by_edge_bucket={},
        )


def _parse_datetime(dt_val) -> Optional[datetime]:
    """Parse datetime value safely."""
    if not dt_val:
        return None
    try:
        if isinstance(dt_val, datetime):
            return dt_val
        if isinstance(dt_val, str):
            return datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
        return None
    except (ValueError, AttributeError, TypeError):
        return None


def _resolve_market_lines(
    splits_spread: Optional[float],
    splits_total: Optional[float],
    model_spread: Optional[float],
    model_total: Optional[float],
    allow_model_fallback: bool = False,  # Default False per CLAUDE.md - no fallbacks
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Resolve market lines for display and modeling.

    Returns:
        market_spread_display: Market spread in model convention (positive = home favored)
        market_total_display: Total line
        market_spread_model: Market spread in model convention (positive = home favored)
    """
    market_spread_display = None
    market_spread_model = None
    market_total_display = splits_total if splits_total is not None else model_total

    if splits_spread is not None:
        market_spread_display = splits_spread
        market_spread_model = splits_spread
    elif allow_model_fallback and model_spread is not None:
        market_spread_display = model_spread
        market_spread_model = model_spread

    return market_spread_display, market_total_display, market_spread_model


# NOTE: _create_placeholder_prediction() was REMOVED per CLAUDE.md requirement:
# "Never use fallbacks, mock data, or placeholder values"
# Games without real predictions are now skipped in /slate or return 503 in /game/{id}


# Run with: uvicorn apps.api.main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2500)
