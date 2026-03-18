"""
Pydantic data contracts for the CBB Lines system.

All data flowing through the system must conform to these schemas.
Strict validation ensures data integrity and prevents silent failures.
"""

from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class ConferenceType(str, Enum):
    """NCAA Division I conferences."""

    ACC = "ACC"
    BIG_12 = "Big 12"
    BIG_EAST = "Big East"
    BIG_TEN = "Big Ten"
    PAC_12 = "Pac-12"
    SEC = "SEC"
    AAC = "AAC"
    ATLANTIC_10 = "Atlantic 10"
    MOUNTAIN_WEST = "Mountain West"
    WCC = "WCC"
    COLONIAL = "Colonial"
    HORIZON = "Horizon"
    IVY = "Ivy"
    MAAC = "MAAC"
    MAC = "MAC"
    MEAC = "MEAC"
    MVC = "MVC"
    NORTHEAST = "Northeast"
    OVC = "OVC"
    PATRIOT = "Patriot"
    SOUTHERN = "Southern"
    SOUTHLAND = "Southland"
    SUMMIT = "Summit"
    SUN_BELT = "Sun Belt"
    SWAC = "SWAC"
    WAC = "WAC"
    BIG_SKY = "Big Sky"
    BIG_SOUTH = "Big South"
    BIG_WEST = "Big West"
    CAA = "CAA"
    CUSA = "C-USA"
    ASUN = "ASUN"
    OTHER = "Other"


class VenueType(str, Enum):
    """Game venue classification."""

    HOME = "home"
    AWAY = "away"
    NEUTRAL = "neutral"


class SeasonPhase(str, Enum):
    """Season regime for model adjustments."""

    EARLY = "early"  # First 2 weeks
    NON_CONFERENCE = "non_conference"
    CONFERENCE = "conference"
    TOURNAMENT = "tournament"


# =============================================================================
# CORE ENTITIES
# =============================================================================


class Team(BaseModel):
    """Team entity with identifiers and metadata."""

    model_config = ConfigDict(strict=True, frozen=True)

    team_id: int = Field(..., description="ESPN team ID (primary key)")
    name: str = Field(..., min_length=1, max_length=100)
    abbreviation: str = Field(..., min_length=2, max_length=10)
    conference: str = Field(..., description="Conference name")
    logo_url: Optional[str] = None
    color: Optional[str] = None
    alternate_color: Optional[str] = None

    @field_validator("team_id")
    @classmethod
    def validate_positive_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("team_id must be positive")
        return v


class TeamStrength(BaseModel):
    """Time-varying team strength parameters (KenPom-style)."""

    model_config = ConfigDict(strict=True)

    team_id: int
    as_of_date: date
    season: int = Field(..., ge=2002, le=2030)

    # Adjusted efficiencies (points per 100 possessions)
    adj_offensive_efficiency: float = Field(..., ge=60.0, le=150.0)
    adj_defensive_efficiency: float = Field(..., ge=60.0, le=150.0)
    adj_tempo: float = Field(..., ge=50.0, le=90.0, description="Possessions per 40 minutes")

    # Four Factors (offensive)
    off_efg_pct: Optional[float] = Field(None, ge=0.20, le=0.75)
    off_to_pct: Optional[float] = Field(None, ge=0.05, le=0.35)
    off_orb_pct: Optional[float] = Field(None, ge=0.10, le=0.50)
    off_ftr: Optional[float] = Field(None, ge=0.10, le=0.70, description="FTA/FGA ratio")

    # Four Factors (defensive)
    def_efg_pct: Optional[float] = Field(None, ge=0.20, le=0.75)
    def_to_pct: Optional[float] = Field(None, ge=0.05, le=0.35)
    def_drb_pct: Optional[float] = Field(None, ge=0.50, le=0.90)
    def_ftr: Optional[float] = Field(None, ge=0.10, le=0.70)

    # Uncertainty
    games_played: int = Field(..., ge=0)
    strength_ci_lower: Optional[float] = None
    strength_ci_upper: Optional[float] = None

    @computed_field
    @property
    def adj_efficiency_margin(self) -> float:
        """Net rating (offense - defense), higher is better."""
        return self.adj_offensive_efficiency - self.adj_defensive_efficiency


class Game(BaseModel):
    """Game entity with full context."""

    model_config = ConfigDict(strict=True)

    game_id: int = Field(..., description="ESPN game ID")
    season: int = Field(..., ge=2002, le=2030)
    game_date: date
    game_datetime: Optional[datetime] = None

    home_team_id: int
    away_team_id: int
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None

    venue_type: VenueType = VenueType.HOME
    neutral_site: bool = False
    venue_name: Optional[str] = None

    # Actual results (None if game not yet played)
    home_score: Optional[int] = Field(None, ge=0, le=250)
    away_score: Optional[int] = Field(None, ge=0, le=250)
    status: str = Field(default="scheduled", description="scheduled, in_progress, final")

    # Context
    conference_game: bool = False
    season_phase: SeasonPhase = SeasonPhase.NON_CONFERENCE

    # Possession data (from box score, filled after game)
    estimated_possessions: Optional[float] = Field(None, ge=40.0, le=120.0)

    @computed_field
    @property
    def actual_spread(self) -> Optional[float]:
        """Home - Away spread (positive = home won by more)."""
        if self.home_score is not None and self.away_score is not None:
            return float(self.home_score - self.away_score)
        return None

    @computed_field
    @property
    def actual_total(self) -> Optional[int]:
        """Combined score."""
        if self.home_score is not None and self.away_score is not None:
            return self.home_score + self.away_score
        return None

    @computed_field
    @property
    def is_completed(self) -> bool:
        """Whether the game has finished."""
        return self.status == "final" and self.home_score is not None


class BoxScore(BaseModel):
    """Team box score for a game (needed for Four Factors calculation)."""

    model_config = ConfigDict(strict=True)

    game_id: int
    team_id: int
    is_home: bool

    # Basic stats
    field_goals_made: int = Field(..., ge=0)
    field_goals_attempted: int = Field(..., ge=0)
    three_pointers_made: int = Field(..., ge=0)
    three_pointers_attempted: int = Field(..., ge=0)
    free_throws_made: int = Field(..., ge=0)
    free_throws_attempted: int = Field(..., ge=0)

    # Rebounds
    offensive_rebounds: int = Field(..., ge=0)
    defensive_rebounds: int = Field(..., ge=0)

    # Other
    turnovers: int = Field(..., ge=0)
    assists: int = Field(..., ge=0)
    steals: int = Field(..., ge=0)
    blocks: int = Field(..., ge=0)
    personal_fouls: int = Field(..., ge=0)

    # Final
    points: int = Field(..., ge=0, le=250)

    @computed_field
    @property
    def total_rebounds(self) -> int:
        """Total rebounds."""
        return self.offensive_rebounds + self.defensive_rebounds

    @computed_field
    @property
    def possessions(self) -> float:
        """Dean Oliver possession estimate."""
        return (
            self.field_goals_attempted
            - self.offensive_rebounds
            + self.turnovers
            + 0.475 * self.free_throws_attempted
        )

    @computed_field
    @property
    def efg_pct(self) -> float:
        """Effective field goal percentage."""
        if self.field_goals_attempted == 0:
            return 0.0
        return (
            self.field_goals_made + 0.5 * self.three_pointers_made
        ) / self.field_goals_attempted

    @computed_field
    @property
    def to_pct(self) -> float:
        """Turnover percentage (turnovers per possession)."""
        poss = self.possessions
        return self.turnovers / poss if poss > 0 else 0.0

    @computed_field
    @property
    def orb_pct(self) -> float:
        """Offensive rebound percentage (simplified, team-level)."""
        total = self.offensive_rebounds + self.defensive_rebounds
        return self.offensive_rebounds / total if total > 0 else 0.0

    @computed_field
    @property
    def ftr(self) -> float:
        """Free throw rate (FTA/FGA)."""
        if self.field_goals_attempted == 0:
            return 0.0
        return self.free_throws_attempted / self.field_goals_attempted

    @computed_field
    @property
    def points_per_possession(self) -> float:
        """Raw points per possession."""
        poss = self.possessions
        return (self.points / poss) * 100 if poss > 0 else 0.0


class LineSnapshot(BaseModel):
    """Point-in-time odds snapshot from a bookmaker."""

    model_config = ConfigDict(strict=True)

    game_id: int
    bookmaker: str = Field(..., min_length=1, max_length=50)
    snapshot_timestamp: datetime
    snapshot_type: Literal["open", "current", "close"]

    # Spread market
    spread_home: Optional[float] = Field(None, ge=-60.0, le=60.0)
    spread_home_price: Optional[int] = Field(None, ge=-1000, le=1000, description="American odds")
    spread_away: Optional[float] = Field(None, ge=-60.0, le=60.0)
    spread_away_price: Optional[int] = Field(None, ge=-1000, le=1000)

    # Total market
    total_line: Optional[float] = Field(None, ge=60.0, le=250.0)
    over_price: Optional[int] = Field(None, ge=-1000, le=1000)
    under_price: Optional[int] = Field(None, ge=-1000, le=1000)

    # Moneyline
    home_ml: Optional[int] = None
    away_ml: Optional[int] = None


# =============================================================================
# PREDICTION OUTPUTS
# =============================================================================


class PredictionRow(BaseModel):
    """Model prediction output for a single game."""

    model_config = ConfigDict(strict=True)

    game_id: int
    prediction_timestamp: datetime
    model_version: str = Field(..., pattern=r"^v\d+\.\d+\.\d+(?:-[A-Za-z0-9]+)?$")

    # Point estimates
    proj_home_score: float = Field(..., ge=30.0, le=170.0)
    proj_away_score: float = Field(..., ge=30.0, le=170.0)
    proj_spread: float  # home - away
    proj_total: float
    proj_possessions: float = Field(..., ge=45.0, le=95.0)

    # Win probability
    home_win_prob: float = Field(..., ge=0.0, le=1.0)

    # Uncertainty intervals (spread)
    spread_ci_50_lower: float
    spread_ci_50_upper: float
    spread_ci_80_lower: float
    spread_ci_80_upper: float
    spread_ci_95_lower: float
    spread_ci_95_upper: float

    # Uncertainty intervals (total)
    total_ci_50_lower: float
    total_ci_50_upper: float
    total_ci_80_lower: float
    total_ci_80_upper: float
    total_ci_95_lower: float
    total_ci_95_upper: float

    # Market comparison (if lines available)
    market_spread: Optional[float] = None
    edge_vs_market_spread: Optional[float] = None  # proj_spread - market_spread
    market_total: Optional[float] = None
    edge_vs_market_total: Optional[float] = None  # proj_total - market_total

    # Bet recommendation
    recommended_side: Optional[
        Literal["home_spread", "away_spread", "over", "under", "no_bet"]
    ] = None
    recommended_units: Optional[float] = Field(None, ge=0.0, le=5.0)
    confidence_rating: Optional[Literal["low", "medium", "high"]] = None


class CLVReport(BaseModel):
    """Closing Line Value analysis for a prediction."""

    model_config = ConfigDict(strict=True)

    game_id: int
    prediction_timestamp: datetime
    bet_timestamp: Optional[datetime] = None

    # Our prediction at time of hypothetical bet
    our_spread: float
    our_total: float

    # Market at time of bet
    market_spread_at_bet: float
    market_total_at_bet: float

    # Closing line (sharp reference)
    closing_spread: float
    closing_total: float

    # CLV calculations
    spread_clv: float  # closing_spread - market_spread_at_bet (positive = beat the close)
    total_clv: float

    # Actual result for validation
    actual_spread: Optional[float] = None
    actual_total: Optional[int] = None

    # Bet outcome (if simulated)
    spread_bet_side: Optional[Literal["home", "away"]] = None
    spread_bet_won: Optional[bool] = None
    total_bet_side: Optional[Literal["over", "under"]] = None
    total_bet_won: Optional[bool] = None


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


class FeatureRow(BaseModel):
    """Engineered features for a matchup."""

    model_config = ConfigDict(strict=True)

    game_id: int
    feature_timestamp: datetime

    # Team strengths (home)
    home_adj_off_eff: float
    home_adj_def_eff: float
    home_adj_tempo: float
    home_adj_em: float  # efficiency margin

    # Team strengths (away)
    away_adj_off_eff: float
    away_adj_def_eff: float
    away_adj_tempo: float
    away_adj_em: float

    # Home court
    home_court_global: float = Field(default=3.5, description="Global HCA in points")
    home_court_team_effect: float = Field(default=0.0, description="Team-specific deviation")
    is_neutral: bool = False

    # Four Factors differentials
    efg_diff: Optional[float] = None  # home off - away def
    to_diff: Optional[float] = None
    orb_diff: Optional[float] = None
    ftr_diff: Optional[float] = None

    # Context features
    home_rest_days: Optional[int] = Field(None, ge=0, le=30)
    away_rest_days: Optional[int] = Field(None, ge=0, le=30)
    rest_advantage: Optional[int] = Field(
        None, ge=-30, le=30, description="Home rest days - Away rest days"
    )

    # Season regime
    season_phase: SeasonPhase = SeasonPhase.NON_CONFERENCE
    games_into_season_home: int = Field(default=0, ge=0)
    games_into_season_away: int = Field(default=0, ge=0)

    # Market features (if available)
    opening_spread: Optional[float] = None
    current_spread: Optional[float] = None
    spread_movement: Optional[float] = None  # current - open
    book_disagreement: Optional[float] = None  # std dev across books


# =============================================================================
# API RESPONSE MODELS
# =============================================================================


class SlateResponse(BaseModel):
    """Daily slate of games with predictions."""

    date: date
    games_count: int
    predictions: list[PredictionRow]
    last_updated: datetime


class GameDetailResponse(BaseModel):
    """Detailed prediction for a single game."""

    game: Game
    prediction: PredictionRow
    home_team: Team
    away_team: Team
    line_history: list[LineSnapshot]
    feature_breakdown: Optional[FeatureRow] = None


class BacktestRequest(BaseModel):
    """Backtest configuration."""

    start_date: date
    end_date: date
    model_version: Optional[str] = None
    edge_threshold: float = Field(default=0.0, ge=0.0, le=10.0)


class BacktestResponse(BaseModel):
    """Backtest results summary."""

    period_start: date
    period_end: date
    total_games: int
    games_with_lines: int

    # Accuracy metrics
    spread_mae: float
    total_mae: float
    spread_rmse: float
    total_rmse: float

    # Calibration
    spread_50_coverage: float  # % of games within 50% CI
    spread_80_coverage: float
    spread_95_coverage: float

    # CLV metrics
    mean_spread_clv: float
    mean_total_clv: float
    clv_positive_rate: float  # % of predictions that beat close

    # Simulated betting (W-L-P)
    spread_wins: int
    spread_losses: int
    spread_pushes: int
    total_wins: int
    total_losses: int
    total_pushes: int
    simulated_roi: float


class HealthResponse(BaseModel):
    """Service health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    version: str
    database_connected: bool
    last_prediction_date: Optional[date] = None
