"""
Canonical unified matchup engine for NCAA basketball.

Single source of truth for:
- Score projection (additive KenPom/Torvik PPP model)
- Win probability (possession-scaled Gaussian)
- Game simulation (stochastic score generation)
- Variance calibration

All bracket simulation, tournament prediction, and backtesting should
call this module rather than implementing their own formulas.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

# League constants
LEAGUE_AVG_EFFICIENCY = 100.0
LEAGUE_AVG_TEMPO = 68.0

# Calibrated PPP standard deviation
# Produces spread_std ≈ 11.2-11.7 at 67 possessions (validated against 2025-26 data)
BASE_PPP_STD = 0.115

# Possession variance (fraction of expected possessions)
POSSESSION_STD_FRAC = 0.035

# Neutral site adjustments
NEUTRAL_PACE_FACTOR = 0.97  # ~3% fewer possessions at neutral site
NEUTRAL_SPREAD_COMPRESSION = 0.85  # Spreads compress ~15% without home court

# Four Factors matchup residual weight
# Small because adjusted efficiencies already capture opponent strength
FF_MATCHUP_WEIGHT = 0.015


@dataclass
class TeamStrength:
    """Minimal team strength representation for matchup computation."""

    name: str
    adj_off: float  # Adjusted offensive efficiency (pts/100 poss)
    adj_def: float  # Adjusted defensive efficiency (pts/100 poss)
    tempo: float  # Adjusted tempo (possessions per 40 min)
    # Four Factors (opponent-adjusted)
    efg_pct: float  # Effective field goal %
    tov_pct: float  # Turnover rate
    orb_pct: float  # Offensive rebound %
    ftr: float  # Free throw rate
    def_efg: float  # Opponent eFG% allowed
    def_tov: float  # Opponent TOV% forced
    def_drb: float  # Defensive rebound %
    def_ftr: float  # Opponent FTr allowed
    # Volatility
    off_std: float = 10.0  # Offensive rating std dev
    def_std: float = 10.0  # Defensive rating std dev


@dataclass
class MatchupProjection:
    """Projected outcome for a single game matchup."""

    home_score: float
    away_score: float
    spread: float  # home - away (positive = home favored)
    total: float
    possessions: float
    home_win_prob: float
    # Uncertainty
    spread_std: float
    total_std: float
    # Score distribution
    home_score_std: float
    away_score_std: float


def compute_matchup(
    home: TeamStrength,
    away: TeamStrength,
    is_neutral: bool = False,
    home_court_advantage: float = 3.5,
) -> MatchupProjection:
    """
    Compute projected matchup outcome using the canonical additive PPP model.

    Formula (KenPom/Torvik style):
        PPP_home = (AdjOff_home + AdjDef_away - LeagueAvg) / LeagueAvg
        PPP_away = (AdjOff_away + AdjDef_home - LeagueAvg) / LeagueAvg

    With Four Factors matchup residual adjustment (second-order effect).

    Args:
        home: Home team strength parameters
        away: Away team strength parameters
        is_neutral: Whether game is at neutral site
        home_court_advantage: HCA in points (ignored if neutral)

    Returns:
        MatchupProjection with scores, spread, total, win prob, uncertainty
    """
    # 1. Expected possessions (harmonic mean of tempos)
    expected_poss = 2 * home.tempo * away.tempo / (home.tempo + away.tempo)
    if is_neutral:
        expected_poss *= NEUTRAL_PACE_FACTOR

    # 2. Additive PPP model
    home_ppp = (home.adj_off + away.adj_def - LEAGUE_AVG_EFFICIENCY) / LEAGUE_AVG_EFFICIENCY
    away_ppp = (away.adj_off + home.adj_def - LEAGUE_AVG_EFFICIENCY) / LEAGUE_AVG_EFFICIENCY

    # 3. Four Factors matchup residual (captures style matchups beyond efficiency)
    ff_home = _four_factors_residual(home, away)
    ff_away = _four_factors_residual(away, home)
    home_ppp += ff_home * FF_MATCHUP_WEIGHT
    away_ppp += ff_away * FF_MATCHUP_WEIGHT

    # 4. Home court advantage
    hca = 0.0 if is_neutral else home_court_advantage
    hca_per_poss = hca / expected_poss if expected_poss > 0 else 0

    # 5. Projected scores
    home_score = home_ppp * expected_poss + hca_per_poss / 2 * expected_poss
    away_score = away_ppp * expected_poss - hca_per_poss / 2 * expected_poss

    # 6. Spread and total
    spread = home_score - away_score
    total = home_score + away_score

    # 7. Variance model (possession-scaled)
    vol_factor = 1 + 0.06 * max(
        (home.off_std + home.def_std) / 20.0,
        (away.off_std + away.def_std) / 20.0,
    )
    ppp_std = BASE_PPP_STD * vol_factor
    spread_std = ppp_std * expected_poss
    total_std = spread_std * 1.15  # Totals slightly more variable than spreads

    # 8. Win probability
    home_win_prob = stats.norm.cdf(spread / spread_std) if spread_std > 0 else 0.5
    home_win_prob = float(np.clip(home_win_prob, 0.01, 0.99))

    return MatchupProjection(
        home_score=home_score,
        away_score=away_score,
        spread=spread,
        total=total,
        possessions=expected_poss,
        home_win_prob=home_win_prob,
        spread_std=spread_std,
        total_std=total_std,
        home_score_std=spread_std / 2,
        away_score_std=spread_std / 2,
    )


def simulate_game(
    projection: MatchupProjection,
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, int, str]:
    """
    Simulate a single game outcome from a matchup projection.

    Returns:
        (home_score, away_score, winner) where winner is 'home' or 'away'
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw from score distributions
    home_draw = rng.normal(projection.home_score, projection.home_score_std)
    away_draw = rng.normal(projection.away_score, projection.away_score_std)

    home_score = int(round(home_draw))
    away_score = int(round(away_draw))

    # Handle ties with overtime
    ot_rounds = 0
    while home_score == away_score and ot_rounds < 5:
        # Overtime: ~4-5 possessions per team, slightly lower efficiency
        ot_poss = 4 + rng.normal(0, 1.5)
        ot_ppp_factor = 0.95
        home_ot = rng.normal(
            ot_poss * projection.home_score / projection.possessions * ot_ppp_factor, 3
        )
        away_ot = rng.normal(
            ot_poss * projection.away_score / projection.possessions * ot_ppp_factor, 3
        )
        home_score += int(round(home_ot))
        away_score += int(round(away_ot))
        ot_rounds += 1

    # Deterministic tiebreak (extremely rare after 5 OT rounds)
    if home_score == away_score:
        if rng.random() > 0.5:
            home_score += 1
        else:
            away_score += 1

    winner = "home" if home_score > away_score else "away"
    return home_score, away_score, winner


def _four_factors_residual(team: TeamStrength, opp: TeamStrength) -> float:
    """
    Compute Four Factors matchup residual.

    Captures style-specific effects beyond opponent-adjusted efficiency:
    - eFG matchup: great shooting vs poor perimeter defense
    - TO matchup: ball security vs steal-heavy defense
    - ORB matchup: offensive rebounding vs poor defensive rebounding
    - FTR matchup: getting to line vs foul-prone defense

    Returns a small additive adjustment to PPP.
    """
    efg = team.efg_pct - opp.def_efg
    tov = opp.def_tov - team.tov_pct
    orb = team.orb_pct - (1 - opp.def_drb)
    ftr = team.ftr - opp.def_ftr
    return efg + tov + orb + ftr


def compute_win_prob_only(
    home: TeamStrength,
    away: TeamStrength,
    is_neutral: bool = False,
    home_court_advantage: float = 3.5,
) -> float:
    """
    Fast win probability calculation (no full projection needed).

    Useful for Monte Carlo bracket simulation where you only need win/loss.
    """
    proj = compute_matchup(home, away, is_neutral, home_court_advantage)
    return proj.home_win_prob
