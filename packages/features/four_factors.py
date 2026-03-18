"""
Dean Oliver's Four Factors for basketball analysis.

The Four Factors explain why teams win:
1. eFG% (Effective Field Goal %) - shooting efficiency
2. TO% (Turnover %) - ball security
3. ORB% (Offensive Rebound %) - second chances
4. FTr (Free Throw Rate) - getting to the line

References:
- KenPom: https://kenpom.com/blog/four-factors/
- Basketball Reference methodology
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

from packages.common.schemas import BoxScore

logger = structlog.get_logger()

# Research-backed weights for Four Factors (Oliver/Basketball Reference updated 2023)
# These represent relative importance in explaining wins
# Reference: https://kenpom.com/blog/four-factors/
FOUR_FACTOR_WEIGHTS = {
    "efg": 0.46,  # Most important (was 0.40)
    "to": 0.35,  # Second most important (was 0.25)
    "orb": 0.12,  # Third (was 0.20)
    "ftr": 0.07,  # Least important (was 0.15)
}


@dataclass
class FourFactors:
    """Four Factors for a team in a game."""

    efg_pct: float  # Effective Field Goal %
    to_pct: float  # Turnover % (turnovers per possession)
    orb_pct: float  # Offensive Rebound %
    ftr: float  # Free Throw Rate (FTA/FGA)


@dataclass
class FourFactorsDifferential:
    """Differential between two teams' Four Factors."""

    efg_diff: float  # Positive = offense better than defense
    to_diff: float  # Positive = offense has fewer turnovers (better)
    orb_diff: float  # Positive = offense gets more ORBs
    ftr_diff: float  # Positive = offense gets to line more


def calculate_efg_pct(
    field_goals_made: int,
    field_goals_attempted: int,
    three_pointers_made: int,
) -> float:
    """
    Calculate Effective Field Goal Percentage.

    eFG% = (FGM + 0.5 * 3PM) / FGA

    This adjusts for the extra value of three-pointers.

    Args:
        field_goals_made: Total field goals made (includes 3s)
        field_goals_attempted: Total field goal attempts
        three_pointers_made: Three pointers made

    Returns:
        Effective field goal percentage (0-1)
    """
    if field_goals_attempted == 0:
        return 0.0
    return (field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted


def calculate_to_pct(turnovers: int, possessions: float) -> float:
    """
    Calculate Turnover Percentage.

    TO% = Turnovers / Possessions

    Lower is better (fewer turnovers per possession).

    Args:
        turnovers: Total turnovers
        possessions: Team possessions

    Returns:
        Turnover percentage (0-1)
    """
    if possessions <= 0:
        return 0.0
    return turnovers / possessions


def calculate_orb_pct(
    offensive_rebounds: int,
    team_offensive_rebounds: int,
    opponent_defensive_rebounds: int,
) -> float:
    """
    Calculate Offensive Rebound Percentage.

    ORB% = ORB / (ORB + Opp DRB)

    This is the percentage of available offensive rebounds grabbed.

    Args:
        offensive_rebounds: Team offensive rebounds
        team_offensive_rebounds: Same as offensive_rebounds (for clarity)
        opponent_defensive_rebounds: Opponent's defensive rebounds

    Returns:
        Offensive rebound percentage (0-1)
    """
    total = offensive_rebounds + opponent_defensive_rebounds
    if total == 0:
        return 0.0
    return offensive_rebounds / total


def calculate_drb_pct(
    defensive_rebounds: int,
    opponent_offensive_rebounds: int,
) -> float:
    """
    Calculate Defensive Rebound Percentage.

    DRB% = DRB / (DRB + Opp ORB)

    Args:
        defensive_rebounds: Team defensive rebounds
        opponent_offensive_rebounds: Opponent's offensive rebounds

    Returns:
        Defensive rebound percentage (0-1)
    """
    total = defensive_rebounds + opponent_offensive_rebounds
    if total == 0:
        return 0.0
    return defensive_rebounds / total


def calculate_ftr(free_throws_attempted: int, field_goals_attempted: int) -> float:
    """
    Calculate Free Throw Rate.

    FTr = FTA / FGA

    Higher means getting to the free throw line more often.

    Args:
        free_throws_attempted: Free throw attempts
        field_goals_attempted: Field goal attempts

    Returns:
        Free throw rate
    """
    if field_goals_attempted == 0:
        return 0.0
    return free_throws_attempted / field_goals_attempted


def calculate_four_factors_from_boxscore(box: BoxScore) -> FourFactors:
    """
    Calculate Four Factors from a BoxScore object.

    Args:
        box: BoxScore object

    Returns:
        FourFactors dataclass
    """
    return FourFactors(
        efg_pct=box.efg_pct,
        to_pct=box.to_pct,
        orb_pct=box.orb_pct,
        ftr=box.ftr,
    )


def calculate_four_factors_differential(
    offense_factors: FourFactors,
    defense_factors: FourFactors,
) -> FourFactorsDifferential:
    """
    Calculate differential between offense and defense Four Factors.

    For eFG, ORB, FTr: higher offense is better
    For TO: lower offense is better (so we reverse the sign)

    Args:
        offense_factors: Offensive team's factors
        defense_factors: Defensive team's factors (opponent's defensive numbers)

    Returns:
        FourFactorsDifferential dataclass
    """
    return FourFactorsDifferential(
        efg_diff=offense_factors.efg_pct - defense_factors.efg_pct,
        # For TO, lower is better, so positive diff means offense is better
        to_diff=defense_factors.to_pct - offense_factors.to_pct,
        orb_diff=offense_factors.orb_pct - defense_factors.orb_pct,
        ftr_diff=offense_factors.ftr - defense_factors.ftr,
    )


def four_factors_composite_score(
    factors: FourFactors,
    weights: Optional[dict] = None,
) -> float:
    """
    Calculate weighted composite score from Four Factors.

    Higher is better for offense.

    Args:
        factors: FourFactors dataclass
        weights: Optional custom weights (default: FOUR_FACTOR_WEIGHTS)

    Returns:
        Composite score (normalized)
    """
    if weights is None:
        weights = FOUR_FACTOR_WEIGHTS

    # Normalize factors to similar scales
    # eFG typically 0.40-0.60
    # TO% typically 0.12-0.22
    # ORB% typically 0.20-0.40
    # FTr typically 0.20-0.40

    # Scale to 0-1 range based on typical D1 ranges
    efg_scaled = (factors.efg_pct - 0.40) / 0.20  # 0.40-0.60 -> 0-1
    to_scaled = (0.22 - factors.to_pct) / 0.10  # 0.12-0.22 -> 1-0 (inverted)
    orb_scaled = (factors.orb_pct - 0.20) / 0.20  # 0.20-0.40 -> 0-1
    ftr_scaled = (factors.ftr - 0.20) / 0.20  # 0.20-0.40 -> 0-1

    # Clamp to 0-1
    efg_scaled = max(0, min(1, efg_scaled))
    to_scaled = max(0, min(1, to_scaled))
    orb_scaled = max(0, min(1, orb_scaled))
    ftr_scaled = max(0, min(1, ftr_scaled))

    return (
        weights["efg"] * efg_scaled
        + weights["to"] * to_scaled
        + weights["orb"] * orb_scaled
        + weights["ftr"] * ftr_scaled
    )


def differential_composite_score(
    diff: FourFactorsDifferential,
    weights: Optional[dict] = None,
) -> float:
    """
    Calculate weighted composite score from Four Factors differential.

    Positive score favors the offensive team.

    Args:
        diff: FourFactorsDifferential dataclass
        weights: Optional custom weights

    Returns:
        Composite differential score
    """
    if weights is None:
        weights = FOUR_FACTOR_WEIGHTS

    # Scale differentials to points
    # Each 1% of eFG is roughly worth 1 point per game
    # TO differential of 1% is worth about 0.6 points
    # ORB differential of 1% is worth about 0.5 points
    # FTr differential of 1% is worth about 0.3 points

    return (
        diff.efg_diff * 100 * weights["efg"] * 2.5  # ~1 point per 1%
        + diff.to_diff * 100 * weights["to"] * 0.6
        + diff.orb_diff * 100 * weights["orb"] * 0.5
        + diff.ftr_diff * 100 * weights["ftr"] * 0.3
    )


def estimate_efficiency_from_four_factors(
    factors: FourFactors,
    base_efficiency: float = 100.0,
) -> float:
    """
    Estimate points per 100 possessions from Four Factors.

    This is an approximation based on typical relationships.

    Args:
        factors: FourFactors dataclass
        base_efficiency: Base efficiency to adjust from

    Returns:
        Estimated efficiency (points per 100 possessions)
    """
    # Start with base efficiency
    efficiency = base_efficiency

    # Adjust for eFG (dominant factor)
    # League avg eFG is ~0.50, each 1% above/below is ~2 points
    efg_adjustment = (factors.efg_pct - 0.50) * 200
    efficiency += efg_adjustment

    # Adjust for TO% (each 1% costs ~1.3 points)
    # League avg TO% is ~0.17
    to_adjustment = (0.17 - factors.to_pct) * 130
    efficiency += to_adjustment

    # Adjust for ORB% (each 1% adds ~0.3 points)
    # League avg ORB% is ~0.30
    orb_adjustment = (factors.orb_pct - 0.30) * 30
    efficiency += orb_adjustment

    # Adjust for FTr (each 0.01 adds ~0.2 points)
    # League avg FTr is ~0.30
    ftr_adjustment = (factors.ftr - 0.30) * 20
    efficiency += ftr_adjustment

    return efficiency


def aggregate_team_four_factors(
    game_factors: list[FourFactors],
    weights: Optional[list[float]] = None,
) -> FourFactors:
    """
    Aggregate Four Factors across multiple games.

    Args:
        game_factors: List of FourFactors from individual games
        weights: Optional game weights (e.g., recency, opponent strength)

    Returns:
        Aggregated FourFactors
    """
    if not game_factors:
        return FourFactors(efg_pct=0.5, to_pct=0.17, orb_pct=0.30, ftr=0.30)

    if weights is None:
        weights = [1.0] * len(game_factors)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted average
    efg = sum(f.efg_pct * w for f, w in zip(game_factors, weights))
    to = sum(f.to_pct * w for f, w in zip(game_factors, weights))
    orb = sum(f.orb_pct * w for f, w in zip(game_factors, weights))
    ftr = sum(f.ftr * w for f, w in zip(game_factors, weights))

    return FourFactors(efg_pct=efg, to_pct=to, orb_pct=orb, ftr=ftr)
