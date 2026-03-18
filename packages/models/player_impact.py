"""
RAPM-lite (Regularized Adjusted Plus-Minus) for NCAA basketball.

Implements a simplified RAPM model using ridge regression to estimate
player impact from lineup-level plus-minus data.

Since NCAA player tracking is limited, this uses:
- Box score stats as features (instead of raw +/-)
- Minutes-weighted ridge regression
- Team-level aggregation for prediction impact

Reference: RAPM concepts from tothemean.com/2018/10/05/ncaa-rapm.html
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
from sklearn.linear_model import Ridge

logger = structlog.get_logger()


@dataclass
class PlayerImpact:
    """Estimated player impact value."""

    player_id: str
    team_id: int
    player_name: str
    # RAPM-style estimates (points per 100 possessions)
    offensive_impact: float  # Positive = helps team score
    defensive_impact: float  # Positive = helps team defend (lower opp scoring)
    overall_impact: float  # Net impact (off - def)
    # Metadata
    minutes_per_game: float
    games_played: int
    confidence: float  # 0-1, higher with more minutes


@dataclass
class TeamPlayerImpact:
    """Aggregated player impact for a team."""

    team_id: int
    total_impact: float  # Sum of starter + key bench impacts
    starter_impact: float  # Impact of top 5 players
    bench_impact: float  # Impact of 6th+ players
    depth_rating: float  # 0-1, higher = deeper team
    star_power: float  # Impact of best player
    availability_factor: float  # 0-1, affected by injuries


def estimate_player_impact_from_boxscores(
    player_game_stats: list[dict],
    alpha: float = 100.0,  # Ridge regularization (high for NCAA noise)
) -> dict[str, PlayerImpact]:
    """
    Estimate player impact using ridge regression on per-game stats.

    Uses a simplified approach: predict team offensive/defensive efficiency
    from the players who played in each game, weighted by minutes.

    Args:
        player_game_stats: List of dicts with player_id, team_id, game_id,
                          minutes, points, rebounds, assists, etc.
        alpha: Ridge regularization parameter

    Returns:
        Dict of player_id -> PlayerImpact
    """
    if not player_game_stats:
        return {}

    # Build feature matrix: each row is a game, each column is a player
    # Feature = player's minutes in that game (normalized)
    game_ids = sorted(set(s["game_id"] for s in player_game_stats))
    player_ids = sorted(set(s["player_id"] for s in player_game_stats))

    game_idx = {g: i for i, g in enumerate(game_ids)}
    player_idx = {p: i for i, p in enumerate(player_ids)}

    n_games = len(game_ids)
    n_players = len(player_ids)

    if n_games < 10 or n_players < 5:
        logger.warning("Insufficient data for RAPM", games=n_games, players=n_players)
        return {}

    # Build lineup matrix and target vector
    # X[i, j] = normalized minutes for player j in game i
    X = np.zeros((n_games, n_players))
    y_off = np.zeros(n_games)  # Team offensive efficiency
    y_def = np.zeros(n_games)  # Team defensive efficiency

    # Track player minutes for confidence
    player_minutes = {p: 0.0 for p in player_ids}
    player_games = {p: 0 for p in player_ids}
    player_team = {}

    for stat in player_game_stats:
        pid = stat["player_id"]
        gid = stat["game_id"]
        mins = stat.get("minutes_per_game", 0)
        team_id = stat.get("team_id", 0)

        if pid in player_idx and gid in game_idx:
            gi = game_idx[gid]
            pi = player_idx[pid]
            X[gi, pi] = mins / 40.0  # Normalize to fraction of game
            player_minutes[pid] += mins
            player_games[pid] += 1
            player_team[pid] = team_id

    # Fill targets from game data (would need team game stats)
    # For now, use box score aggregates to approximate
    game_stats = {}
    for stat in player_game_stats:
        gid = stat["game_id"]
        if gid not in game_stats:
            game_stats[gid] = {"points": 0, "opp_points": 0, "possessions": 70}
        game_stats[gid]["points"] += stat.get("points_per_game", 0)

    for gid in game_ids:
        if gid in game_stats:
            gs = game_stats[gid]
            poss = max(gs.get("possessions", 70), 50)
            y_off[game_idx[gid]] = (gs["points"] / poss) * 100

    # Normalize features per game (so each row sums to ~1)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    X_norm = X / row_sums

    # Fit ridge regression for offensive impact
    ridge_off = Ridge(alpha=alpha, fit_intercept=True)
    ridge_off.fit(X_norm, y_off)

    # Fit ridge regression for defensive impact (use opponent scoring if available)
    ridge_def = Ridge(alpha=alpha, fit_intercept=True)
    # Approximate defensive target (lower = better defense)
    y_def_approx = 100.0 - y_off  # Simplified
    ridge_def.fit(X_norm, y_def_approx)

    # Extract player impacts
    player_impacts = {}
    for pid in player_ids:
        pi = player_idx[pid]
        total_mins = player_minutes[pid]
        gp = player_games[pid]

        if gp < 3:  # Skip players with too few games
            continue

        off_impact = float(ridge_off.coef_[pi])
        def_impact = float(ridge_def.coef_[pi])
        overall = off_impact - def_impact

        # Confidence based on sample size (more minutes = more confident)
        confidence = min(1.0, total_mins / (gp * 20.0))  # Full confidence at 20+ min/game

        player_impacts[pid] = PlayerImpact(
            player_id=pid,
            team_id=player_team.get(pid, 0),
            player_name=pid,  # Would need name lookup
            offensive_impact=round(off_impact, 2),
            defensive_impact=round(def_impact, 2),
            overall_impact=round(overall, 2),
            minutes_per_game=round(total_mins / max(gp, 1), 1),
            games_played=gp,
            confidence=round(confidence, 2),
        )

    logger.info("RAPM-lite estimated", players=len(player_impacts))
    return player_impacts


def aggregate_team_impact(
    player_impacts: dict[str, PlayerImpact],
    team_id: int,
    injured_player_ids: Optional[set[str]] = None,
) -> TeamPlayerImpact:
    """
    Aggregate individual player impacts into team-level impact.

    Args:
        player_impacts: Dict of player_id -> PlayerImpact
        team_id: Team to aggregate
        injured_player_ids: Set of injured player IDs to exclude

    Returns:
        TeamPlayerImpact
    """
    if injured_player_ids is None:
        injured_player_ids = set()

    # Get players for this team
    team_players = [
        p
        for p in player_impacts.values()
        if p.team_id == team_id and p.player_id not in injured_player_ids
    ]

    if not team_players:
        return TeamPlayerImpact(
            team_id=team_id,
            total_impact=0.0,
            starter_impact=0.0,
            bench_impact=0.0,
            depth_rating=0.5,
            star_power=0.0,
            availability_factor=1.0,
        )

    # Sort by overall impact (best first)
    team_players.sort(key=lambda p: p.overall_impact * p.confidence, reverse=True)

    # Weight by minutes and confidence
    total_impact = sum(p.overall_impact * p.confidence for p in team_players)

    # Starters = top 5
    starters = team_players[:5]
    starter_impact = sum(p.overall_impact * p.confidence for p in starters)

    # Bench = rest
    bench = team_players[5:]
    bench_impact = sum(p.overall_impact * p.confidence for p in bench)

    # Depth rating: how many players play significant minutes
    depth = sum(1 for p in team_players if p.minutes_per_game > 10)
    depth_rating = min(1.0, depth / 8.0)

    # Star power: best player's impact
    star_power = (
        team_players[0].overall_impact * team_players[0].confidence if team_players else 0.0
    )

    # Availability: what % of total impact is available
    all_team_players = [p for p in player_impacts.values() if p.team_id == team_id]
    total_possible = sum(p.overall_impact * p.confidence for p in all_team_players)
    available = sum(p.overall_impact * p.confidence for p in team_players)
    availability_factor = available / total_possible if abs(total_possible) > 0.01 else 1.0

    return TeamPlayerImpact(
        team_id=team_id,
        total_impact=round(total_impact, 2),
        starter_impact=round(starter_impact, 2),
        bench_impact=round(bench_impact, 2),
        depth_rating=round(depth_rating, 2),
        star_power=round(star_power, 2),
        availability_factor=round(availability_factor, 2),
    )


def get_player_impact_adjustment(
    home_impact: TeamPlayerImpact,
    away_impact: TeamPlayerImpact,
) -> float:
    """
    Calculate spread adjustment from player impact differential.

    Positive = favors home team.

    Args:
        home_impact: Home team's player impact aggregate
        away_impact: Away team's player impact aggregate

    Returns:
        Spread adjustment in points
    """
    # Differential impact (scaled down - player impact is noisy)
    impact_diff = home_impact.total_impact - away_impact.total_impact

    # Scale to points (RAPM is per 100 possessions, ~70 poss/game)
    points_adjustment = impact_diff * 0.7  # Conservative scaling

    # Clamp to reasonable range
    return max(-5.0, min(5.0, points_adjustment))
