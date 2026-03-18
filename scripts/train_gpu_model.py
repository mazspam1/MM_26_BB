#!/usr/bin/env python3
"""
Train the GPU-accelerated hierarchical Bayesian model.

This script:
1. Fetches historical game data from ESPN
2. Fits the hierarchical model using MCMC on GPU
3. Saves the trained model for predictions
4. Generates predictions for today's games

Usage:
    python scripts/train_gpu_model.py [--n-games 1000] [--n-sims 100000]

Requirements:
    - CUDA 12.8+ (RTX 5090)
    - pip install -e ".[gpu]"
"""

import argparse
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path

import structlog

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def fetch_historical_games(n_games: int = 1000) -> list[dict]:
    """
    Fetch historical game results for model training.

    Returns list of dicts with:
    - home_team_id, away_team_id
    - home_score, away_score
    - is_neutral
    - rest_diff
    - game_date
    """
    from packages.common.database import get_connection

    logger.info("Fetching historical games from database", n_games=n_games)

    with get_connection() as conn:
        # Get completed games with scores
        rows = conn.execute(
            """
            SELECT
                g.game_id,
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score,
                g.neutral_site as is_neutral,
                g.game_date,
                ht.conference as home_conference,
                at.conference as away_conference
            FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.team_id
            LEFT JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.status = 'final'
                AND g.home_score IS NOT NULL
                AND g.away_score IS NOT NULL
            ORDER BY g.game_date DESC
            LIMIT ?
            """,
            (n_games,)
        ).fetchall()

    games = []
    for row in rows:
        games.append({
            "game_id": row[0],
            "home_team_id": row[1],
            "away_team_id": row[2],
            "home_score": row[3],
            "away_score": row[4],
            "is_neutral": bool(row[5]),
            "game_date": row[6],
            "home_conference": row[7] or "Unknown",
            "away_conference": row[8] or "Unknown",
            "rest_diff": 0,  # Would need to calculate from schedule
        })

    logger.info("Fetched games", count=len(games))
    return games


def build_mappings(games: list[dict]) -> tuple[dict, dict]:
    """
    Build team_id -> index and team_id -> conference mappings.
    """
    team_ids = set()
    team_to_conf = {}

    for g in games:
        team_ids.add(g["home_team_id"])
        team_ids.add(g["away_team_id"])
        team_to_conf[g["home_team_id"]] = g.get("home_conference", "Unknown")
        team_to_conf[g["away_team_id"]] = g.get("away_conference", "Unknown")

    team_id_map = {tid: idx for idx, tid in enumerate(sorted(team_ids))}

    # Conference to index
    conferences = sorted(set(team_to_conf.values()))
    conf_to_idx = {c: i for i, c in enumerate(conferences)}
    team_to_conf_idx = {tid: conf_to_idx[conf] for tid, conf in team_to_conf.items()}

    logger.info(
        "Built mappings",
        n_teams=len(team_id_map),
        n_conferences=len(conferences),
    )

    return team_id_map, team_to_conf_idx


def train_model(games: list[dict], n_sims: int = 100_000):
    """
    Train the hierarchical Bayesian model on GPU.
    """
    try:
        from packages.models.gpu_bayes import (
            HierarchicalNCAABModel,
            GPUModelConfig,
            GPU_AVAILABLE,
        )
    except ImportError as e:
        logger.error(
            "GPU model not available. Install with: pip install -e '.[gpu]'",
            error=str(e),
        )
        return None

    if not GPU_AVAILABLE:
        logger.warning("GPU not available, training will be slow on CPU")

    # Build mappings
    team_id_map, team_to_conf_idx = build_mappings(games)

    # Configure model
    config = GPUModelConfig(
        num_warmup=500,  # Reduce for faster testing
        num_samples=2000,
        num_chains=2,  # Reduce for faster testing
        n_simulations=n_sims,
        use_four_factors=False,  # Not yet implemented
        use_conference_hca=True,
        heteroscedastic=True,
    )

    # Initialize model
    model = HierarchicalNCAABModel(config)

    # Fit on historical data
    logger.info("Starting model training...")
    model.fit(
        games_data=games,
        team_id_map=team_id_map,
        conference_map=team_to_conf_idx,
    )

    return model, team_id_map, team_to_conf_idx


def save_model(model, team_id_map, team_to_conf_idx, output_path: Path):
    """Save trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "posterior_samples": {k: v.tolist() for k, v in model.posterior_samples.items()},
        "team_idx_map": model.team_idx_map,
        "conference_idx_map": model.conference_idx_map,
        "config": model.config,
        "trained_at": datetime.utcnow().isoformat(),
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info("Model saved", path=str(output_path))


def generate_predictions(model, team_id_map, team_to_conf_idx):
    """Generate predictions for today's games."""
    from packages.common.database import get_connection

    today = date.today()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                g.game_id,
                g.home_team_id,
                g.away_team_id,
                g.home_team_name,
                g.away_team_name,
                g.neutral_site,
                bs.spread_line_home,
                ls.spread_home, ls.spread_away
            FROM games g
            LEFT JOIN (
                SELECT
                    game_id,
                    spread_line_home,
                    ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY snapshot_timestamp DESC) AS rn
                FROM betting_splits
            ) bs ON g.game_id = bs.game_id AND bs.rn = 1
            LEFT JOIN (
                SELECT
                    game_id,
                    AVG(spread_home) AS spread_home,
                    AVG(spread_away) AS spread_away
                FROM line_snapshots
                GROUP BY game_id, snapshot_timestamp
                QUALIFY snapshot_timestamp = MAX(snapshot_timestamp)
                    OVER (PARTITION BY game_id)
            ) ls ON g.game_id = ls.game_id
            WHERE g.game_date = ?
                AND g.status = 'scheduled'
            """,
            (today.isoformat(),)
        ).fetchall()

    if not rows:
        logger.info("No games scheduled for today")
        return []

    predictions = []
    for row in rows:
        (
            game_id,
            home_id,
            away_id,
            home_name,
            away_name,
            is_neutral,
            splits_spread_home,
            spread_home,
            spread_away,
        ) = row
        if splits_spread_home is not None:
            market_spread = splits_spread_home
        elif spread_home is not None:
            market_spread = -spread_home
        elif spread_away is not None:
            market_spread = spread_away
        else:
            market_spread = None

        # Skip if team not in training data
        if home_id not in team_id_map or away_id not in team_id_map:
            logger.warning(
                "Skipping game with unknown team",
                home_id=home_id,
                away_id=away_id,
            )
            continue

        home_conf_idx = team_to_conf_idx.get(home_id, 0)

        pred = model.predict_game_mc(
            home_team_id=home_id,
            away_team_id=away_id,
            home_conference_idx=home_conf_idx,
            is_neutral=bool(is_neutral),
            rest_diff=0,
            market_spread=market_spread,
            game_id=game_id,
        )

        edge = pred.spread_mean - market_spread if market_spread else None

        predictions.append({
            "game_id": game_id,
            "matchup": f"{away_name} @ {home_name}",
            "model_spread": pred.spread_mean,
            "model_spread_std": pred.spread_std,
            "vegas_spread": market_spread,
            "edge": edge,
            "home_win_prob": pred.home_win_prob,
            "cover_prob": pred.cover_prob,
            "ci_95": pred.spread_ci_95,
            "n_sims": pred.n_simulations,
        })

        logger.info(
            "Prediction generated",
            matchup=f"{away_name} @ {home_name}",
            spread=f"{pred.spread_mean:+.1f} (±{pred.spread_std:.1f})",
            edge=f"{edge:+.1f}" if edge else "N/A",
            win_prob=f"{pred.home_win_prob:.1%}",
        )

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Train GPU-accelerated NCAAB model")
    parser.add_argument(
        "--n-games",
        type=int,
        default=500,
        help="Number of historical games to train on (default: 500)",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=100_000,
        help="Number of Monte Carlo simulations (default: 100,000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/gpu_model.pkl",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Generate predictions after training",
    )

    args = parser.parse_args()

    logger.info(
        "Starting GPU model training",
        n_games=args.n_games,
        n_sims=args.n_sims,
    )

    # Fetch data
    games = fetch_historical_games(args.n_games)

    if len(games) < 50:
        logger.error("Not enough historical games. Need at least 50.")
        return

    # Train model
    result = train_model(games, args.n_sims)

    if result is None:
        logger.error("Training failed")
        return

    model, team_id_map, team_to_conf_idx = result

    # Save model
    save_model(model, team_id_map, team_to_conf_idx, Path(args.output))

    # Generate predictions if requested
    if args.predict:
        predictions = generate_predictions(model, team_id_map, team_to_conf_idx)

        if predictions:
            logger.info("=" * 60)
            logger.info("TODAY'S PREDICTIONS")
            logger.info("=" * 60)

            for p in sorted(predictions, key=lambda x: abs(x.get("edge", 0) or 0), reverse=True):
                edge_str = f"{p['edge']:+.1f}" if p["edge"] else "N/A"
                print(
                    f"{p['matchup']:45} | "
                    f"Model: {p['model_spread']:+6.1f} (±{p['model_spread_std']:.1f}) | "
                    f"Vegas: {p['vegas_spread'] or 'N/A':>6} | "
                    f"Edge: {edge_str:>6} | "
                    f"Win: {p['home_win_prob']:.0%}"
                )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
