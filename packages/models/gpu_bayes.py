"""
GPU-Accelerated Hierarchical Bayesian Model for NCAA Basketball.

Uses NumPyro/JAX for GPU-accelerated Bayesian inference with:
- Hierarchical team strength priors (partial pooling)
- Conference-specific home court advantage
- Heteroscedastic variance (game-dependent uncertainty)
- Full uncertainty propagation through Monte Carlo
- 10k-100k simulations on RTX 5090

Requires: CUDA 12.8+, JAX with GPU support, NumPyro
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import warnings

import numpy as np
import structlog

logger = structlog.get_logger()

# Check for GPU availability
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive

    # Try to use GPU
    try:
        # Don't force platform unless we're sure
        # jax.config.update("jax_platform_name", "gpu") 
        GPU_AVAILABLE = len(jax.devices("gpu")) > 0
        if GPU_AVAILABLE:
            logger.info("JAX GPU backend available", devices=jax.devices("gpu"))
        else:
            logger.warning("JAX GPU backend not found. Using CPU.")
    except Exception as e:
        GPU_AVAILABLE = False
        logger.warning("GPU check failed, falling back to CPU", error=str(e))

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    GPU_AVAILABLE = False
    logger.warning("NumPyro/JAX not installed. GPU model unavailable.")


MODEL_VERSION = "v0.2.0-gpu"


@dataclass
class GPUModelConfig:
    """Configuration for GPU-accelerated model."""

    # MCMC settings
    num_warmup: int = 1000
    num_samples: int = 5000
    num_chains: int = 4

    # Monte Carlo simulation
    n_simulations: int = 100_000

    # Priors
    hca_prior_mean: float = 3.5
    hca_prior_std: float = 1.5
    efficiency_prior_mean: float = 100.0
    efficiency_prior_std: float = 15.0

    # Model settings
    use_four_factors: bool = True
    use_conference_hca: bool = True
    heteroscedastic: bool = True

    version: str = MODEL_VERSION


@dataclass
class MCPrediction:
    """Monte Carlo prediction with full uncertainty quantification."""

    game_id: int
    spread_mean: float
    spread_std: float
    spread_median: float
    total_mean: float
    total_std: float
    total_median: float
    home_win_prob: float
    cover_prob: Optional[float]  # Prob of covering market spread

    # Confidence intervals from Monte Carlo
    spread_ci_50: tuple[float, float]
    spread_ci_80: tuple[float, float]
    spread_ci_95: tuple[float, float]
    total_ci_50: tuple[float, float]
    total_ci_80: tuple[float, float]
    total_ci_95: tuple[float, float]

    # Simulation metadata
    n_simulations: int
    model_version: str


class HierarchicalNCAABModel:
    """
    Hierarchical Bayesian model for NCAA basketball predictions.

    Key improvements over baseline:
    1. Partial pooling: Teams share information through hierarchical priors
    2. Conference-specific HCA: Different conferences have different home advantages
    3. Heteroscedastic variance: Uncertainty varies by game type
    4. Full posterior sampling: Not just point estimates
    5. GPU acceleration: 100k simulations in seconds
    """

    def __init__(self, config: Optional[GPUModelConfig] = None):
        if not NUMPYRO_AVAILABLE:
            raise ImportError(
                "NumPyro/JAX required. Install with: pip install numpyro jax[cuda12]"
            )

        self.config = config or GPUModelConfig()
        self.posterior_samples = None
        self.team_idx_map = {}  # team_id -> index
        self.conference_idx_map = {}  # conference -> index
        self._rng_key = random.PRNGKey(42)

        logger.info(
            "Initialized HierarchicalNCAABModel",
            version=self.config.version,
            gpu_available=GPU_AVAILABLE,
            n_sims=self.config.n_simulations,
        )

    def _model(
        self,
        home_idx,
        away_idx,
        home_conf_idx,
        away_conf_idx,
        home_week_idx,
        away_week_idx,
        is_neutral,
        rest_diff,
        n_weeks,
        home_score=None,
        away_score=None,
    ):
        """
        Dynamic State-Space Model (Gaussian Random Walk).

        Teams evolve over time:
        theta[t] ~ Normal(theta[t-1], sigma_drift)

        Allows the model to track recent form and injuries automatically.
        """
        n_teams = len(self.team_idx_map)
        n_conferences = max(len(self.conference_idx_map), 1)

        # ===== HYPERPARAMETERS =====

        # Global average efficiency (static)
        mu_off = numpyro.sample(
            "mu_off",
            dist.Normal(self.config.efficiency_prior_mean, 5.0)
        )
        mu_def = numpyro.sample(
            "mu_def",
            dist.Normal(self.config.efficiency_prior_mean, 5.0)
        )

        # Volatility: How much do teams change week-to-week?
        # A tight prior keeps it stable, loose prior chases noise.
        # ~1.0-2.0 points variance per week is reasonable for NCAA.
        sigma_drift_off = numpyro.sample("sigma_drift_off", dist.HalfNormal(1.0))
        sigma_drift_def = numpyro.sample("sigma_drift_def", dist.HalfNormal(1.0))

        # Initial Team Variation (Start of Season)
        sigma_initial = numpyro.sample("sigma_initial", dist.HalfNormal(10.0))

        # Learned Rest Factor (was hardcoded 0.3)
        rest_factor = numpyro.sample("rest_factor", dist.Normal(0.3, 0.5))

        # ===== DYNAMIC TEAM ABILITIES (Gaussian Random Walk) =====

        # 1. Initial Ratings (Week 0)
        # using non-centered parameterization for better geometry
        with numpyro.plate("teams", n_teams):
            # Shape: (n_teams,)
            off_raw_init = numpyro.sample("off_raw_init", dist.Normal(0.0, 1.0))
            def_raw_init = numpyro.sample("def_raw_init", dist.Normal(0.0, 1.0))
            
            # Static Tempo (Tempo is much more stable than efficiency)
            team_tempo = numpyro.sample("team_tempo", dist.Normal(68.0, 4.0))

        off_init = mu_off + off_raw_init * sigma_initial
        def_init = mu_def + def_raw_init * sigma_initial

        # 2. Weekly Evolution (Random Walk)
        # We need (n_weeks - 1) steps of drift
        if n_weeks > 1:
            with numpyro.plate("weeks", n_weeks - 1):
                with numpyro.plate("teams_evolve", n_teams):
                    off_drift = numpyro.sample("off_drift", dist.Normal(0.0, 1.0))
                    def_drift = numpyro.sample("def_drift", dist.Normal(0.0, 1.0))

            # Accumulate drift: Shape (n_weeks, n_teams)
            # Week 0 is just init. Week t is init + sum(drifts_0_to_t)
            # Transpose because plate nesting might yield (n_teams, n_weeks) depending on dim assignment
            off_drifts_scaled = off_drift.T * sigma_drift_off
            def_drifts_scaled = def_drift.T * sigma_drift_def
            
            # Scan or Cumsum to build the walk
            # jnp.cumsum is efficient. We prepend zeros for week 0.
            off_walk = jnp.concatenate([
                jnp.zeros((1, n_teams)),
                jnp.cumsum(off_drifts_scaled, axis=0)
            ], axis=0)
            
            def_walk = jnp.concatenate([
                jnp.zeros((1, n_teams)),
                jnp.cumsum(def_drifts_scaled, axis=0)
            ], axis=0)
            
            # Add baseline to walk
            # Broadcasting: (n_weeks, n_teams) + (n_teams,)
            off_ability_time = off_walk + off_init
            def_ability_time = def_walk + def_init
        else:
            # Fallback for single week data
            off_ability_time = jnp.expand_dims(off_init, 0)
            def_ability_time = jnp.expand_dims(def_init, 0)

        # Store for posterior lookup
        numpyro.deterministic("off_ability_final", off_ability_time[-1])
        numpyro.deterministic("def_ability_final", def_ability_time[-1])

        # ===== HOME COURT ADVANTAGE =====

        hca_global = numpyro.sample(
            "hca_global",
            dist.Normal(self.config.hca_prior_mean, self.config.hca_prior_std)
        )

        if self.config.use_conference_hca and n_conferences > 1:
            with numpyro.plate("conferences", n_conferences):
                hca_conf = numpyro.sample("hca_conf", dist.Normal(0.0, 1.0))
        else:
            hca_conf = jnp.zeros(n_conferences)

        # ===== GAME PREDICTIONS =====

        # Lookup abilities for the specific week of each game
        # Advanced indexing: off_ability_time[week_idx, team_idx]
        home_off = off_ability_time[home_week_idx, home_idx]
        home_def = def_ability_time[home_week_idx, home_idx]
        away_off = off_ability_time[away_week_idx, away_idx]
        away_def = def_ability_time[away_week_idx, away_idx]

        # Effective HCA
        effective_hca = jnp.where(
            is_neutral,
            0.0,
            hca_global + hca_conf[home_conf_idx]
        )

        # Rest adjustment (LEARNED now)
        rest_adjustment = rest_factor * rest_diff

        # Expected scores
        home_ppp = (home_off + (100.0 - away_def)) / 100.0
        away_ppp = (away_off + (100.0 - home_def)) / 100.0

        expected_poss = (
            2.0 * team_tempo[home_idx] * team_tempo[away_idx]
            / (team_tempo[home_idx] + team_tempo[away_idx])
        )

        expected_home = home_ppp * expected_poss + effective_hca / 2.0 + rest_adjustment
        expected_away = away_ppp * expected_poss - effective_hca / 2.0 - rest_adjustment

        # ===== LIKELIHOOD =====

        if self.config.heteroscedastic:
            sigma_base = numpyro.sample("sigma_base", dist.HalfNormal(8.0))
            expected_margin = jnp.abs(expected_home - expected_away)
            sigma_game = sigma_base + 0.1 * expected_margin
        else:
            sigma_game = numpyro.sample("sigma_game", dist.HalfNormal(11.0))

        numpyro.sample("home_score", dist.Normal(expected_home, sigma_game), obs=home_score)
        numpyro.sample("away_score", dist.Normal(expected_away, sigma_game), obs=away_score)
        
        # Track rest factor for inspection
        numpyro.deterministic("rest_factor_posterior", rest_factor)

    def fit(
        self,
        games_data: list[dict],
        team_id_map: dict[int, int],
        conference_map: dict[int, int],
    ):
        """
        Fit the dynamic model. Calculates weeks from game dates.
        """
        self.team_idx_map = team_id_map
        self.conference_idx_map = {v: i for i, v in enumerate(set(conference_map.values()))}

        # Date processing for Time Steps (Weeks)
        # Find earliest date to establish week 0
        dates = [
            datetime.fromisoformat(g["game_date"]).date() 
            if isinstance(g["game_date"], str) else g["game_date"] 
            for g in games_data
        ]
        min_date = min(dates)
        
        # Calculate week index for each game (integer weeks from start)
        game_weeks = []
        for d in dates:
            days_diff = (d - min_date).days
            week_idx = days_diff // 7
            game_weeks.append(week_idx)
        
        n_weeks = max(game_weeks) + 1
        self.current_week_idx = n_weeks - 1  # Store for prediction

        # Prepare arrays
        home_idx = jnp.array([team_id_map[g["home_team_id"]] for g in games_data])
        away_idx = jnp.array([team_id_map[g["away_team_id"]] for g in games_data])
        
        home_conf_idx = jnp.array([
            self.conference_idx_map.get(conference_map.get(g["home_team_id"], 0), 0)
            for g in games_data
        ])
        away_conf_idx = jnp.array([
            self.conference_idx_map.get(conference_map.get(g["away_team_id"], 0), 0)
            for g in games_data
        ])
        
        home_week_idx = jnp.array(game_weeks)
        away_week_idx = jnp.array(game_weeks) # Same game, same week
        
        is_neutral = jnp.array([g.get("is_neutral", False) for g in games_data])
        rest_diff = jnp.array([g.get("rest_diff", 0) for g in games_data])
        home_score = jnp.array([g["home_score"] for g in games_data])
        away_score = jnp.array([g["away_score"] for g in games_data])

        logger.info(
            "Fitting dynamic state-space model",
            n_games=len(games_data),
            n_teams=len(team_id_map),
            n_weeks=n_weeks,
        )

        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=self.config.num_warmup,
            num_samples=self.config.num_samples,
            num_chains=self.config.num_chains,
            progress_bar=True,
        )

        self._rng_key, subkey = random.split(self._rng_key)
        mcmc.run(
            subkey,
            home_idx=home_idx,
            away_idx=away_idx,
            home_conf_idx=home_conf_idx,
            away_conf_idx=away_conf_idx,
            home_week_idx=home_week_idx,
            away_week_idx=away_week_idx,
            is_neutral=is_neutral,
            rest_diff=rest_diff,
            n_weeks=n_weeks,
            home_score=home_score,
            away_score=away_score,
        )

        self.posterior_samples = mcmc.get_samples()
        
        # Log learned rest factor
        rest_mean = float(jnp.mean(self.posterior_samples["rest_factor"]))
        logger.info(
            "Model fitting complete",
            rest_factor_mean=rest_mean,
            hca_mean=float(jnp.mean(self.posterior_samples["hca_global"])),
        )

    def predict_game_mc(
        self,
        home_team_id: int,
        away_team_id: int,
        home_conference_idx: int = 0,
        is_neutral: bool = False,
        rest_diff: int = 0,
        market_spread: Optional[float] = None,
        game_id: int = 0,
    ) -> MCPrediction:
        """
        Generate prediction with full Monte Carlo uncertainty propagation.

        Uses the FINAL week's learned ratings (current form).
        """
        if self.posterior_samples is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        home_idx = self.team_idx_map.get(home_team_id)
        away_idx = self.team_idx_map.get(away_team_id)

        if home_idx is None or away_idx is None:
            missing = []
            if home_idx is None:
                missing.append(f"home_team_id={home_team_id}")
            if away_idx is None:
                missing.append(f"away_team_id={away_team_id}")
            raise ValueError(
                f"Cannot predict: team(s) not in training data: {', '.join(missing)}. "
            )

        n_sims = self.config.n_simulations
        n_posterior = len(self.posterior_samples["hca_global"])

        # Get posterior samples (using FINAL week for prediction)
        # These are deterministic outputs we created in _model
        off_samples = self.posterior_samples["off_ability_final"]  # [n_posterior, n_teams]
        def_samples = self.posterior_samples["def_ability_final"]
        tempo_samples = self.posterior_samples["team_tempo"]
        hca_samples = self.posterior_samples["hca_global"]
        rest_factor_samples = self.posterior_samples["rest_factor"]

        if "hca_conf" in self.posterior_samples:
            hca_conf_samples = self.posterior_samples["hca_conf"]
        else:
            hca_conf_samples = jnp.zeros((n_posterior, 1))

        if self.config.heteroscedastic and "sigma_base" in self.posterior_samples:
            sigma_samples = self.posterior_samples["sigma_base"]
        else:
            sigma_samples = self.posterior_samples.get(
                "sigma_game",
                jnp.ones(n_posterior) * 11.0
            )

        # Vectorized Monte Carlo simulation on GPU
        self._rng_key, *subkeys = random.split(self._rng_key, n_sims + 1)
        subkeys = jnp.array(subkeys)

        def single_simulation(key):
            """Single game simulation with full uncertainty."""
            key, k1, k2, k3 = random.split(key, 4)

            # Sample random posterior index
            post_idx = random.randint(k1, (), 0, n_posterior)

            # Get team abilities from posterior
            home_off = off_samples[post_idx, home_idx]
            home_def = def_samples[post_idx, home_idx]
            home_tempo = tempo_samples[post_idx, home_idx]

            away_off = off_samples[post_idx, away_idx]
            away_def = def_samples[post_idx, away_idx]
            away_tempo = tempo_samples[post_idx, away_idx]

            # HCA
            hca = jnp.where(
                is_neutral,
                0.0,
                hca_samples[post_idx] + hca_conf_samples[post_idx, home_conference_idx]
            )

            # Rest adjustment (learned)
            rest_adj = rest_factor_samples[post_idx] * rest_diff

            # Calculate expected scores
            home_ppp = (home_off + (100.0 - away_def)) / 100.0
            away_ppp = (away_off + (100.0 - home_def)) / 100.0
            expected_poss = 2.0 * home_tempo * away_tempo / (home_tempo + away_tempo)

            expected_home = home_ppp * expected_poss + hca / 2.0 + rest_adj
            expected_away = away_ppp * expected_poss - hca / 2.0 - rest_adj

            # Get game variance
            sigma = sigma_samples[post_idx]
            if self.config.heteroscedastic:
                expected_margin = jnp.abs(expected_home - expected_away)
                sigma = sigma + 0.1 * expected_margin

            # Add game-level noise
            home_score = expected_home + sigma * random.normal(k2)
            away_score = expected_away + sigma * random.normal(k3)

            return home_score, away_score

        # Run simulations in parallel on GPU!
        home_scores, away_scores = jax.vmap(single_simulation)(subkeys)

        # Calculate derived quantities
        margins = home_scores - away_scores
        totals = home_scores + away_scores

        # Calculate cover probability if market spread provided
        cover_prob = None
        if market_spread is not None:
            # Positive market_spread means home is favored
            cover_prob = float(jnp.mean(margins > market_spread))

        return MCPrediction(
            game_id=game_id,
            spread_mean=float(jnp.mean(margins)),
            spread_std=float(jnp.std(margins)),
            spread_median=float(jnp.median(margins)),
            total_mean=float(jnp.mean(totals)),
            total_std=float(jnp.std(totals)),
            total_median=float(jnp.median(totals)),
            home_win_prob=float(jnp.mean(margins > 0)),
            cover_prob=cover_prob,
            spread_ci_50=(
                float(jnp.percentile(margins, 25)),
                float(jnp.percentile(margins, 75)),
            ),
            spread_ci_80=(
                float(jnp.percentile(margins, 10)),
                float(jnp.percentile(margins, 90)),
            ),
            spread_ci_95=(
                float(jnp.percentile(margins, 2.5)),
                float(jnp.percentile(margins, 97.5)),
            ),
            total_ci_50=(
                float(jnp.percentile(totals, 25)),
                float(jnp.percentile(totals, 75)),
            ),
            total_ci_80=(
                float(jnp.percentile(totals, 10)),
                float(jnp.percentile(totals, 90)),
            ),
            total_ci_95=(
                float(jnp.percentile(totals, 2.5)),
                float(jnp.percentile(totals, 97.5)),
            ),
            n_simulations=n_sims,
            model_version=self.config.version,
        )

    def get_team_ratings(self) -> dict[int, dict]:
        """
        Get posterior mean ratings for all teams (Final Week).

        Returns:
            Dict mapping team_id to {off, def, tempo, overall}
        """
        if self.posterior_samples is None:
            raise RuntimeError("Model not fitted")

        # Use FINAL week ratings
        off_mean = jnp.mean(self.posterior_samples["off_ability_final"], axis=0)
        def_mean = jnp.mean(self.posterior_samples["def_ability_final"], axis=0)
        tempo_mean = jnp.mean(self.posterior_samples["team_tempo"], axis=0)

        # Reverse the team_idx_map
        idx_to_team = {v: k for k, v in self.team_idx_map.items()}

        ratings = {}
        for idx in range(len(off_mean)):
            team_id = idx_to_team.get(idx)
            if team_id is not None:
                ratings[team_id] = {
                    "adj_off": float(off_mean[idx]),
                    "adj_def": float(def_mean[idx]),
                    "tempo": float(tempo_mean[idx]),
                    "adj_em": float(off_mean[idx] - def_mean[idx]),
                }

        return ratings


def create_gpu_predictor(config: Optional[GPUModelConfig] = None) -> HierarchicalNCAABModel:
    """Create GPU-accelerated predictor with default config."""
    return HierarchicalNCAABModel(config or GPUModelConfig())


# Fallback for when GPU not available
class CPUFallbackModel:
    """
    CPU fallback when JAX/NumPyro not available.

    Uses the baseline predictor with bootstrap uncertainty.
    """

    def __init__(self):
        from packages.models.core_bayes import BaselinePredictor
        self.baseline = BaselinePredictor()
        logger.warning("Using CPU fallback model (GPU not available)")

    def predict_game_mc(
        self,
        home_team_id: int,
        away_team_id: int,
        home_adj_off: float,
        home_adj_def: float,
        home_tempo: float,
        away_adj_off: float,
        away_adj_def: float,
        away_tempo: float,
        is_neutral: bool = False,
        market_spread: Optional[float] = None,
        game_id: int = 0,
        n_sims: int = 10000,
    ) -> MCPrediction:
        """Generate prediction with bootstrap uncertainty."""

        # Base prediction
        pred = self.baseline.predict_game(
            home_adj_off=home_adj_off,
            home_adj_def=home_adj_def,
            home_adj_tempo=home_tempo,
            away_adj_off=away_adj_off,
            away_adj_def=away_adj_def,
            away_adj_tempo=away_tempo,
            game_id=game_id,
            is_neutral=is_neutral,
            market_spread=market_spread,
        )

        # Bootstrap uncertainty (simple approach)
        np.random.seed(42)
        margins = np.random.normal(pred.spread, pred.spread_std, n_sims)
        totals = np.random.normal(pred.total, 13.0, n_sims)

        cover_prob = None
        if market_spread is not None:
            cover_prob = float(np.mean(margins > market_spread))

        return MCPrediction(
            game_id=game_id,
            spread_mean=pred.spread,
            spread_std=pred.spread_std,
            spread_median=float(np.median(margins)),
            total_mean=pred.total,
            total_std=13.0,
            total_median=float(np.median(totals)),
            home_win_prob=pred.home_win_prob,
            cover_prob=cover_prob,
            spread_ci_50=(
                float(np.percentile(margins, 25)),
                float(np.percentile(margins, 75)),
            ),
            spread_ci_80=(
                float(np.percentile(margins, 10)),
                float(np.percentile(margins, 90)),
            ),
            spread_ci_95=(
                float(np.percentile(margins, 2.5)),
                float(np.percentile(margins, 97.5)),
            ),
            total_ci_50=(
                float(np.percentile(totals, 25)),
                float(np.percentile(totals, 75)),
            ),
            total_ci_80=(
                float(np.percentile(totals, 10)),
                float(np.percentile(totals, 90)),
            ),
            total_ci_95=(
                float(np.percentile(totals, 2.5)),
                float(np.percentile(totals, 97.5)),
            ),
            n_simulations=n_sims,
            model_version="v0.1.0-cpu-fallback",
        )
