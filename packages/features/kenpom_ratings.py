"""
KenPom-style iterative adjusted efficiency ratings.

Implements proper least-squares adjustment for opponent strength using:
- Iterative rating adjustment until convergence
- Recency weighting (Torvik-style decay)
- Home court advantage adjustment during rating calculation
- Four Factors integration

NO MOCK DATA. NO FALLBACKS. REAL CALCULATIONS ONLY.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import math
import structlog

import pandas as pd
import numpy as np

from packages.common.database import get_connection
from packages.common.season import season_start_date

logger = structlog.get_logger()

# Constants
LEAGUE_AVG_EFFICIENCY = 100.0  # D1 average points per 100 possessions
LEAGUE_AVG_TEMPO = 68.0  # D1 average possessions per 40 minutes
HOME_COURT_ADVANTAGE = 3.5  # Points (will be refined per-conference)
CONVERGENCE_THRESHOLD = 0.05  # Stop when max change < this
MAX_ITERATIONS = 100
DAMPING_FACTOR = 0.35  # Damp updates to stabilize convergence


def _weighted_avg(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    """Compute weighted average, returning None if no weight."""
    total_weight = weights.sum()
    if total_weight <= 0:
        return None
    return float(np.average(values, weights=weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted standard deviation."""
    total_weight = weights.sum()
    if total_weight <= 0:
        return 0.0
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return float(math.sqrt(variance))


def load_division_i_team_ids() -> set[int]:
    """Load the current Division I modeling universe from the teams table."""
    with get_connection() as conn:
        rows = conn.execute("SELECT team_id FROM teams").fetchall()
    return {int(row[0]) for row in rows}


@dataclass
class TeamRatings:
    """Complete adjusted ratings for a team."""

    team_id: int
    adj_off: float  # Adjusted Offensive Efficiency (pts/100 poss)
    adj_def: float  # Adjusted Defensive Efficiency
    adj_tempo: float  # Adjusted Tempo (poss/40 min)
    adj_em: float  # Efficiency Margin (adj_off - adj_def)

    # Four Factors (offensive)
    adj_efg: float
    adj_tov: float
    adj_orb: float
    adj_ftr: float

    # Four Factors (defensive)
    adj_efg_def: float
    adj_tov_def: float
    adj_drb: float  # Opponent's ORB% = 1 - our DRB%
    adj_ftr_def: float

    # Metadata
    games_played: int
    sos_off: float  # Strength of schedule (offensive)
    sos_def: float  # Strength of schedule (defensive)
    as_of_date: date
    home_off_delta: float = 0.0  # Home off - overall off (raw)
    home_def_delta: float = 0.0  # Home def - overall def (raw)
    away_off_delta: float = 0.0  # Away off - overall off (raw)
    away_def_delta: float = 0.0  # Away def - overall def (raw)
    home_games_played: int = 0
    away_games_played: int = 0
    off_std: float = 0.0
    def_std: float = 0.0
    tempo_std: float = 0.0


def calculate_recency_weight(game_date: date, as_of_date: date) -> float:
    """
    Calculate recency weight using Torvik-style decay.

    - First 40 days: Full weight (1.0)
    - Days 41-80: Linear decay from 1.0 to 0.6
    - Beyond 80 days: Constant 0.6
    """
    days_ago = (as_of_date - game_date).days

    if days_ago <= 40:
        return 1.0
    elif days_ago <= 80:
        return 1.0 - 0.01 * (days_ago - 40)
    else:
        return 0.6


def calculate_raw_ratings(
    team_stats: pd.DataFrame, as_of_date: date, use_recency_weights: bool = True
) -> dict[int, dict]:
    """
    Calculate raw (unadjusted) ratings from team game stats.

    Returns dict mapping team_id to {off, def, tempo, games, ...}
    """
    raw_ratings = {}

    for team_id, team_games in team_stats.groupby("team_id"):
        if len(team_games) == 0:
            continue

        # Calculate weights
        if use_recency_weights:
            weights = (
                team_games["game_date"]
                .apply(lambda d: calculate_recency_weight(d, as_of_date))
                .values
            )
        else:
            weights = np.ones(len(team_games))

        total_weight = weights.sum()
        if total_weight == 0:
            continue

        # Weighted averages
        off_rating = float(np.average(team_games["off_rating"].values, weights=weights))
        def_rating = float(np.average(team_games["def_rating"].values, weights=weights))
        tempo = float(np.average(team_games["possessions"].values, weights=weights)) * (40 / 40)

        # Four Factors (weighted)
        off_efg = np.average(team_games["off_efg"].values, weights=weights)
        off_tov = np.average(team_games["off_tov"].values, weights=weights)
        off_orb = np.average(team_games["off_orb"].values, weights=weights)
        off_ftr = np.average(team_games["off_ftr"].values, weights=weights)

        def_efg = np.average(team_games["def_efg"].values, weights=weights)
        def_tov = np.average(team_games["def_tov"].values, weights=weights)
        def_orb = np.average(team_games["def_orb"].values, weights=weights)
        def_ftr = np.average(team_games["def_ftr"].values, weights=weights)

        # Home/away splits (exclude neutral)
        home_mask = ((team_games["is_home"] == True) & (~team_games["is_neutral"])).to_numpy()
        away_mask = ((team_games["is_home"] == False) & (~team_games["is_neutral"])).to_numpy()
        home_games = int(home_mask.sum())
        away_games = int(away_mask.sum())

        home_off = _weighted_avg(team_games["off_rating"].values[home_mask], weights[home_mask])
        home_def = _weighted_avg(team_games["def_rating"].values[home_mask], weights[home_mask])
        away_off = _weighted_avg(team_games["off_rating"].values[away_mask], weights[away_mask])
        away_def = _weighted_avg(team_games["def_rating"].values[away_mask], weights[away_mask])

        home_off_delta = (home_off - off_rating) if home_off is not None else 0.0
        home_def_delta = (home_def - def_rating) if home_def is not None else 0.0
        away_off_delta = (away_off - off_rating) if away_off is not None else 0.0
        away_def_delta = (away_def - def_rating) if away_def is not None else 0.0

        # Volatility metrics
        off_std = _weighted_std(team_games["off_rating"].values, weights)
        def_std = _weighted_std(team_games["def_rating"].values, weights)
        tempo_std = _weighted_std(team_games["possessions"].values, weights)

        raw_ratings[team_id] = {
            "off": off_rating,
            "def": def_rating,
            "tempo": tempo,
            "games": len(team_games),
            "off_efg": off_efg,
            "off_tov": off_tov,
            "off_orb": off_orb,
            "off_ftr": off_ftr,
            "def_efg": def_efg,
            "def_tov": def_tov,
            "def_orb": def_orb,
            "def_ftr": def_ftr,
            "home_off_delta": home_off_delta,
            "home_def_delta": home_def_delta,
            "away_off_delta": away_off_delta,
            "away_def_delta": away_def_delta,
            "home_games": home_games,
            "away_games": away_games,
            "off_std": off_std,
            "def_std": def_std,
            "tempo_std": tempo_std,
        }

    return raw_ratings


def _opponent_adjust_four_factors(
    team_stats: pd.DataFrame,
    raw_ratings: dict[int, dict],
    as_of_date: date,
    use_recency_weights: bool,
) -> dict[int, dict]:
    """
    Single-pass opponent adjustment for Four Factors.

    For offensive stats: correct for opponent's defensive quality.
        adj = raw + (league_avg_def - opp_avg_def)
    For defensive stats: correct for opponent's offensive quality.
        adj = raw - (opp_avg_off - league_avg_off)

    This removes schedule-strength bias so that Four Factors represent
    true team skill, not inflated/deflated by who they played.
    """
    teams_with_games = [t for t in raw_ratings.values() if t["games"] > 0]
    if not teams_with_games:
        return {}

    n = len(teams_with_games)
    league_off = {
        "efg": sum(t["off_efg"] for t in teams_with_games) / n,
        "tov": sum(t["off_tov"] for t in teams_with_games) / n,
        "orb": sum(t["off_orb"] for t in teams_with_games) / n,
        "ftr": sum(t["off_ftr"] for t in teams_with_games) / n,
    }
    league_def = {
        "efg": sum(t["def_efg"] for t in teams_with_games) / n,
        "tov": sum(t["def_tov"] for t in teams_with_games) / n,
        "orb": sum(t["def_orb"] for t in teams_with_games) / n,
        "ftr": sum(t["def_ftr"] for t in teams_with_games) / n,
    }

    adjusted = {}

    for team_id, team_games in team_stats.groupby("team_id"):
        if team_id not in raw_ratings or len(team_games) == 0:
            continue

        if use_recency_weights:
            weights = (
                team_games["game_date"]
                .apply(lambda d: calculate_recency_weight(d, as_of_date))
                .values
            )
        else:
            weights = np.ones(len(team_games))

        # Accumulate per-game adjusted values
        keys = [
            "off_efg", "off_tov", "off_orb", "off_ftr",
            "def_efg", "def_tov", "def_orb", "def_ftr",
        ]
        vals = {k: [] for k in keys}
        valid_weights = []

        for idx, (_, game) in enumerate(team_games.iterrows()):
            opp_id = game["opponent_id"]
            if opp_id not in raw_ratings:
                continue

            opp = raw_ratings[opp_id]

            # Offensive: adj = raw + (league_avg_def - opp_avg_def)
            vals["off_efg"].append(game["off_efg"] + (league_def["efg"] - opp["def_efg"]))
            vals["off_tov"].append(game["off_tov"] + (league_def["tov"] - opp["def_tov"]))
            vals["off_orb"].append(game["off_orb"] + (league_def["orb"] - opp["def_orb"]))
            vals["off_ftr"].append(game["off_ftr"] + (league_def["ftr"] - opp["def_ftr"]))

            # Defensive: adj = raw - (opp_avg_off - league_avg_off)
            vals["def_efg"].append(game["def_efg"] - (opp["off_efg"] - league_off["efg"]))
            vals["def_tov"].append(game["def_tov"] - (opp["off_tov"] - league_off["tov"]))
            vals["def_orb"].append(game["def_orb"] - (opp["off_orb"] - league_off["orb"]))
            vals["def_ftr"].append(game["def_ftr"] - (opp["off_ftr"] - league_off["ftr"]))

            valid_weights.append(weights[idx])

        if not valid_weights:
            continue

        w = np.array(valid_weights)
        adjusted[team_id] = {
            "adj_efg": float(np.average(vals["off_efg"], weights=w)),
            "adj_tov": float(np.average(vals["off_tov"], weights=w)),
            "adj_orb": float(np.average(vals["off_orb"], weights=w)),
            "adj_ftr": float(np.average(vals["off_ftr"], weights=w)),
            "adj_efg_def": float(np.average(vals["def_efg"], weights=w)),
            "adj_tov_def": float(np.average(vals["def_tov"], weights=w)),
            "adj_orb_def": float(np.average(vals["def_orb"], weights=w)),
            "adj_ftr_def": float(np.average(vals["def_ftr"], weights=w)),
        }

    return adjusted


def calculate_adjusted_ratings(
    team_stats: pd.DataFrame,
    as_of_date: Optional[date] = None,
    use_recency_weights: bool = True,
    conference_hca: Optional[dict[int, float]] = None,
    season_start: Optional[date] = None,
    division_i_team_ids: Optional[set[int]] = None,
) -> dict[int, TeamRatings]:
    """
    Calculate KenPom-style adjusted efficiency ratings.

    Uses iterative least-squares adjustment:
    1. Start with raw ratings
    2. Adjust each team's ratings based on opponent strength
    3. Repeat until convergence

    Args:
        team_stats: DataFrame with game-level stats
        as_of_date: Date to calculate ratings as of (for recency weighting)
        use_recency_weights: Whether to apply Torvik-style decay
        conference_hca: Optional dict of team_id -> conference HCA adjustment
        division_i_team_ids: Optional explicit modeling universe. When omitted,
            the teams table is used if available.

    Returns:
        Dict mapping team_id to TeamRatings
    """
    if as_of_date is None:
        as_of_date = date.today()

    if season_start is None:
        season_start = season_start_date(as_of_date)

    if division_i_team_ids is None:
        try:
            division_i_team_ids = load_division_i_team_ids()
        except Exception as exc:
            logger.warning("Could not load Division I team universe", error=str(exc))
            division_i_team_ids = None

    logger.info(
        "Calculating adjusted ratings",
        as_of_date=as_of_date.isoformat(),
        season_start=season_start.isoformat(),
    )

    # Ensure game_date is a date type for comparison
    import pandas as pd

    if team_stats["game_date"].dtype == "object":
        team_stats["game_date"] = pd.to_datetime(team_stats["game_date"]).dt.date

    # Filter to games within current season window
    team_stats = team_stats[
        (team_stats["game_date"] <= as_of_date) & (team_stats["game_date"] >= season_start)
    ].copy()

    if division_i_team_ids:
        pre_filter_teams = int(team_stats["team_id"].nunique())
        pre_filter_rows = len(team_stats)
        team_stats = team_stats[team_stats["team_id"].isin(division_i_team_ids)].copy()
        logger.info(
            "Applied Division I filter to rating universe",
            teams_before=pre_filter_teams,
            teams_after=int(team_stats["team_id"].nunique()) if not team_stats.empty else 0,
            rows_before=pre_filter_rows,
            rows_after=len(team_stats),
        )

    if len(team_stats) == 0:
        logger.warning("No games found before as_of_date")
        return {}

    # Get unique teams
    all_teams = set(team_stats["team_id"].unique())

    # Initialize with raw ratings
    raw = calculate_raw_ratings(team_stats, as_of_date, use_recency_weights)

    # Initialize adjusted ratings
    adj_off = {t: raw[t]["off"] if t in raw else LEAGUE_AVG_EFFICIENCY for t in all_teams}
    adj_def = {t: raw[t]["def"] if t in raw else LEAGUE_AVG_EFFICIENCY for t in all_teams}
    adj_tempo = {t: raw[t]["tempo"] if t in raw else LEAGUE_AVG_TEMPO for t in all_teams}

    # Iterative adjustment
    for iteration in range(MAX_ITERATIONS):
        new_adj_off = {}
        new_adj_def = {}

        for team_id in all_teams:
            if team_id not in raw:
                new_adj_off[team_id] = LEAGUE_AVG_EFFICIENCY
                new_adj_def[team_id] = LEAGUE_AVG_EFFICIENCY
                continue

            team_games = team_stats[team_stats["team_id"] == team_id].copy()

            # Calculate weights
            if use_recency_weights:
                weights = (
                    team_games["game_date"]
                    .apply(lambda d: calculate_recency_weight(d, as_of_date))
                    .values
                )
            else:
                weights = np.ones(len(team_games))

            total_weight = weights.sum()
            if total_weight == 0:
                new_adj_off[team_id] = LEAGUE_AVG_EFFICIENCY
                new_adj_def[team_id] = LEAGUE_AVG_EFFICIENCY
                continue

            # Adjust each game's efficiency based on opponent
            adjusted_off_ratings = []
            adjusted_def_ratings = []

            for idx, (_, game) in enumerate(team_games.iterrows()):
                opp_id = game["opponent_id"]
                is_home = game["is_home"]
                is_neutral = game["is_neutral"]

                # Get opponent's current adjusted ratings
                opp_adj_def = adj_def.get(opp_id, LEAGUE_AVG_EFFICIENCY)
                opp_adj_off = adj_off.get(opp_id, LEAGUE_AVG_EFFICIENCY)

                # Home court adjustment (only if not neutral)
                hca = 0.0
                if not is_neutral:
                    if conference_hca and team_id in conference_hca:
                        hca = conference_hca[team_id]
                    else:
                        hca = HOME_COURT_ADVANTAGE
                    if not is_home:
                        hca = -hca

                # Adjust offensive rating for opponent defense quality
                # If opp has good defense (low adj_def), our raw off looks worse
                # Also adjust for HCA: home team offensive rating is inflated, so subtract
                raw_off = game["off_rating"]
                adj_game_off = raw_off + (LEAGUE_AVG_EFFICIENCY - opp_adj_def) - (hca / 2)
                adjusted_off_ratings.append((adj_game_off, weights[idx]))

                # Adjust defensive rating for opponent offense quality
                # Home team's defensive rating looks better due to crowd, so add penalty
                raw_def = game["def_rating"]
                adj_game_def = raw_def - (opp_adj_off - LEAGUE_AVG_EFFICIENCY) + (hca / 2)
                adjusted_def_ratings.append((adj_game_def, weights[idx]))

            # Weighted average of adjusted ratings
            new_adj_off[team_id] = sum(r * w for r, w in adjusted_off_ratings) / total_weight
            new_adj_def[team_id] = sum(r * w for r, w in adjusted_def_ratings) / total_weight

        # Damp updates to reduce oscillation
        for team_id in new_adj_off:
            prev_off = adj_off.get(team_id, LEAGUE_AVG_EFFICIENCY)
            prev_def = adj_def.get(team_id, LEAGUE_AVG_EFFICIENCY)
            new_adj_off[team_id] = prev_off + DAMPING_FACTOR * (new_adj_off[team_id] - prev_off)
            new_adj_def[team_id] = prev_def + DAMPING_FACTOR * (new_adj_def[team_id] - prev_def)

        # Normalize to keep league averages centered at LEAGUE_AVG_EFFICIENCY
        valid_teams = [team_id for team_id in new_adj_off.keys() if team_id in raw]
        if valid_teams:
            mean_off = sum(new_adj_off[t] for t in valid_teams) / len(valid_teams)
            mean_def = sum(new_adj_def[t] for t in valid_teams) / len(valid_teams)
            off_delta = mean_off - LEAGUE_AVG_EFFICIENCY
            def_delta = mean_def - LEAGUE_AVG_EFFICIENCY
            for team_id in new_adj_off:
                new_adj_off[team_id] -= off_delta
                new_adj_def[team_id] -= def_delta

        # Track convergence after damping + normalization
        max_change = 0.0
        for team_id in new_adj_off:
            prev_off = adj_off.get(team_id, LEAGUE_AVG_EFFICIENCY)
            prev_def = adj_def.get(team_id, LEAGUE_AVG_EFFICIENCY)
            max_change = max(
                max_change,
                abs(new_adj_off[team_id] - prev_off),
                abs(new_adj_def[team_id] - prev_def),
            )

        # Update ratings
        adj_off = new_adj_off
        adj_def = new_adj_def

        # Check convergence
        if max_change < CONVERGENCE_THRESHOLD:
            logger.info("Ratings converged", iterations=iteration + 1, max_change=max_change)
            converged = True
            break
    else:
        logger.warning(
            "Ratings did not converge", max_iterations=MAX_ITERATIONS, max_change=max_change
        )
        converged = False

    # Calculate strength of schedule with temporal decay
    sos = calculate_strength_of_schedule(
        team_stats, adj_off, adj_def, as_of_date, use_recency_weights
    )

    # Opponent-adjust Four Factors (single-pass using raw opponent averages)
    adjusted_ff = _opponent_adjust_four_factors(
        team_stats, raw, as_of_date, use_recency_weights
    )

    # Build final ratings
    final_ratings = {}
    for team_id in all_teams:
        if team_id not in raw:
            continue

        team_raw = raw[team_id]
        ff = adjusted_ff.get(team_id, {})

        final_ratings[team_id] = TeamRatings(
            team_id=team_id,
            adj_off=adj_off[team_id],
            adj_def=adj_def[team_id],
            adj_tempo=adj_tempo[team_id],
            adj_em=adj_off[team_id] - adj_def[team_id],
            adj_efg=ff.get("adj_efg", team_raw["off_efg"]),
            adj_tov=ff.get("adj_tov", team_raw["off_tov"]),
            adj_orb=ff.get("adj_orb", team_raw["off_orb"]),
            adj_ftr=ff.get("adj_ftr", team_raw["off_ftr"]),
            adj_efg_def=ff.get("adj_efg_def", team_raw["def_efg"]),
            adj_tov_def=ff.get("adj_tov_def", team_raw["def_tov"]),
            adj_drb=1.0 - ff.get("adj_orb_def", team_raw["def_orb"]),
            adj_ftr_def=ff.get("adj_ftr_def", team_raw["def_ftr"]),
            games_played=team_raw["games"],
            sos_off=sos.get(team_id, {}).get("off", 0.0),
            sos_def=sos.get(team_id, {}).get("def", 0.0),
            as_of_date=as_of_date,
            home_off_delta=team_raw.get("home_off_delta", 0.0),
            home_def_delta=team_raw.get("home_def_delta", 0.0),
            away_off_delta=team_raw.get("away_off_delta", 0.0),
            away_def_delta=team_raw.get("away_def_delta", 0.0),
            home_games_played=team_raw.get("home_games", 0),
            away_games_played=team_raw.get("away_games", 0),
            off_std=team_raw.get("off_std", 0.0),
            def_std=team_raw.get("def_std", 0.0),
            tempo_std=team_raw.get("tempo_std", 0.0),
        )

    logger.info("Adjusted ratings calculated", teams=len(final_ratings))
    return final_ratings


def calculate_strength_of_schedule(
    team_stats: pd.DataFrame,
    adj_off: dict[int, float],
    adj_def: dict[int, float],
    as_of_date: date,
    use_recency_weights: bool = True,
) -> dict[int, dict]:
    """
    Calculate strength of schedule for each team.

    SOS = weighted average of opponent ratings.
    """
    sos = {}

    for team_id, team_games in team_stats.groupby("team_id"):
        if len(team_games) == 0:
            continue

        # Get weights
        if use_recency_weights:
            weights = (
                team_games["game_date"]
                .apply(lambda d: calculate_recency_weight(d, as_of_date))
                .values
            )
        else:
            weights = np.ones(len(team_games))

        total_weight = weights.sum()
        if total_weight == 0:
            continue

        # Calculate opponent averages with temporal decay
        opp_off = []
        opp_def = []

        for idx, (_, game) in enumerate(team_games.iterrows()):
            opp_id = game["opponent_id"]
            # SOS temporal decay: earlier games contribute less to SOS
            days_ago = (as_of_date - game["game_date"]).days
            temporal_decay = 0.98 ** (days_ago / 7)  # 2% decay per week
            combined_weight = weights[idx] * temporal_decay

            opp_off.append((adj_off.get(opp_id, LEAGUE_AVG_EFFICIENCY), combined_weight))
            opp_def.append((adj_def.get(opp_id, LEAGUE_AVG_EFFICIENCY), combined_weight))

        total_combined_weight = sum(w for _, w in opp_off)
        sos[team_id] = {
            "off": sum(r * w for r, w in opp_off) / total_combined_weight - LEAGUE_AVG_EFFICIENCY,
            "def": sum(r * w for r, w in opp_def) / total_combined_weight - LEAGUE_AVG_EFFICIENCY,
        }

    return sos


def save_ratings_to_db(ratings: dict[int, TeamRatings]):
    """
    Save adjusted ratings to database.
    """
    from packages.common.database import get_connection

    logger.info("Saving ratings to database", teams=len(ratings))

    with get_connection() as conn:
        # Keep history; ensure table exists with expected schema.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_strengths (
                team_id INTEGER,
                as_of_date TEXT,
                adj_offensive_efficiency REAL,
                adj_defensive_efficiency REAL,
                adj_tempo REAL,
                adj_em REAL,
                off_efg REAL,
                off_tov REAL,
                off_orb REAL,
                off_ftr REAL,
                def_efg REAL,
                def_tov REAL,
                def_drb REAL,
                def_ftr REAL,
                games_played INTEGER,
                sos_off REAL,
                sos_def REAL,
                home_off_delta REAL,
                home_def_delta REAL,
                away_off_delta REAL,
                away_def_delta REAL,
                home_games_played INTEGER,
                away_games_played INTEGER,
                off_rating_std REAL,
                def_rating_std REAL,
                tempo_std REAL,
                PRIMARY KEY (team_id, as_of_date)
            )
        """)

        for team_id, r in ratings.items():
            # Convert numpy types to Python types for DuckDB
            conn.execute(
                """
                INSERT OR REPLACE INTO team_strengths (
                    team_id, as_of_date,
                    adj_offensive_efficiency, adj_defensive_efficiency, adj_tempo,
                    adj_em,
                    off_efg, off_tov, off_orb, off_ftr,
                    def_efg, def_tov, def_drb, def_ftr,
                    games_played, sos_off, sos_def,
                    home_off_delta, home_def_delta, away_off_delta, away_def_delta,
                    home_games_played, away_games_played,
                    off_rating_std, def_rating_std, tempo_std
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    int(team_id),
                    r.as_of_date.isoformat(),
                    float(r.adj_off),
                    float(r.adj_def),
                    float(r.adj_tempo),
                    float(r.adj_em),
                    float(r.adj_efg),
                    float(r.adj_tov),
                    float(r.adj_orb),
                    float(r.adj_ftr),
                    float(r.adj_efg_def),
                    float(r.adj_tov_def),
                    float(r.adj_drb),
                    float(r.adj_ftr_def),
                    int(r.games_played),
                    float(r.sos_off),
                    float(r.sos_def),
                    float(r.home_off_delta),
                    float(r.home_def_delta),
                    float(r.away_off_delta),
                    float(r.away_def_delta),
                    int(r.home_games_played),
                    int(r.away_games_played),
                    float(r.off_std),
                    float(r.def_std),
                    float(r.tempo_std),
                ),
            )

    logger.info("Ratings saved")


def get_team_ratings_from_db(as_of_date: Optional[date] = None) -> dict[int, TeamRatings]:
    """
    Load ratings from database.
    """
    from packages.common.database import get_connection

    if as_of_date is None:
        as_of_date = date.today()

    ratings = {}

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
            WHERE as_of_date = ?
        """,
            (as_of_date.isoformat(),),
        ).fetchall()

        for row in rows:
            ratings[row[0]] = TeamRatings(
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
                as_of_date=date.fromisoformat(row[1]),
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

    return ratings
