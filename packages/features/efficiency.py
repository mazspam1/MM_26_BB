"""
Efficiency metrics for NCAA basketball.

Raw and adjusted efficiency calculations following KenPom methodology.

References:
- KenPom: https://kenpom.com/blog/ratings-explanation/
- Bart Torvik: https://www.reddit.com/r/CollegeBasketball/comments/dp9fzk/
"""

from typing import Optional

import structlog

from packages.common.schemas import BoxScore
from packages.features.possession import (
    calculate_possessions_from_boxscore,
    points_per_100_possessions,
)

logger = structlog.get_logger()

# League average efficiency (D1 men's basketball)
LEAGUE_AVG_EFFICIENCY = 100.0


def calculate_raw_offensive_efficiency(box: BoxScore) -> float:
    """
    Calculate raw offensive efficiency (points per 100 possessions).

    Args:
        box: BoxScore object

    Returns:
        Raw offensive efficiency (points per 100 possessions)
    """
    possessions = calculate_possessions_from_boxscore(box)
    return points_per_100_possessions(box.points, possessions)


def calculate_raw_defensive_efficiency(
    opponent_box: BoxScore,
) -> float:
    """
    Calculate raw defensive efficiency (opponent points per 100 possessions).

    Lower is better for defense.

    Args:
        opponent_box: Opponent's BoxScore

    Returns:
        Raw defensive efficiency
    """
    possessions = calculate_possessions_from_boxscore(opponent_box)
    return points_per_100_possessions(opponent_box.points, possessions)


def calculate_efficiency_margin(
    offensive_efficiency: float,
    defensive_efficiency: float,
) -> float:
    """
    Calculate efficiency margin (net rating).

    Args:
        offensive_efficiency: Points per 100 possessions scored
        defensive_efficiency: Points per 100 possessions allowed

    Returns:
        Efficiency margin (positive = outscoring opponents)
    """
    return offensive_efficiency - defensive_efficiency


def project_points_from_efficiency(
    offensive_efficiency: float,
    defensive_efficiency: float,
    expected_possessions: float,
    home_court_adjustment: float = 0.0,
) -> tuple[float, float]:
    """
    Project expected points for a matchup.

    Args:
        offensive_efficiency: Team's adjusted offensive efficiency
        defensive_efficiency: Opponent's adjusted defensive efficiency
        expected_possessions: Expected game possessions
        home_court_adjustment: HCA in efficiency points (typically ~3.5)

    Returns:
        Tuple of (expected_points_for, expected_points_against)
    """
    # Adjust for opponent strength (KenPom/Torvik-style additive)
    # Expected PPP = (AdjO + AdjD - LeagueAvg) / LeagueAvg
    expected_off_ppp = (
        offensive_efficiency + defensive_efficiency - LEAGUE_AVG_EFFICIENCY
    ) / LEAGUE_AVG_EFFICIENCY

    # Add home court
    expected_off_ppp += home_court_adjustment / LEAGUE_AVG_EFFICIENCY

    expected_points = expected_off_ppp * expected_possessions

    return max(0.0, expected_points), 0.0  # Second value unused in this context


def matchup_expected_scores(
    home_adj_off: float,
    home_adj_def: float,
    away_adj_off: float,
    away_adj_def: float,
    expected_possessions: float,
    home_court_advantage: float = 3.5,
    is_neutral: bool = False,
) -> tuple[float, float]:
    """
    Project expected scores for a matchup using KenPom/Torvik methodology.

    Formula (per Torvik AMA):
    Expected_Home_PPP = (Home_AdjO + Away_AdjD - LeagueAvg) / LeagueAvg + HCA
    Expected_Away_PPP = (Away_AdjO + Home_AdjD - LeagueAvg) / LeagueAvg

    Args:
        home_adj_off: Home team adjusted offensive efficiency
        home_adj_def: Home team adjusted defensive efficiency
        away_adj_off: Away team adjusted offensive efficiency
        away_adj_def: Away team adjusted defensive efficiency
        expected_possessions: Expected game possessions
        home_court_advantage: HCA in points (typically 3.0-4.0)
        is_neutral: Whether game is at neutral site

    Returns:
        Tuple of (expected_home_score, expected_away_score)
    """
    # Calculate expected PPP for each team
    # Home offense vs Away defense
    home_expected_ppp = (
        home_adj_off + away_adj_def - LEAGUE_AVG_EFFICIENCY
    ) / LEAGUE_AVG_EFFICIENCY

    # Away offense vs Home defense
    away_expected_ppp = (
        away_adj_off + home_adj_def - LEAGUE_AVG_EFFICIENCY
    ) / LEAGUE_AVG_EFFICIENCY

    # Apply home court advantage (convert to PPP terms)
    if not is_neutral:
        hca_ppp = home_court_advantage / expected_possessions
        home_expected_ppp += hca_ppp / 2
        away_expected_ppp -= hca_ppp / 2

    # Calculate expected scores
    home_expected_score = home_expected_ppp * expected_possessions
    away_expected_score = away_expected_ppp * expected_possessions

    return home_expected_score, away_expected_score


def calculate_win_probability(
    projected_margin: float,
    spread_std: float = 11.0,
) -> float:
    """
    Calculate win probability from projected margin.

    Uses normal CDF with empirical spread standard deviation.

    Args:
        projected_margin: Projected point differential (positive = home favored)
        spread_std: Standard deviation of spread outcomes (~11 for NCAAB)

    Returns:
        Win probability for home team (0-1)
    """
    from scipy.stats import norm

    # P(Home wins) = P(Margin > 0)
    # Using normal approximation
    return float(norm.cdf(projected_margin / spread_std))


def opponent_quality_weight(
    opponent_efficiency_margin: float,
    games_played: int = 1,
) -> float:
    """
    Calculate weight for opponent quality in adjusted calculations.

    Better opponents get more weight in strength of schedule.

    Args:
        opponent_efficiency_margin: Opponent's net rating
        games_played: Number of games opponent has played

    Returns:
        Quality weight (0.5 - 1.5 range)
    """
    # Base weight of 1.0
    weight = 1.0

    # Adjust based on opponent strength
    if opponent_efficiency_margin > 10:
        weight = 1.3
    elif opponent_efficiency_margin > 5:
        weight = 1.15
    elif opponent_efficiency_margin < -10:
        weight = 0.7
    elif opponent_efficiency_margin < -5:
        weight = 0.85

    # Reduce weight for opponents with few games (uncertain rating)
    if games_played < 5:
        weight *= 0.9
    elif games_played < 10:
        weight *= 0.95

    return weight


class EfficiencyCalculator:
    """
    Calculate team efficiencies from box score data.

    This class maintains state for iterative efficiency calculations,
    allowing for opponent-adjusted ratings.
    """

    def __init__(self, league_avg: float = LEAGUE_AVG_EFFICIENCY):
        self.league_avg = league_avg
        self._team_efficiencies: dict[int, dict] = {}

    def add_game(
        self,
        team_id: int,
        opponent_id: int,
        team_box: BoxScore,
        opponent_box: BoxScore,
        is_home: bool,
    ) -> None:
        """
        Add a game result for efficiency tracking.

        Args:
            team_id: Team ID
            opponent_id: Opponent ID
            team_box: Team's box score
            opponent_box: Opponent's box score
            is_home: Whether team was home
        """
        possessions = calculate_possessions_from_boxscore(team_box)
        off_eff = points_per_100_possessions(team_box.points, possessions)
        def_eff = points_per_100_possessions(opponent_box.points, possessions)

        if team_id not in self._team_efficiencies:
            self._team_efficiencies[team_id] = {
                "games": [],
                "raw_off": [],
                "raw_def": [],
                "opponents": [],
                "is_home": [],
            }

        self._team_efficiencies[team_id]["games"].append(team_box.game_id)
        self._team_efficiencies[team_id]["raw_off"].append(off_eff)
        self._team_efficiencies[team_id]["raw_def"].append(def_eff)
        self._team_efficiencies[team_id]["opponents"].append(opponent_id)
        self._team_efficiencies[team_id]["is_home"].append(is_home)

    def get_raw_efficiency(self, team_id: int) -> Optional[dict]:
        """
        Get raw (unadjusted) efficiency for a team.

        Returns:
            Dictionary with raw offensive and defensive efficiency
        """
        if team_id not in self._team_efficiencies:
            return None

        data = self._team_efficiencies[team_id]

        if not data["raw_off"]:
            return None

        return {
            "games_played": len(data["games"]),
            "raw_off": sum(data["raw_off"]) / len(data["raw_off"]),
            "raw_def": sum(data["raw_def"]) / len(data["raw_def"]),
        }

    def calculate_adjusted_efficiencies(
        self,
        iterations: int = 10,
        home_court_adjustment: float = 3.5,
    ) -> dict[int, dict]:
        """
        Calculate opponent-adjusted efficiencies through iteration.

        Uses an iterative approach where each team's efficiency is
        adjusted based on the strength of their opponents.

        Args:
            iterations: Number of adjustment iterations
            home_court_adjustment: HCA in points

        Returns:
            Dictionary mapping team_id to adjusted efficiencies
        """
        # Start with raw efficiencies
        adjusted = {}
        for team_id in self._team_efficiencies:
            raw = self.get_raw_efficiency(team_id)
            if raw:
                adjusted[team_id] = {
                    "adj_off": raw["raw_off"],
                    "adj_def": raw["raw_def"],
                    "games_played": raw["games_played"],
                }

        # Iterate to adjust for opponent strength
        for _ in range(iterations):
            new_adjusted = {}

            for team_id, team_data in self._team_efficiencies.items():
                if team_id not in adjusted:
                    continue

                # Calculate weighted average adjustments
                off_adjustments = []
                def_adjustments = []

                for i, opp_id in enumerate(team_data["opponents"]):
                    if opp_id not in adjusted:
                        continue

                    opp = adjusted[opp_id]
                    is_home = team_data["is_home"][i]

                    # Adjust for opponent strength
                    # Good defense makes your offense look worse
                    opp_def_quality = self.league_avg - opp["adj_def"]
                    off_adj = team_data["raw_off"][i] + opp_def_quality

                    # Good offense makes your defense look worse
                    opp_off_quality = opp["adj_off"] - self.league_avg
                    def_adj = team_data["raw_def"][i] - opp_off_quality

                    # Home court adjustment
                    if is_home:
                        off_adj -= home_court_adjustment / 2
                        def_adj += home_court_adjustment / 2
                    else:
                        off_adj += home_court_adjustment / 2
                        def_adj -= home_court_adjustment / 2

                    off_adjustments.append(off_adj)
                    def_adjustments.append(def_adj)

                if off_adjustments:
                    new_adjusted[team_id] = {
                        "adj_off": sum(off_adjustments) / len(off_adjustments),
                        "adj_def": sum(def_adjustments) / len(def_adjustments),
                        "games_played": adjusted[team_id]["games_played"],
                    }
                else:
                    new_adjusted[team_id] = adjusted[team_id]

            adjusted = new_adjusted

        # Add efficiency margin
        for team_id in adjusted:
            adjusted[team_id]["adj_em"] = (
                adjusted[team_id]["adj_off"] - adjusted[team_id]["adj_def"]
            )

        return adjusted
