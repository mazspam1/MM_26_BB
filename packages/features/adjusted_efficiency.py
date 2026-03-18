"""
KenPom-style adjusted efficiency calculations.

Implements opponent-adjusted offensive and defensive efficiency ratings
through iterative adjustment for schedule strength.

References:
- KenPom: https://kenpom.com/blog/ratings-explanation/
- Bart Torvik AMA: https://www.reddit.com/r/CollegeBasketball/comments/dp9fzk/
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import structlog

from packages.common.schemas import BoxScore, TeamStrength
from packages.features.possession import (
    calculate_possessions_from_boxscore,
    points_per_100_possessions,
)

logger = structlog.get_logger()

# D1 league average efficiency (points per 100 possessions)
LEAGUE_AVG_EFFICIENCY = 100.0

# Default home court advantage in efficiency points
DEFAULT_HCA = 3.5


@dataclass
class GameEfficiency:
    """Efficiency metrics for a single game."""

    game_id: int
    team_id: int
    opponent_id: int
    is_home: bool
    possessions: float
    raw_off_eff: float  # Points per 100 poss scored
    raw_def_eff: float  # Points per 100 poss allowed
    tempo: float  # Possessions per 40 min


@dataclass
class AdjustedRatings:
    """Opponent-adjusted efficiency ratings."""

    team_id: int
    games_played: int
    adj_off_eff: float  # Adjusted offensive efficiency
    adj_def_eff: float  # Adjusted defensive efficiency
    adj_tempo: float  # Adjusted tempo
    adj_em: float  # Adjusted efficiency margin (off - def)
    raw_off_eff: float  # Raw (unadjusted) offensive
    raw_def_eff: float  # Raw defensive
    sos_off: float  # Strength of schedule faced (offensive)
    sos_def: float  # Strength of schedule faced (defensive)


def calculate_game_efficiency(
    team_box: BoxScore,
    opponent_box: BoxScore,
) -> GameEfficiency:
    """
    Calculate efficiency metrics for a team in a single game.

    Args:
        team_box: Team's box score
        opponent_box: Opponent's box score

    Returns:
        GameEfficiency dataclass
    """
    # Use average of both teams' possession estimates
    team_poss = calculate_possessions_from_boxscore(team_box)
    opp_poss = calculate_possessions_from_boxscore(opponent_box)
    possessions = (team_poss + opp_poss) / 2

    # Calculate efficiencies (per 100 possessions)
    raw_off_eff = points_per_100_possessions(team_box.points, possessions)
    raw_def_eff = points_per_100_possessions(opponent_box.points, possessions)

    # Tempo (possessions per 40 minutes)
    tempo = possessions  # Already per 40 min in regulation game

    return GameEfficiency(
        game_id=team_box.game_id,
        team_id=team_box.team_id,
        opponent_id=opponent_box.team_id,
        is_home=team_box.is_home,
        possessions=possessions,
        raw_off_eff=raw_off_eff,
        raw_def_eff=raw_def_eff,
        tempo=tempo,
    )


class AdjustedEfficiencyCalculator:
    """
    Calculate opponent-adjusted efficiency ratings.

    Uses iterative adjustment where each team's efficiency is
    adjusted based on the quality of opponents faced.
    """

    def __init__(
        self,
        league_avg: float = LEAGUE_AVG_EFFICIENCY,
        home_court_advantage: float = DEFAULT_HCA,
        iterations: int = 10,
    ):
        """
        Initialize calculator.

        Args:
            league_avg: League average efficiency
            home_court_advantage: Home court advantage in points
            iterations: Number of adjustment iterations
        """
        self.league_avg = league_avg
        self.hca = home_court_advantage
        self.iterations = iterations
        self._games: dict[int, list[GameEfficiency]] = {}
        self._team_ids: set[int] = set()

    def add_game(self, team_eff: GameEfficiency) -> None:
        """
        Add a game's efficiency data.

        Args:
            team_eff: GameEfficiency for one team
        """
        if team_eff.team_id not in self._games:
            self._games[team_eff.team_id] = []

        self._games[team_eff.team_id].append(team_eff)
        self._team_ids.add(team_eff.team_id)
        self._team_ids.add(team_eff.opponent_id)

    def add_games_from_boxscores(
        self,
        home_box: BoxScore,
        away_box: BoxScore,
    ) -> None:
        """
        Add a game from both teams' box scores.

        Args:
            home_box: Home team's box score
            away_box: Away team's box score
        """
        home_eff = calculate_game_efficiency(home_box, away_box)
        away_eff = calculate_game_efficiency(away_box, home_box)

        self.add_game(home_eff)
        self.add_game(away_eff)

    def _get_raw_ratings(self) -> dict[int, dict]:
        """Calculate raw (unadjusted) ratings for all teams."""
        ratings = {}

        for team_id in self._team_ids:
            games = self._games.get(team_id, [])

            if not games:
                # No games played - use league average
                ratings[team_id] = {
                    "raw_off": self.league_avg,
                    "raw_def": self.league_avg,
                    "raw_tempo": 68.0,
                    "games": 0,
                }
                continue

            # Simple average of raw efficiencies
            raw_off = np.mean([g.raw_off_eff for g in games])
            raw_def = np.mean([g.raw_def_eff for g in games])
            raw_tempo = np.mean([g.tempo for g in games])

            ratings[team_id] = {
                "raw_off": raw_off,
                "raw_def": raw_def,
                "raw_tempo": raw_tempo,
                "games": len(games),
            }

        return ratings

    def calculate_adjusted_ratings(self) -> dict[int, AdjustedRatings]:
        """
        Calculate opponent-adjusted ratings through iteration.

        Returns:
            Dictionary mapping team_id to AdjustedRatings
        """
        # Start with raw ratings
        raw = self._get_raw_ratings()

        # Initialize adjusted ratings
        adjusted = {}
        for team_id, data in raw.items():
            adjusted[team_id] = {
                "adj_off": data["raw_off"],
                "adj_def": data["raw_def"],
                "adj_tempo": data["raw_tempo"],
                "games": data["games"],
            }

        # Iterate to adjust for opponent strength
        for iteration in range(self.iterations):
            new_adjusted = {}

            for team_id in self._team_ids:
                games = self._games.get(team_id, [])

                if not games:
                    new_adjusted[team_id] = adjusted[team_id]
                    continue

                # Calculate adjusted efficiency for each game
                adj_off_values = []
                adj_def_values = []
                adj_tempo_values = []

                for game in games:
                    opp_id = game.opponent_id

                    if opp_id not in adjusted:
                        continue

                    opp = adjusted[opp_id]

                    # Adjust offensive efficiency
                    # Good opposing defense makes our offense look worse
                    # So we adjust UP if opponent has good D (low adj_def)
                    opp_def_quality = self.league_avg - opp["adj_def"]
                    adj_off = game.raw_off_eff + opp_def_quality

                    # Adjust defensive efficiency
                    # Good opposing offense makes our defense look worse
                    # So we adjust DOWN if opponent has good O (high adj_off)
                    opp_off_quality = opp["adj_off"] - self.league_avg
                    adj_def = game.raw_def_eff - opp_off_quality

                    # Apply home court adjustment
                    if game.is_home:
                        # Home team's raw efficiency is inflated
                        adj_off -= self.hca / 2
                        adj_def += self.hca / 2
                    else:
                        # Away team's raw efficiency is deflated
                        adj_off += self.hca / 2
                        adj_def -= self.hca / 2

                    adj_off_values.append(adj_off)
                    adj_def_values.append(adj_def)
                    adj_tempo_values.append(game.tempo)

                if adj_off_values:
                    new_adjusted[team_id] = {
                        "adj_off": np.mean(adj_off_values),
                        "adj_def": np.mean(adj_def_values),
                        "adj_tempo": np.mean(adj_tempo_values),
                        "games": len(adj_off_values),
                    }
                else:
                    new_adjusted[team_id] = adjusted[team_id]

            valid_teams = [t for t in new_adjusted.values() if t.get("games", 0) > 0]
            if valid_teams:
                mean_off = sum(t["adj_off"] for t in valid_teams) / len(valid_teams)
                mean_def = sum(t["adj_def"] for t in valid_teams) / len(valid_teams)
                off_delta = mean_off - self.league_avg
                def_delta = mean_def - self.league_avg
                for team_id in new_adjusted:
                    new_adjusted[team_id]["adj_off"] -= off_delta
                    new_adjusted[team_id]["adj_def"] -= def_delta

            adjusted = new_adjusted

        # Calculate strength of schedule
        sos = self._calculate_sos(adjusted)

        # Build final results
        results = {}
        for team_id in self._team_ids:
            data = adjusted.get(team_id, {})
            raw_data = raw.get(team_id, {})
            sos_data = sos.get(team_id, {"sos_off": self.league_avg, "sos_def": self.league_avg})

            results[team_id] = AdjustedRatings(
                team_id=team_id,
                games_played=data.get("games", 0),
                adj_off_eff=data.get("adj_off", self.league_avg),
                adj_def_eff=data.get("adj_def", self.league_avg),
                adj_tempo=data.get("adj_tempo", 68.0),
                adj_em=data.get("adj_off", self.league_avg) - data.get("adj_def", self.league_avg),
                raw_off_eff=raw_data.get("raw_off", self.league_avg),
                raw_def_eff=raw_data.get("raw_def", self.league_avg),
                sos_off=sos_data["sos_off"],
                sos_def=sos_data["sos_def"],
            )

        return results

    def _calculate_sos(self, adjusted: dict[int, dict]) -> dict[int, dict]:
        """Calculate strength of schedule for each team."""
        sos = {}

        for team_id in self._team_ids:
            games = self._games.get(team_id, [])

            if not games:
                sos[team_id] = {"sos_off": self.league_avg, "sos_def": self.league_avg}
                continue

            # Average opponent ratings
            opp_off_ratings = []
            opp_def_ratings = []

            for game in games:
                opp_id = game.opponent_id
                if opp_id in adjusted:
                    opp_off_ratings.append(adjusted[opp_id]["adj_off"])
                    opp_def_ratings.append(adjusted[opp_id]["adj_def"])

            if opp_off_ratings:
                sos[team_id] = {
                    "sos_off": np.mean(opp_off_ratings),
                    "sos_def": np.mean(opp_def_ratings),
                }
            else:
                sos[team_id] = {"sos_off": self.league_avg, "sos_def": self.league_avg}

        return sos


def project_game_score(
    home_adj_off: float,
    home_adj_def: float,
    home_adj_tempo: float,
    away_adj_off: float,
    away_adj_def: float,
    away_adj_tempo: float,
    home_court_advantage: float = DEFAULT_HCA,
    is_neutral: bool = False,
    league_avg: float = LEAGUE_AVG_EFFICIENCY,
) -> tuple[float, float, float]:
    """
    Project game score using Torvik-style methodology.

    Formula:
    Expected_PPP_home = (Home_AdjO + Away_AdjD - LeagueAvg) / LeagueAvg
    Expected_PPP_away = (Away_AdjO + Home_AdjD - LeagueAvg) / LeagueAvg

    Args:
        home_adj_off: Home team adjusted offensive efficiency
        home_adj_def: Home team adjusted defensive efficiency
        home_adj_tempo: Home team adjusted tempo
        away_adj_off: Away team adjusted offensive efficiency
        away_adj_def: Away team adjusted defensive efficiency
        away_adj_tempo: Away team adjusted tempo
        home_court_advantage: HCA in points
        is_neutral: Whether game is at neutral site
        league_avg: League average efficiency

    Returns:
        Tuple of (home_score, away_score, expected_possessions)
    """
    # Expected possessions (harmonic mean of tempos)
    expected_poss = 2 * home_adj_tempo * away_adj_tempo / (home_adj_tempo + away_adj_tempo)

    # Expected PPP using additive model
    home_ppp = (home_adj_off + away_adj_def - league_avg) / league_avg
    away_ppp = (away_adj_off + home_adj_def - league_avg) / league_avg

    # Apply home court advantage
    if not is_neutral:
        hca_per_poss = home_court_advantage / expected_poss
        home_ppp += hca_per_poss / 2
        away_ppp -= hca_per_poss / 2

    # Calculate expected scores
    home_score = home_ppp * expected_poss
    away_score = away_ppp * expected_poss

    return home_score, away_score, expected_poss


def ratings_to_team_strength(
    rating: AdjustedRatings,
    as_of_date: date,
    season: int,
) -> TeamStrength:
    """
    Convert AdjustedRatings to TeamStrength schema.

    Args:
        rating: AdjustedRatings object
        as_of_date: Date of the ratings
        season: Season year

    Returns:
        TeamStrength object
    """
    return TeamStrength(
        team_id=rating.team_id,
        as_of_date=as_of_date,
        season=season,
        adj_offensive_efficiency=rating.adj_off_eff,
        adj_defensive_efficiency=rating.adj_def_eff,
        adj_tempo=rating.adj_tempo,
        games_played=rating.games_played,
    )
