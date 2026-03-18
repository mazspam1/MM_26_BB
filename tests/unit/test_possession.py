"""
Unit tests for possession calculations.

Tests Dean Oliver formula, tempo calculations, and edge cases.
"""

import pytest

from packages.common.schemas import BoxScore
from packages.features.possession import (
    FTA_COEFFICIENT,
    calculate_game_possessions,
    calculate_possessions_from_boxscore,
    calculate_possessions_from_stats,
    calculate_tempo,
    expected_game_possessions,
    points_per_100_possessions,
    points_per_possession,
    possessions_per_100,
    tempo_context_adjustment,
)


class TestPossessionFormula:
    """Tests for Dean Oliver possession formula."""

    def test_basic_possession_calculation(self):
        """Test basic possession formula: FGA - OR + TO + 0.475*FTA."""
        # 60 FGA, 10 OR, 12 TO, 20 FTA
        # = 60 - 10 + 12 + 0.475*20 = 71.5
        result = calculate_possessions_from_stats(
            field_goals_attempted=60,
            offensive_rebounds=10,
            turnovers=12,
            free_throws_attempted=20,
        )
        assert abs(result - 71.5) < 0.001

    def test_fta_coefficient(self):
        """Test that FTA coefficient is 0.475."""
        assert FTA_COEFFICIENT == 0.475

    def test_zero_stats(self):
        """Test with all zeros - should return 0."""
        result = calculate_possessions_from_stats(
            field_goals_attempted=0,
            offensive_rebounds=0,
            turnovers=0,
            free_throws_attempted=0,
        )
        assert result == 0.0

    def test_negative_result_clamped(self):
        """Test that negative possessions are clamped to 0."""
        # Extreme case: more OR than FGA and no turnovers
        result = calculate_possessions_from_stats(
            field_goals_attempted=10,
            offensive_rebounds=20,
            turnovers=0,
            free_throws_attempted=0,
        )
        assert result == 0.0  # Should be clamped, not negative

    def test_typical_game_possessions(self):
        """Test with typical D1 game stats."""
        # Typical team: 55 FGA, 8 OR, 12 TO, 18 FTA
        result = calculate_possessions_from_stats(
            field_goals_attempted=55,
            offensive_rebounds=8,
            turnovers=12,
            free_throws_attempted=18,
        )
        # 55 - 8 + 12 + 8.55 = 67.55
        expected = 55 - 8 + 12 + 0.475 * 18
        assert abs(result - expected) < 0.001
        # Should be in typical range (60-75)
        assert 60 < result < 75


class TestBoxScorePossessions:
    """Tests for possession calculation from BoxScore objects."""

    @pytest.fixture
    def typical_boxscore(self):
        """Create typical D1 box score."""
        return BoxScore(
            game_id=1,
            team_id=100,
            is_home=True,
            field_goals_made=25,
            field_goals_attempted=58,
            three_pointers_made=7,
            three_pointers_attempted=18,
            free_throws_made=14,
            free_throws_attempted=18,
            offensive_rebounds=9,
            defensive_rebounds=22,
            turnovers=11,
            assists=14,
            steals=6,
            blocks=2,
            personal_fouls=16,
            points=71,
        )

    def test_boxscore_possessions(self, typical_boxscore):
        """Test possessions from BoxScore object."""
        result = calculate_possessions_from_boxscore(typical_boxscore)
        expected = 58 - 9 + 11 + 0.475 * 18
        assert abs(result - expected) < 0.001


class TestGamePossessions:
    """Tests for calculating total game possessions."""

    @pytest.fixture
    def home_box(self):
        return BoxScore(
            game_id=1,
            team_id=100,
            is_home=True,
            field_goals_made=25,
            field_goals_attempted=55,
            three_pointers_made=7,
            three_pointers_attempted=15,
            free_throws_made=12,
            free_throws_attempted=16,
            offensive_rebounds=8,
            defensive_rebounds=24,
            turnovers=10,
            assists=15,
            steals=5,
            blocks=3,
            personal_fouls=14,
            points=69,
        )

    @pytest.fixture
    def away_box(self):
        return BoxScore(
            game_id=1,
            team_id=200,
            is_home=False,
            field_goals_made=23,
            field_goals_attempted=52,
            three_pointers_made=6,
            three_pointers_attempted=18,
            free_throws_made=10,
            free_throws_attempted=14,
            offensive_rebounds=6,
            defensive_rebounds=22,
            turnovers=12,
            assists=12,
            steals=4,
            blocks=2,
            personal_fouls=16,
            points=62,
        )

    def test_average_method(self, home_box, away_box):
        """Test average method for game possessions."""
        home_poss = calculate_possessions_from_boxscore(home_box)
        away_poss = calculate_possessions_from_boxscore(away_box)
        expected = (home_poss + away_poss) / 2

        result = calculate_game_possessions(home_box, away_box, method="average")
        assert abs(result - expected) < 0.001

    def test_home_method(self, home_box, away_box):
        """Test home team method."""
        expected = calculate_possessions_from_boxscore(home_box)
        result = calculate_game_possessions(home_box, away_box, method="home")
        assert abs(result - expected) < 0.001

    def test_away_method(self, home_box, away_box):
        """Test away team method."""
        expected = calculate_possessions_from_boxscore(away_box)
        result = calculate_game_possessions(home_box, away_box, method="away")
        assert abs(result - expected) < 0.001

    def test_invalid_method_raises(self, home_box, away_box):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError):
            calculate_game_possessions(home_box, away_box, method="invalid")


class TestTempo:
    """Tests for tempo calculations."""

    def test_basic_tempo(self):
        """Test tempo = possessions per 40 minutes."""
        # If 68 possessions in 40 minutes, tempo = 68
        result = calculate_tempo(possessions=68.0, game_minutes=40.0)
        assert result == 68.0

    def test_overtime_tempo_adjustment(self):
        """Test tempo adjustment for overtime game."""
        # 85 possessions in 45 minutes (OT game)
        # Tempo = 85 * (40/45) = 75.56
        result = calculate_tempo(possessions=85.0, game_minutes=45.0)
        expected = 85.0 * (40.0 / 45.0)
        assert abs(result - expected) < 0.01

    def test_double_overtime(self):
        """Test tempo for double OT game."""
        # 95 possessions in 50 minutes
        result = calculate_tempo(possessions=95.0, game_minutes=50.0)
        expected = 95.0 * (40.0 / 50.0)
        assert abs(result - expected) < 0.01

    def test_zero_minutes_raises(self):
        """Test that zero minutes raises error."""
        with pytest.raises(ValueError):
            calculate_tempo(possessions=68.0, game_minutes=0.0)


class TestExpectedPossessions:
    """Tests for projecting expected game possessions."""

    def test_harmonic_mean(self):
        """Test harmonic mean of tempos."""
        # Two teams: 70 and 66 tempo
        # Harmonic mean = 2*70*66/(70+66) = 67.94
        result = expected_game_possessions(
            home_tempo=70.0,
            away_tempo=66.0,
            method="harmonic",
        )
        expected = 2 * 70.0 * 66.0 / (70.0 + 66.0)
        assert abs(result - expected) < 0.01

    def test_arithmetic_mean(self):
        """Test arithmetic mean of tempos."""
        result = expected_game_possessions(
            home_tempo=70.0,
            away_tempo=66.0,
            method="arithmetic",
        )
        assert result == 68.0

    def test_geometric_mean(self):
        """Test geometric mean of tempos."""
        import math

        result = expected_game_possessions(
            home_tempo=70.0,
            away_tempo=66.0,
            method="geometric",
        )
        expected = math.sqrt(70.0 * 66.0)
        assert abs(result - expected) < 0.01

    def test_zero_tempo_uses_league_avg(self):
        """Test that zero tempo returns league average."""
        result = expected_game_possessions(
            home_tempo=0.0,
            away_tempo=66.0,
            league_avg_tempo=68.0,
            method="harmonic",
        )
        assert result == 68.0

    def test_extreme_tempos(self):
        """Test with extreme tempo values."""
        # Very fast (75) vs very slow (60)
        result = expected_game_possessions(
            home_tempo=75.0,
            away_tempo=60.0,
            method="harmonic",
        )
        # Harmonic mean is always <= arithmetic mean
        assert result < (75.0 + 60.0) / 2
        assert result > 60.0


class TestTempoContextAdjustment:
    """Tests for context-based tempo adjustments."""

    def test_neutral_site_adjustment(self):
        """Test that neutral site slightly reduces pace."""
        base = 68.0
        result = tempo_context_adjustment(base, is_neutral_site=True)
        assert result < base
        assert result == base * 0.98

    def test_back_to_back_adjustment(self):
        """Test back-to-back game adjustment."""
        base = 68.0
        result = tempo_context_adjustment(base, home_rest_days=0)
        assert result < base

    def test_both_teams_back_to_back(self):
        """Test when both teams on back-to-back."""
        base = 68.0
        result = tempo_context_adjustment(base, home_rest_days=0, away_rest_days=0)
        # Both adjustments stack
        assert result == base * 0.97 * 0.97

    def test_no_adjustment_normal_rest(self):
        """Test no adjustment with normal rest."""
        base = 68.0
        result = tempo_context_adjustment(base, home_rest_days=3, away_rest_days=2)
        assert result == base

    def test_altitude_adjustment(self):
        """Test high altitude adjustment."""
        base = 68.0
        result = tempo_context_adjustment(base, altitude_diff=6000.0)
        assert result < base
        assert result == base * 0.99


class TestEfficiencyRates:
    """Tests for per-possession rate calculations."""

    def test_possessions_per_100(self):
        """Test converting stat to per-100-possessions."""
        # 15 assists in 65 possessions
        result = possessions_per_100(15.0, 65.0)
        expected = (15.0 / 65.0) * 100
        assert abs(result - expected) < 0.01

    def test_zero_possessions_returns_zero(self):
        """Test zero possessions returns zero rate."""
        result = possessions_per_100(15.0, 0.0)
        assert result == 0.0

    def test_points_per_possession(self):
        """Test raw PPP calculation."""
        result = points_per_possession(75, 68.0)
        expected = 75 / 68.0
        assert abs(result - expected) < 0.001

    def test_points_per_100(self):
        """Test points per 100 possessions."""
        result = points_per_100_possessions(75, 68.0)
        expected = (75 / 68.0) * 100
        assert abs(result - expected) < 0.1
        # Typical D1 range is 85-115
        assert 85 < result < 130
