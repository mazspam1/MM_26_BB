"""
Unit tests for Pydantic schemas.

Tests data validation, computed fields, and edge cases.
"""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from packages.common.schemas import (
    BoxScore,
    Game,
    LineSnapshot,
    PredictionRow,
    SeasonPhase,
    Team,
    TeamStrength,
    VenueType,
)


class TestTeamSchema:
    """Tests for Team schema."""

    def test_valid_team(self):
        """Test creating a valid team."""
        team = Team(
            team_id=150,
            name="Duke Blue Devils",
            abbreviation="DUKE",
            conference="ACC",
        )
        assert team.team_id == 150
        assert team.name == "Duke Blue Devils"
        assert team.abbreviation == "DUKE"
        assert team.conference == "ACC"

    def test_invalid_team_id_rejected(self):
        """Test that negative team IDs are rejected."""
        with pytest.raises(ValidationError):
            Team(
                team_id=-1,
                name="Duke",
                abbreviation="DUKE",
                conference="ACC",
            )

    def test_zero_team_id_rejected(self):
        """Test that zero team ID is rejected."""
        with pytest.raises(ValidationError):
            Team(
                team_id=0,
                name="Duke",
                abbreviation="DUKE",
                conference="ACC",
            )

    def test_team_is_frozen(self):
        """Test that Team is immutable (frozen)."""
        team = Team(
            team_id=150,
            name="Duke",
            abbreviation="DUKE",
            conference="ACC",
        )
        with pytest.raises(ValidationError):
            team.name = "North Carolina"

    def test_empty_name_rejected(self):
        """Test that empty team name is rejected."""
        with pytest.raises(ValidationError):
            Team(
                team_id=150,
                name="",
                abbreviation="DUKE",
                conference="ACC",
            )


class TestTeamStrengthSchema:
    """Tests for TeamStrength schema."""

    def test_valid_team_strength(self):
        """Test creating valid team strength."""
        strength = TeamStrength(
            team_id=150,
            as_of_date=date(2024, 12, 22),
            season=2024,
            adj_offensive_efficiency=115.5,
            adj_defensive_efficiency=95.0,
            adj_tempo=68.5,
            games_played=10,
        )
        assert strength.adj_offensive_efficiency == 115.5
        assert strength.adj_defensive_efficiency == 95.0
        assert strength.adj_tempo == 68.5

    def test_efficiency_margin_computed(self):
        """Test that efficiency margin is computed correctly."""
        strength = TeamStrength(
            team_id=150,
            as_of_date=date(2024, 12, 22),
            season=2024,
            adj_offensive_efficiency=115.0,
            adj_defensive_efficiency=95.0,
            adj_tempo=68.0,
            games_played=10,
        )
        assert strength.adj_efficiency_margin == 20.0

    def test_invalid_efficiency_rejected(self):
        """Test that out-of-range efficiency is rejected."""
        with pytest.raises(ValidationError):
            TeamStrength(
                team_id=150,
                as_of_date=date(2024, 12, 22),
                season=2024,
                adj_offensive_efficiency=200.0,  # Too high
                adj_defensive_efficiency=95.0,
                adj_tempo=68.0,
                games_played=10,
            )


class TestGameSchema:
    """Tests for Game schema."""

    def test_valid_game(self):
        """Test creating a valid game."""
        game = Game(
            game_id=401638245,
            season=2024,
            game_date=date(2024, 12, 22),
            home_team_id=150,
            away_team_id=2390,
        )
        assert game.game_id == 401638245
        assert game.status == "scheduled"
        assert game.actual_spread is None

    def test_game_with_scores(self):
        """Test game with final scores."""
        game = Game(
            game_id=401638245,
            season=2024,
            game_date=date(2024, 12, 22),
            home_team_id=150,
            away_team_id=2390,
            home_score=85,
            away_score=78,
            status="final",
        )
        assert game.actual_spread == 7.0
        assert game.actual_total == 163
        assert game.is_completed is True

    def test_game_spread_calculation(self):
        """Test spread calculation (home - away)."""
        game = Game(
            game_id=1,
            season=2024,
            game_date=date(2024, 12, 22),
            home_team_id=100,
            away_team_id=200,
            home_score=70,
            away_score=80,
            status="final",
        )
        assert game.actual_spread == -10.0  # Home lost by 10

    def test_neutral_site_game(self):
        """Test neutral site configuration."""
        game = Game(
            game_id=1,
            season=2024,
            game_date=date(2024, 12, 22),
            home_team_id=100,
            away_team_id=200,
            neutral_site=True,
            venue_type=VenueType.NEUTRAL,
        )
        assert game.neutral_site is True
        assert game.venue_type == VenueType.NEUTRAL


class TestBoxScoreSchema:
    """Tests for BoxScore schema with computed stats."""

    @pytest.fixture
    def sample_boxscore(self):
        """Create a sample box score for testing."""
        return BoxScore(
            game_id=1,
            team_id=150,
            is_home=True,
            field_goals_made=25,
            field_goals_attempted=55,
            three_pointers_made=8,
            three_pointers_attempted=20,
            free_throws_made=12,
            free_throws_attempted=15,
            offensive_rebounds=10,
            defensive_rebounds=25,
            turnovers=12,
            assists=15,
            steals=5,
            blocks=3,
            personal_fouls=18,
            points=70,
        )

    def test_possession_calculation(self, sample_boxscore):
        """Test Dean Oliver possession formula."""
        # Possessions = FGA - OR + TO + 0.475*FTA
        # = 55 - 10 + 12 + 0.475*15 = 64.125
        expected = 55 - 10 + 12 + 0.475 * 15
        assert abs(sample_boxscore.possessions - expected) < 0.001

    def test_efg_calculation(self, sample_boxscore):
        """Test effective field goal percentage."""
        # eFG% = (FGM + 0.5*3PM) / FGA = (25 + 4) / 55 = 0.527
        expected = (25 + 0.5 * 8) / 55
        assert abs(sample_boxscore.efg_pct - expected) < 0.001

    def test_turnover_percentage(self, sample_boxscore):
        """Test turnover percentage calculation."""
        expected = 12 / sample_boxscore.possessions
        assert abs(sample_boxscore.to_pct - expected) < 0.001

    def test_free_throw_rate(self, sample_boxscore):
        """Test free throw rate (FTA/FGA)."""
        expected = 15 / 55
        assert abs(sample_boxscore.ftr - expected) < 0.001

    def test_orb_percentage(self, sample_boxscore):
        """Test offensive rebound percentage."""
        expected = 10 / (10 + 25)
        assert abs(sample_boxscore.orb_pct - expected) < 0.001

    def test_total_rebounds(self, sample_boxscore):
        """Test total rebounds computed field."""
        assert sample_boxscore.total_rebounds == 35

    def test_points_per_possession(self, sample_boxscore):
        """Test points per 100 possessions."""
        expected = (70 / sample_boxscore.possessions) * 100
        assert abs(sample_boxscore.points_per_possession - expected) < 0.1

    def test_zero_fga_edge_case(self):
        """Test handling of zero field goal attempts."""
        box = BoxScore(
            game_id=1,
            team_id=150,
            is_home=True,
            field_goals_made=0,
            field_goals_attempted=0,
            three_pointers_made=0,
            three_pointers_attempted=0,
            free_throws_made=0,
            free_throws_attempted=0,
            offensive_rebounds=0,
            defensive_rebounds=0,
            turnovers=0,
            assists=0,
            steals=0,
            blocks=0,
            personal_fouls=0,
            points=0,
        )
        assert box.efg_pct == 0.0
        assert box.ftr == 0.0


class TestLineSnapshotSchema:
    """Tests for LineSnapshot schema."""

    def test_valid_line_snapshot(self):
        """Test creating a valid line snapshot."""
        snapshot = LineSnapshot(
            game_id=401638245,
            bookmaker="draftkings",
            snapshot_timestamp=datetime(2024, 12, 22, 10, 0, 0),
            snapshot_type="current",
            spread_home=-5.5,
            spread_home_price=-110,
            spread_away=5.5,
            spread_away_price=-110,
            total_line=145.5,
            over_price=-110,
            under_price=-110,
        )
        assert snapshot.spread_home == -5.5
        assert snapshot.total_line == 145.5

    def test_snapshot_type_validation(self):
        """Test that invalid snapshot types are rejected."""
        with pytest.raises(ValidationError):
            LineSnapshot(
                game_id=1,
                bookmaker="test",
                snapshot_timestamp=datetime.now(),
                snapshot_type="invalid",  # Not open/current/close
            )

    def test_spread_range_validation(self):
        """Test that extreme spreads are rejected."""
        with pytest.raises(ValidationError):
            LineSnapshot(
                game_id=1,
                bookmaker="test",
                snapshot_timestamp=datetime.now(),
                snapshot_type="current",
                spread_home=-100.0,  # Too extreme
            )


class TestPredictionRowSchema:
    """Tests for PredictionRow schema."""

    def test_valid_prediction(self):
        """Test creating a valid prediction."""
        pred = PredictionRow(
            game_id=401638245,
            prediction_timestamp=datetime(2024, 12, 22, 6, 0, 0),
            model_version="v0.1.0",
            proj_home_score=78.5,
            proj_away_score=72.0,
            proj_spread=6.5,
            proj_total=150.5,
            proj_possessions=68.0,
            home_win_prob=0.72,
            spread_ci_50_lower=3.0,
            spread_ci_50_upper=10.0,
            spread_ci_80_lower=-1.0,
            spread_ci_80_upper=14.0,
            spread_ci_95_lower=-6.0,
            spread_ci_95_upper=19.0,
            total_ci_50_lower=143.0,
            total_ci_50_upper=158.0,
            total_ci_80_lower=136.0,
            total_ci_80_upper=165.0,
            total_ci_95_lower=130.0,
            total_ci_95_upper=171.0,
        )
        assert pred.proj_spread == 6.5
        assert pred.home_win_prob == 0.72

    def test_model_version_format(self):
        """Test that model version must follow semver pattern."""
        with pytest.raises(ValidationError):
            PredictionRow(
                game_id=1,
                prediction_timestamp=datetime.now(),
                model_version="1.0.0",  # Missing 'v' prefix
                proj_home_score=75.0,
                proj_away_score=70.0,
                proj_spread=5.0,
                proj_total=145.0,
                proj_possessions=68.0,
                home_win_prob=0.6,
                spread_ci_50_lower=0.0,
                spread_ci_50_upper=10.0,
                spread_ci_80_lower=-5.0,
                spread_ci_80_upper=15.0,
                spread_ci_95_lower=-10.0,
                spread_ci_95_upper=20.0,
                total_ci_50_lower=140.0,
                total_ci_50_upper=150.0,
                total_ci_80_lower=135.0,
                total_ci_80_upper=155.0,
                total_ci_95_lower=130.0,
                total_ci_95_upper=160.0,
            )

    def test_win_prob_range(self):
        """Test that win probability must be 0-1."""
        with pytest.raises(ValidationError):
            PredictionRow(
                game_id=1,
                prediction_timestamp=datetime.now(),
                model_version="v0.1.0",
                proj_home_score=75.0,
                proj_away_score=70.0,
                proj_spread=5.0,
                proj_total=145.0,
                proj_possessions=68.0,
                home_win_prob=1.5,  # Invalid
                spread_ci_50_lower=0.0,
                spread_ci_50_upper=10.0,
                spread_ci_80_lower=-5.0,
                spread_ci_80_upper=15.0,
                spread_ci_95_lower=-10.0,
                spread_ci_95_upper=20.0,
                total_ci_50_lower=140.0,
                total_ci_50_upper=150.0,
                total_ci_80_lower=135.0,
                total_ci_80_upper=155.0,
                total_ci_95_lower=130.0,
                total_ci_95_upper=160.0,
            )
