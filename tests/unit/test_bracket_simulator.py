"""Tests for dynamic bracket simulation behavior."""

from packages.simulation.bracket_simulator import BracketSimulator, GamePrediction


def _game_prediction(
    slot_id: int,
    round_value: int,
    team_a_id: int,
    team_b_id: int,
    team_a_win_prob: float,
) -> GamePrediction:
    return GamePrediction(
        slot_id=slot_id,
        round=round_value,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        team_a_seed=1,
        team_b_seed=16,
        team_a_name=f"Team {team_a_id}",
        team_b_name=f"Team {team_b_id}",
        team_a_win_prob=team_a_win_prob,
        team_b_win_prob=1.0 - team_a_win_prob,
        proj_spread=0.0,
        proj_total=140.0,
        upset_prob=0.0,
    )


def test_simulator_reprices_emergent_matchup_after_upset() -> None:
    bracket_slots = [
        {
            "slot_id": 1,
            "year": 2026,
            "round": 1,
            "team_a_id": 10,
            "team_b_id": 20,
            "seed_a": 1,
            "seed_b": 16,
            "next_slot_id": 3,
            "victor_game_position": "Top",
        },
        {
            "slot_id": 2,
            "year": 2026,
            "round": 1,
            "team_a_id": 30,
            "team_b_id": 40,
            "seed_a": 8,
            "seed_b": 9,
            "next_slot_id": 3,
            "victor_game_position": "Bottom",
        },
        {
            "slot_id": 3,
            "year": 2026,
            "round": 2,
            "team_a_id": None,
            "team_b_id": None,
            "seed_a": None,
            "seed_b": None,
            "next_slot_id": None,
            "victor_game_position": None,
        },
    ]
    game_predictions = {
        1: _game_prediction(1, 1, 10, 20, 0.0),
        2: _game_prediction(2, 1, 30, 40, 1.0),
    }

    calls: list[tuple[int, int, int]] = []

    def matchup_predictor(team_a_id: int, team_b_id: int, slot: dict) -> GamePrediction:
        calls.append((team_a_id, team_b_id, int(slot["slot_id"])))
        return _game_prediction(int(slot["slot_id"]), int(slot["round"]), team_a_id, team_b_id, 1.0)

    simulator = BracketSimulator(num_simulations=1, use_gpu=False, seed=7)
    result = simulator.simulate(
        bracket_slots=bracket_slots,
        game_predictions=game_predictions,
        year=2026,
        matchup_predictor=matchup_predictor,
        team_names={20: "Upset Winner", 30: "Favorite"},
    )

    assert calls == [(20, 30, 3)]
    assert result.championship_probs[20] == 1.0
    assert result.championship_probs[30] == 0.0
