"""Render a filled March Madness bracket as an HTML file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROUND_ORDER = [
    ("First Four", 0),
    ("Round of 64", 1),
    ("Round of 32", 2),
    ("Sweet 16", 3),
    ("Elite 8", 4),
    ("Final Four", 5),
    ("Championship", 6),
]


def _games_by_round(games: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_round: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        by_round.setdefault(game["round"], []).append(game)
    return by_round


def _team_cell(name: str, seed: int, score: float, p10: float, p90: float, is_winner: bool) -> str:
    winner_cls = " winner" if is_winner else ""
    return (
        f"<div class='team{winner_cls}'>"
        f"<span class='seed'>{seed}</span>"
        f"<span class='name'>{name}</span>"
        f"<span class='score'>{score:.0f}</span>"
        f"<span class='band'>({p10:.0f}-{p90:.0f})</span>"
        f"</div>"
    )


def render_bracket_html(
    games: list[dict[str, Any]], champion: str, output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    by_round = _games_by_round(games)

    region_slots: dict[str, list[dict[str, Any]]] = {
        "East": [],
        "West": [],
        "South": [],
        "Midwest": [],
        "First Four": [],
        "National": [],
    }
    for game in games:
        if game["round"] == 0:
            region_slots["First Four"].append(game)
        elif game["round"] >= 5:
            region_slots["National"].append(game)
        elif game["round"] <= 4:
            # Infer region from game slot_id ordering in NCAA bracket
            slot = game["slot_id"]
            if slot >= 4 and slot <= 15:
                region_slots["East"].append(game)
            elif slot >= 16 and slot <= 27:
                region_slots["West"].append(game)
            elif slot >= 28 and slot <= 39:
                region_slots["South"].append(game)
            elif slot >= 40 and slot <= 51:
                region_slots["Midwest"].append(game)
            else:
                # Fallback: put by round
                pass

    # Build round columns
    round_columns = []
    for round_name, round_num in ROUND_ORDER:
        round_games = by_round.get(round_num, [])
        if not round_games:
            continue
        round_columns.append((round_name, round_num, round_games))

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #0a0e1a; 
        color: #e8e8e8; 
        padding: 20px;
        min-width: 3200px;
    }
    h1 { 
        text-align: center; 
        font-size: 28px; 
        margin: 10px 0 5px;
        color: #fff;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 14px;
        margin-bottom: 20px;
    }
    .champion-banner {
        text-align: center;
        font-size: 22px;
        margin: 15px 0 25px;
        padding: 15px;
        background: linear-gradient(135deg, #1a3a6e, #2a5a9e);
        border-radius: 10px;
        border: 2px solid #ffd700;
    }
    .champion-banner .trophy { font-size: 28px; }
    .champion-banner .champ-name { color: #ffd700; font-weight: 700; font-size: 26px; }
    .bracket-container {
        display: flex;
        justify-content: center;
        gap: 0;
        overflow-x: auto;
    }
    .round-column {
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        padding: 10px 8px;
        min-width: 220px;
    }
    .round-header {
        text-align: center;
        font-size: 13px;
        font-weight: 700;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 8px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 10px;
    }
    .matchup {
        display: flex;
        flex-direction: column;
        margin: 6px 0;
        position: relative;
    }
    .team {
        display: flex;
        align-items: center;
        padding: 5px 8px;
        background: #141929;
        border: 1px solid #2a2f45;
        border-radius: 4px;
        margin: 1px 0;
        font-size: 12px;
        transition: background 0.2s;
    }
    .team.winner {
        background: #1a2f50;
        border-color: #3a6a9e;
        font-weight: 600;
    }
    .team .seed {
        width: 22px;
        text-align: center;
        font-size: 11px;
        color: #888;
        flex-shrink: 0;
    }
    .team .name {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        padding: 0 6px;
    }
    .team .score {
        font-weight: 700;
        color: #fff;
        width: 28px;
        text-align: right;
        flex-shrink: 0;
    }
    .team .band {
        font-size: 10px;
        color: #666;
        width: 70px;
        text-align: right;
        flex-shrink: 0;
        padding-left: 6px;
    }
    .vs {
        text-align: center;
        font-size: 10px;
        color: #555;
        padding: 1px 0;
    }
    .connector {
        position: absolute;
        right: -12px;
        top: 50%;
        width: 12px;
        border-top: 1px solid #333;
    }
    .region-label {
        writing-mode: vertical-rl;
        text-orientation: mixed;
        font-size: 16px;
        font-weight: 700;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 0 8px;
        display: flex;
        align-items: center;
    }
    .first-four-section {
        margin-bottom: 20px;
        padding: 15px;
        background: #111525;
        border-radius: 8px;
        border: 1px solid #2a2f45;
    }
    .first-four-title {
        font-size: 14px;
        font-weight: 700;
        color: #888;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .ff-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .ff-game {
        flex: 1;
        min-width: 300px;
    }
    .prob-bar {
        height: 3px;
        background: #1a1f35;
        border-radius: 2px;
        margin-top: 2px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        background: #3a6a9e;
        border-radius: 2px;
    }
    """

    # Build HTML
    html_parts = [
        f"<html><head><meta charset='utf-8'><title>2026 March Madness Bracket</title><style>{css}</style></head><body>"
    ]
    html_parts.append("<h1>2026 NCAA March Madness</h1>")
    html_parts.append(
        f"<div class='subtitle'>Projected Bracket · Model v2.0.0-tournament · 100,000 Monte Carlo Simulations</div>"
    )
    html_parts.append(
        f"<div class='champion-banner'><span class='trophy'>🏆</span> Projected Champion: <span class='champ-name'>{champion}</span></div>"
    )

    # First Four section
    ff_games = by_round.get(0, [])
    if ff_games:
        html_parts.append(
            "<div class='first-four-section'><div class='first-four-title'>First Four</div><div class='ff-grid'>"
        )
        for game in ff_games:
            winner_id = game["likely_winner_id"]
            html_parts.append("<div class='ff-game'>")
            for team_key, opp_key in [("team_a", "team_b")]:
                a_winner = game["team_a_id"] == winner_id
                b_winner = game["team_b_id"] == winner_id
                html_parts.append(
                    _team_cell(
                        game["team_a_name"],
                        game["team_a_seed"],
                        game["team_a_score"],
                        game["team_a_score_p10"],
                        game["team_a_score_p90"],
                        a_winner,
                    )
                )
                html_parts.append(f"<div class='vs'>vs → {game['likely_winner_name']}</div>")
                html_parts.append(
                    _team_cell(
                        game["team_b_name"],
                        game["team_b_seed"],
                        game["team_b_score"],
                        game["team_b_score_p10"],
                        game["team_b_score_p90"],
                        b_winner,
                    )
                )
            html_parts.append("</div>")
        html_parts.append("</div></div>")

    # Main bracket grid
    html_parts.append("<div class='bracket-container'>")

    # We'll render round by round, pairing matchups
    # Round of 64 through Championship
    main_rounds = [(name, num) for name, num in ROUND_ORDER if num >= 1]
    num_rounds = len(main_rounds)

    for col_idx, (round_name, round_num) in enumerate(main_rounds):
        round_games = sorted(by_round.get(round_num, []), key=lambda g: g.get("game_in_round", 0))
        if not round_games:
            continue

        html_parts.append(
            f"<div class='round-column' style='flex-grow:{max(1, num_rounds - col_idx)}'>"
        )
        html_parts.append(f"<div class='round-header'>{round_name}</div>")

        for game in round_games:
            winner_id = game["likely_winner_id"]
            a_winner = game["team_a_id"] == winner_id
            b_winner = game["team_b_id"] == winner_id
            prob_a = f"{game['team_a_win_prob']:.0%}"
            prob_b = f"{game['team_b_win_prob']:.0%}"

            html_parts.append("<div class='matchup'>")
            html_parts.append("<div class='connector'></div>")
            html_parts.append(
                _team_cell(
                    game["team_a_name"],
                    game["team_a_seed"],
                    game["team_a_score"],
                    game["team_a_score_p10"],
                    game["team_a_score_p90"],
                    a_winner,
                )
            )
            html_parts.append(
                f"<div class='prob-bar'><div class='prob-fill' style='width:{game['team_a_win_prob'] * 100:.0f}%'></div></div>"
            )
            html_parts.append(
                _team_cell(
                    game["team_b_name"],
                    game["team_b_seed"],
                    game["team_b_score"],
                    game["team_b_score_p10"],
                    game["team_b_score_p90"],
                    b_winner,
                )
            )
            html_parts.append(
                f"<div class='prob-bar'><div class='prob-fill' style='width:{game['team_b_win_prob'] * 100:.0f}%'></div></div>"
            )
            html_parts.append("</div>")

        html_parts.append("</div>")

    html_parts.append("</div>")
    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path
