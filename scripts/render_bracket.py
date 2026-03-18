"""
Render tournament bracket JSON into a printable visual HTML bracket.
Looks like the official NCAA March Madness bracket with filled scores.

Usage:
    python scripts/render_bracket.py [--input predictions/tournament_2026/best_bracket.json]
"""

import json
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def render_bracket_html(games: list[dict], champion: str) -> str:
    """Generate a full NCAA-style bracket HTML."""

    # Organize games by round
    by_round = {}
    for g in games:
        r = g["round"]
        if r not in by_round:
            by_round[r] = []
        by_round[r].append(g)

    # Sort each round by game_in_round
    for r in by_round:
        by_round[r].sort(key=lambda x: x.get("game_in_round", 0))

    # Separate into regions (R64 has 8 games per region)
    # R64: 32 games total (8 per region)
    # We'll process them in order: East games 1-8, West 1-8, South 1-8, Midwest 1-8
    r64 = by_round.get(1, [])
    r32 = by_round.get(2, [])
    s16 = by_round.get(3, [])
    e8 = by_round.get(4, [])
    ff = by_round.get(5, [])
    champ = by_round.get(6, [])
    first_four = by_round.get(0, [])

    def game_html(g, round_num=1):
        """Render a single game box."""
        if not g:
            return '<div class="game-slot empty"></div>'

        ta_name = g.get("team_a_name", "")
        ta_seed = g.get("team_a_seed", "")
        ta_score = g.get("team_a_score", 0)
        tb_name = g.get("team_b_name", "")
        tb_seed = g.get("team_b_seed", "")
        tb_score = g.get("team_b_score", 0)
        winner = g.get("likely_winner_name", "")
        total = g.get("projected_total", 0)
        spread = g.get("projected_spread_team_a", 0)
        win_prob = g.get("team_a_win_prob", 0)
        upset = g.get("upset_prob", 0)

        ta_class = "winner" if ta_name == winner else ""
        tb_class = "winner" if tb_name == winner else ""

        score_a = f"{ta_score:.0f}" if ta_score else "-"
        score_b = f"{tb_score:.0f}" if tb_score else "-"

        upset_html = ""
        if round_num <= 1 and upset > 0.4:
            upset_html = f'<span class="upset-badge">UPSET</span>'

        return f"""
        <div class="game-slot">
            <div class="team-row {ta_class}">
                <span class="seed">#{ta_seed}</span>
                <span class="team-name">{ta_name}</span>
                <span class="score">{score_a}</span>
            </div>
            <div class="team-row {tb_class}">
                <span class="seed">#{tb_seed}</span>
                <span class="team-name">{tb_name}</span>
                <span class="score">{score_b}</span>
            </div>
            <div class="game-info">
                <span class="total">T: {total:.0f}</span>
                {upset_html}
            </div>
        </div>
        """

    # Build region sections (4 regions, each with R64->R32->S16->E8)
    regions = ["East", "West", "South", "Midwest"]
    region_html = ""

    for ri, region_name in enumerate(regions):
        r64_region = r64[ri * 8 : (ri + 1) * 8]
        r32_region = r32[ri * 4 : (ri + 1) * 4]
        s16_region = s16[ri * 2 : (ri + 1) * 2]
        e8_region = e8[ri : ri + 1] if ri < len(e8) else []

        region_html += f"""
        <div class="region">
            <h2 class="region-title">{region_name} Region</h2>
            <div class="region-bracket">
                <div class="round-col r64">
                    <div class="round-label">Round of 64</div>
                    {"".join(game_html(g, 1) for g in r64_region)}
                </div>
                <div class="round-col r32">
                    <div class="round-label">Round of 32</div>
                    {"".join(game_html(g, 2) for g in r32_region)}
                </div>
                <div class="round-col s16">
                    <div class="round-label">Sweet 16</div>
                    {"".join(game_html(g, 3) for g in s16_region)}
                </div>
                <div class="round-col e8">
                    <div class="round-label">Elite 8</div>
                    {"".join(game_html(g, 4) for g in e8_region)}
                </div>
            </div>
        </div>
        """

    # Build Final Four + Championship
    ff_html = ""
    for g in ff:
        ff_html += game_html(g, 5)

    champ_html = ""
    if champ:
        champ_html = game_html(champ[0], 6)

    # First Four
    ff4_html = ""
    for g in first_four:
        ff4_html += game_html(g, 0)

    # Champion name
    champ_name = champion if champion else "TBD"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2026 NCAA Tournament Bracket - Predicted</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f0;
            color: #1a1a2e;
            padding: 20px;
        }}
        
        .bracket-header {{
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        
        .bracket-header h1 {{
            font-size: 2.2rem;
            margin-bottom: 8px;
        }}
        
        .bracket-header .subtitle {{
            font-size: 1rem;
            color: #a0a0b0;
            margin-bottom: 12px;
        }}
        
        .bracket-header .champion-banner {{
            display: inline-block;
            background: linear-gradient(135deg, #c9a227 0%, #e8c84a 100%);
            color: #1a1a2e;
            padding: 10px 30px;
            border-radius: 8px;
            font-size: 1.3rem;
            font-weight: 700;
            margin-top: 10px;
        }}
        
        .bracket-header .meta {{
            font-size: 0.85rem;
            color: #888;
            margin-top: 10px;
        }}
        
        /* First Four section */
        .first-four-section {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        
        .first-four-section h2 {{
            font-size: 1.2rem;
            color: #1a1a2e;
            margin-bottom: 15px;
            border-bottom: 2px solid #c9a227;
            padding-bottom: 8px;
        }}
        
        .first-four-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        
        /* Main bracket layout */
        .bracket-container {{
            display: flex;
            gap: 20px;
            overflow-x: auto;
            padding-bottom: 20px;
        }}
        
        .bracket-half {{
            flex: 1;
            min-width: 0;
        }}
        
        .bracket-half h2 {{
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: #1a1a2e;
        }}
        
        /* Regions */
        .region {{
            background: white;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        
        .region-title {{
            font-size: 1rem;
            color: #1a1a2e;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .region-bracket {{
            display: flex;
            gap: 4px;
        }}
        
        .round-col {{
            flex: 1;
            min-width: 0;
        }}
        
        .round-label {{
            font-size: 0.7rem;
            color: #888;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        /* Game box */
        .game-slot {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin-bottom: 8px;
            overflow: hidden;
        }}
        
        .team-row {{
            display: flex;
            align-items: center;
            padding: 4px 6px;
            gap: 4px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .team-row:last-of-type {{
            border-bottom: none;
        }}
        
        .team-row.winner {{
            background: #e8f5e9;
            font-weight: 600;
        }}
        
        .seed {{
            font-size: 0.65rem;
            color: #888;
            min-width: 22px;
            flex-shrink: 0;
        }}
        
        .team-name {{
            font-size: 0.72rem;
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .score {{
            font-size: 0.72rem;
            font-weight: 600;
            min-width: 22px;
            text-align: right;
            flex-shrink: 0;
        }}
        
        .game-info {{
            padding: 2px 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .total {{
            font-size: 0.6rem;
            color: #999;
        }}
        
        .upset-badge {{
            background: #ff4757;
            color: white;
            font-size: 0.55rem;
            padding: 1px 4px;
            border-radius: 3px;
            font-weight: 600;
        }}
        
        /* Final Four + Championship */
        .final-four-section {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
        }}
        
        .final-four-section h2 {{
            font-size: 1.2rem;
            color: #1a1a2e;
            margin-bottom: 15px;
        }}
        
        .ff-games {{
            display: flex;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
        }}
        
        .ff-game, .champ-game {{
            display: inline-block;
            width: 220px;
        }}
        
        .champ-game {{
            margin-top: 20px;
        }}
        
        .champ-game .game-slot {{
            border: 2px solid #c9a227;
            background: #fffdf5;
        }}
        
        .champ-game .winner .team-name {{
            color: #c9a227;
        }}
        
        .section-label {{
            font-size: 0.8rem;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .bracket-header {{
                background: #1a1a2e !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .region, .first-four-section, .final-four-section {{
                box-shadow: none;
                border: 1px solid #ddd;
            }}
        }}
        
        @media (max-width: 1200px) {{
            .region-bracket {{
                flex-direction: column;
            }}
            .first-four-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>

<div class="bracket-header">
    <h1>2026 NCAA Tournament Bracket</h1>
    <div class="subtitle">Projected Results — AI Predictions</div>
    <div class="champion-banner">CHAMPION: {champ_name}</div>
    <div class="meta">
        Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")} |
        Model: GPU-Accelerated Monte Carlo (RTX 5090 Blackwell) |
        100,000 simulations
    </div>
</div>

<!-- First Four -->
<div class="first-four-section">
    <h2>First Four — Dayton, OH (March 17-18)</h2>
    <div class="first-four-grid">
        {"".join(game_html(g, 0) for g in first_four)}
    </div>
</div>

<!-- Main Bracket -->
<div class="bracket-container">
    <!-- Left half: East + South -->
    <div class="bracket-half">
        <div class="region">
            <h2 class="region-title">East Region</h2>
            <div class="region-bracket">
                <div class="round-col">
                    <div class="round-label">Round of 64</div>
                    {"".join(game_html(g, 1) for g in r64[0:8])}
                </div>
                <div class="round-col">
                    <div class="round-label">Round of 32</div>
                    {"".join(game_html(g, 2) for g in r32[0:4])}
                </div>
                <div class="round-col">
                    <div class="round-label">Sweet 16</div>
                    {"".join(game_html(g, 3) for g in s16[0:2])}
                </div>
                <div class="round-col">
                    <div class="round-label">Elite 8</div>
                    {game_html(e8[0], 4) if e8 else ""}
                </div>
            </div>
        </div>
        
        <div class="region">
            <h2 class="region-title">South Region</h2>
            <div class="region-bracket">
                <div class="round-col">
                    <div class="round-label">Round of 64</div>
                    {"".join(game_html(g, 1) for g in r64[16:24])}
                </div>
                <div class="round-col">
                    <div class="round-label">Round of 32</div>
                    {"".join(game_html(g, 2) for g in r32[8:12])}
                </div>
                <div class="round-col">
                    <div class="round-label">Sweet 16</div>
                    {"".join(game_html(g, 3) for g in s16[4:6])}
                </div>
                <div class="round-col">
                    <div class="round-label">Elite 8</div>
                    {game_html(e8[2], 4) if len(e8) > 2 else ""}
                </div>
            </div>
        </div>
    </div>
    
    <!-- Final Four + Championship Center -->
    <div class="final-four-section" style="flex: 0 0 280px; display: flex; flex-direction: column; justify-content: center;">
        <h2>Final Four</h2>
        <div class="section-label">San Antonio, TX — April 4</div>
        <div class="ff-games">
            {"".join(f'<div class="ff-game">{game_html(g, 5)}</div>' for g in ff)}
        </div>
        <div class="champ-game">
            <div class="section-label">National Championship — April 6</div>
            {champ_html}
        </div>
    </div>
    
    <!-- Right half: West + Midwest -->
    <div class="bracket-half">
        <div class="region">
            <h2 class="region-title">West Region</h2>
            <div class="region-bracket">
                <div class="round-col">
                    <div class="round-label">Round of 64</div>
                    {"".join(game_html(g, 1) for g in r64[8:16])}
                </div>
                <div class="round-col">
                    <div class="round-label">Round of 32</div>
                    {"".join(game_html(g, 2) for g in r32[4:8])}
                </div>
                <div class="round-col">
                    <div class="round-label">Sweet 16</div>
                    {"".join(game_html(g, 3) for g in s16[2:4])}
                </div>
                <div class="round-col">
                    <div class="round-label">Elite 8</div>
                    {game_html(e8[1], 4) if len(e8) > 1 else ""}
                </div>
            </div>
        </div>
        
        <div class="region">
            <h2 class="region-title">Midwest Region</h2>
            <div class="region-bracket">
                <div class="round-col">
                    <div class="round-label">Round of 64</div>
                    {"".join(game_html(g, 1) for g in r64[24:32])}
                </div>
                <div class="round-col">
                    <div class="round-label">Round of 32</div>
                    {"".join(game_html(g, 2) for g in r32[12:16])}
                </div>
                <div class="round-col">
                    <div class="round-label">Sweet 16</div>
                    {"".join(game_html(g, 3) for g in s16[6:8])}
                </div>
                <div class="round-col">
                    <div class="round-label">Elite 8</div>
                    {game_html(e8[3], 4) if len(e8) > 3 else ""}
                </div>
            </div>
        </div>
    </div>
</div>

<div style="text-align: center; margin-top: 30px; color: #999; font-size: 0.8rem;">
    Predictions generated by GPU-accelerated Monte Carlo simulation (RTX 5090 Blackwell)
    <br>Win probabilities based on adjusted efficiency ratings + matchup features
</div>

</body>
</html>
"""
    return html


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render bracket to HTML")
    parser.add_argument(
        "--input",
        default="predictions/tournament_2026/best_bracket.json",
        help="Path to bracket JSON",
    )
    parser.add_argument(
        "--output", default="predictions/tournament_2026/bracket.html", help="Output HTML path"
    )
    args = parser.parse_args()

    input_path = project_root / args.input
    output_path = project_root / args.output

    print(f"Loading bracket from {input_path}")
    with open(input_path) as f:
        bracket_data = json.load(f)

    games = bracket_data.get("games", [])
    champion = bracket_data.get("champion", "TBD")

    print(f"Found {len(games)} games")
    print(f"Champion: {champion}")

    html = render_bracket_html(games, champion)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Bracket rendered to {output_path}")
    print(f"Open in browser: file://{output_path}")


if __name__ == "__main__":
    main()
