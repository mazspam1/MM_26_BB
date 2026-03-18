"""
Update the bracket dashboard HTML with latest simulation results.
Regenerates ALL game data from simulation, not just header.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_march_madness_v2 import (
    load_teams,
    find_team,
    REGIONS,
    R64_MATCHUPS,
    TeamData,
    calc_game,
    BRACKET_TEAM_MAP,
)

import numpy as np

try:
    import cupy as cp

    USE_GPU = True
except:
    USE_GPU = False


def run_simulation(n_sims=1_000_000):
    """Run bracket simulation and return detailed results."""
    teams = load_teams()

    # Resolve bracket teams
    team_data = {}
    seed_map = {}

    for region, seed_teams in REGIONS.items():
        for seed, name in seed_teams.items():
            t = find_team(name, teams)
            if t:
                team_data[name] = t
                seed_map[name] = seed

    # Build R64 games with slot IDs matching dashboard format
    # Dashboard uses: 201-208 (East), 209-216 (South), 217-224 (West), 225-232 (Midwest)
    r64_games = []

    region_order = ["East", "South", "West", "Midwest"]
    base_slots = {"East": 200, "South": 208, "West": 216, "Midwest": 224}

    for region in region_order:
        seed_teams = REGIONS[region]
        base = base_slots[region]
        for i, (high_seed, low_seed) in enumerate(R64_MATCHUPS):
            high_name = seed_teams.get(high_seed, "")
            low_name = seed_teams.get(low_seed, "")
            if high_name and low_name:
                r64_games.append(
                    {
                        "id": base + i + 1,
                        "region": region,
                        "team_a": high_name,
                        "team_b": low_name,
                        "seed_a": high_seed,
                        "seed_b": low_seed,
                    }
                )

    # Generate random values
    values_per_game = 5
    if USE_GPU:
        rng = cp.random.default_rng(42)
        all_rand = rng.standard_normal((n_sims, 63, values_per_game), dtype=cp.float32)
        rand_np = cp.asnumpy(all_rand)
    else:
        rng = np.random.default_rng(42)
        rand_np = rng.standard_normal((n_sims, 63, values_per_game)).astype(np.float32)

    team_names = list(team_data.keys())
    team_idx = {name: i for i, name in enumerate(team_names)}
    n_teams = len(team_data)
    advancement = np.zeros((n_teams, 7), dtype=np.int64)
    championships = np.zeros(n_teams, dtype=np.int64)

    # Track R64 results for display
    r64_aggs = {}  # slot_id -> {score_a_total, score_b_total, wins_a}

    # Track advancement by round for later rounds
    r32_matchups = {}  # slot -> {a, b, wins_a, score_a, score_b}
    s16_matchups = {}
    e8_matchups = {}
    ff_matchups = {}
    champ_matchup = {}

    print(f"Running {n_sims:,} simulations...")

    for sim in range(n_sims):
        r64_winners = {}  # region -> list of (name, seed)

        # R64
        for g in r64_games:
            ta = team_data.get(g["team_a"])
            tb = team_data.get(g["team_b"])
            if not ta or not tb:
                continue

            rv = rand_np[sim, g["id"] % 63]

            sa, sb, wc, _, _, _ = calc_game(ta, tb, rv)
            winner = g["team_a"] if wc == "A" else g["team_b"]
            loser = g["team_b"] if wc == "A" else g["team_a"]
            w_seed = g["seed_a"] if wc == "A" else g["seed_b"]

            w_idx = team_idx.get(winner, -1)
            if w_idx >= 0:
                advancement[w_idx, 1] += 1

            # Aggregate for display
            if g["id"] not in r64_aggs:
                r64_aggs[g["id"]] = {
                    "score_a": 0,
                    "score_b": 0,
                    "wins_a": 0,
                    "a": g["team_a"],
                    "b": g["team_b"],
                    "sa": g["seed_a"],
                    "sb": g["seed_b"],
                }
            r64_aggs[g["id"]]["score_a"] += sa
            r64_aggs[g["id"]]["score_b"] += sb
            if wc == "A":
                r64_aggs[g["id"]]["wins_a"] += 1

            if g["region"] not in r64_winners:
                r64_winners[g["region"]] = []
            r64_winners[g["region"]].append((winner, w_seed))

        # R32 (slots 301-316)
        r32_winners = {}
        r32_slot = 300
        for region in region_order:
            winners = r64_winners.get(region, [])
            for i in range(0, len(winners) - 1, 2):
                r32_slot += 1
                ta = team_data.get(winners[i][0])
                tb = team_data.get(winners[i + 1][0])
                if not ta or not tb:
                    continue

                rv = rand_np[sim, r32_slot % 63]
                sa, sb, wc, _, _, _ = calc_game(ta, tb, rv)
                winner = winners[i][0] if wc == "A" else winners[i + 1][0]

                w_idx = team_idx.get(winner, -1)
                if w_idx >= 0:
                    advancement[w_idx, 2] += 1

                if r32_slot not in r32_matchups:
                    r32_matchups[r32_slot] = {
                        "score_a": 0,
                        "score_b": 0,
                        "wins_a": 0,
                        "a": winners[i][0],
                        "b": winners[i + 1][0],
                        "sa": winners[i][1],
                        "sb": winners[i + 1][1],
                    }
                r32_matchups[r32_slot]["score_a"] += sa
                r32_matchups[r32_slot]["score_b"] += sb
                if wc == "A":
                    r32_matchups[r32_slot]["wins_a"] += 1

                if region not in r32_winners:
                    r32_winners[region] = []
                r32_winners[region].append(
                    (winner, [winners[i][1], winners[i + 1][1]][0 if wc == "A" else 1])
                )

        # S16 (slots 401-408)
        s16_winners = {}
        s16_slot = 400
        for region in region_order:
            winners = r32_winners.get(region, [])
            for i in range(0, len(winners) - 1, 2):
                s16_slot += 1
                ta = team_data.get(winners[i][0])
                tb = team_data.get(winners[i + 1][0])
                if not ta or not tb:
                    continue

                rv = rand_np[sim, s16_slot % 63]
                sa, sb, wc, _, _, _ = calc_game(ta, tb, rv)
                winner = winners[i][0] if wc == "A" else winners[i + 1][0]

                w_idx = team_idx.get(winner, -1)
                if w_idx >= 0:
                    advancement[w_idx, 3] += 1

                if s16_slot not in s16_matchups:
                    s16_matchups[s16_slot] = {
                        "score_a": 0,
                        "score_b": 0,
                        "wins_a": 0,
                        "a": winners[i][0],
                        "b": winners[i + 1][0],
                        "sa": winners[i][1],
                        "sb": winners[i + 1][1],
                    }
                s16_matchups[s16_slot]["score_a"] += sa
                s16_matchups[s16_slot]["score_b"] += sb
                if wc == "A":
                    s16_matchups[s16_slot]["wins_a"] += 1

                if region not in s16_winners:
                    s16_winners[region] = []
                s16_winners[region].append(
                    (winner, [winners[i][1], winners[i + 1][1]][0 if wc == "A" else 1])
                )

        # E8 (slots 501-504)
        e8_winners = []
        e8_slot = 500
        for region in region_order:
            winners = s16_winners.get(region, [])
            for i in range(0, len(winners) - 1, 2):
                e8_slot += 1
                ta = team_data.get(winners[i][0])
                tb = team_data.get(winners[i + 1][0])
                if not ta or not tb:
                    continue

                rv = rand_np[sim, e8_slot % 63]
                sa, sb, wc, _, _, _ = calc_game(ta, tb, rv)
                winner = winners[i][0] if wc == "A" else winners[i + 1][0]

                w_idx = team_idx.get(winner, -1)
                if w_idx >= 0:
                    advancement[w_idx, 4] += 1

                if e8_slot not in e8_matchups:
                    e8_matchups[e8_slot] = {
                        "score_a": 0,
                        "score_b": 0,
                        "wins_a": 0,
                        "a": winners[i][0],
                        "b": winners[i + 1][0],
                        "sa": winners[i][1],
                        "sb": winners[i + 1][1],
                    }
                e8_matchups[e8_slot]["score_a"] += sa
                e8_matchups[e8_slot]["score_b"] += sb
                if wc == "A":
                    e8_matchups[e8_slot]["wins_a"] += 1

                e8_winners.append(
                    (winner, [winners[i][1], winners[i + 1][1]][0 if wc == "A" else 1])
                )

        # F4 (slots 601-602)
        ff_winners = []
        if len(e8_winners) >= 4:
            for idx, (i, j) in enumerate([(0, 1), (2, 3)]):
                slot = 601 + idx
                ta = team_data.get(e8_winners[i][0])
                tb = team_data.get(e8_winners[j][0])
                if not ta or not tb:
                    continue

                rv = rand_np[sim, slot % 63]
                sa, sb, wc, _, _, _ = calc_game(ta, tb, rv)
                winner = e8_winners[i][0] if wc == "A" else e8_winners[j][0]

                w_idx = team_idx.get(winner, -1)
                if w_idx >= 0:
                    advancement[w_idx, 5] += 1

                if slot not in ff_matchups:
                    ff_matchups[slot] = {
                        "score_a": 0,
                        "score_b": 0,
                        "wins_a": 0,
                        "a": e8_winners[i][0],
                        "b": e8_winners[j][0],
                        "sa": e8_winners[i][1],
                        "sb": e8_winners[j][1],
                    }
                ff_matchups[slot]["score_a"] += sa
                ff_matchups[slot]["score_b"] += sb
                if wc == "A":
                    ff_matchups[slot]["wins_a"] += 1

                ff_winners.append(
                    (winner, [e8_winners[i][1], e8_winners[j][1]][0 if wc == "A" else 1])
                )

        # Championship (slot 701)
        if len(ff_winners) >= 2:
            ta = team_data.get(ff_winners[0][0])
            tb = team_data.get(ff_winners[1][0])
            if ta and tb:
                rv = rand_np[sim, 62]
                sa, sb, wc, _, _, _ = calc_game(ta, tb, rv)
                champion = ff_winners[0][0] if wc == "A" else ff_winners[1][0]

                w_idx = team_idx.get(champion, -1)
                if w_idx >= 0:
                    championships[w_idx] += 1

                if 701 not in champ_matchup:
                    champ_matchup[701] = {
                        "score_a": 0,
                        "score_b": 0,
                        "wins_a": 0,
                        "a": ff_winners[0][0],
                        "b": ff_winners[1][0],
                        "sa": ff_winners[0][1],
                        "sb": ff_winners[1][1],
                    }
                champ_matchup[701]["score_a"] += sa
                champ_matchup[701]["score_b"] += sb
                if wc == "A":
                    champ_matchup[701]["wins_a"] += 1

    return {
        "r64_aggs": r64_aggs,
        "r32_matchups": r32_matchups,
        "s16_matchups": s16_matchups,
        "e8_matchups": e8_matchups,
        "ff_matchups": ff_matchups,
        "champ_matchup": champ_matchup,
        "advancement": advancement,
        "championships": championships,
        "team_names": team_names,
        "team_idx": team_idx,
        "seed_map": seed_map,
        "n_sims": n_sims,
    }


def generate_game_js(sim_data):
    """Generate the JavaScript game array for the bracket."""
    n_sims = sim_data["n_sims"]

    # Short name mapping
    short_names = {
        "Duke Blue Devils": "Duke",
        "UConn Huskies": "UConn",
        "Michigan State Spartans": "Michigan St.",
        "Kansas Jayhawks": "Kansas",
        "St. John's Red Storm": "St. John's",
        "Louisville Cardinals": "Louisville",
        "UCLA Bruins": "UCLA",
        "Ohio State Buckeyes": "Ohio State",
        "TCU Horned Frogs": "TCU",
        "UCF Knights": "UCF",
        "South Florida Bulls": "S. Florida",
        "Northern Iowa Panthers": "N. Iowa",
        "California Baptist Lancers": "Cal Baptist",
        "North Dakota State Bison": "N. Dakota St.",
        "Furman Paladins": "Furman",
        "Siena Saints": "Siena",
        "Arizona Wildcats": "Arizona",
        "Purdue Boilermakers": "Purdue",
        "Gonzaga Bulldogs": "Gonzaga",
        "Arkansas Razorbacks": "Arkansas",
        "Wisconsin Badgers": "Wisconsin",
        "BYU Cougars": "BYU",
        "Miami Hurricanes": "Miami",
        "Villanova Wildcats": "Villanova",
        "Utah State Aggies": "Utah State",
        "Missouri Tigers": "Missouri",
        "Texas Longhorns": "Texas",
        "High Point Panthers": "High Point",
        "Hawai'i Rainbow Warriors": "Hawaii",
        "Kennesaw State Owls": "Kennesaw St.",
        "Queens University Royals": "Queens",
        "Long Island University Sharks": "LIU",
        "Florida Gators": "Florida",
        "Houston Cougars": "Houston",
        "Illinois Fighting Illini": "Illinois",
        "Nebraska Cornhuskers": "Nebraska",
        "Vanderbilt Commodores": "Vanderbilt",
        "North Carolina Tar Heels": "N. Carolina",
        "Saint Mary's Gaels": "Saint Mary's",
        "Clemson Tigers": "Clemson",
        "Iowa Hawkeyes": "Iowa",
        "Texas A&M Aggies": "Texas A&M",
        "VCU Rams": "VCU",
        "McNeese Cowboys": "McNeese",
        "Troy Trojans": "Troy",
        "Pennsylvania Quakers": "Penn",
        "Idaho Vandals": "Idaho",
        "Lehigh Mountain Hawks": "Lehigh",
        "Michigan Wolverines": "Michigan",
        "Iowa State Cyclones": "Iowa State",
        "Virginia Cavaliers": "Virginia",
        "Alabama Crimson Tide": "Alabama",
        "Texas Tech Red Raiders": "Texas Tech",
        "Tennessee Volunteers": "Tennessee",
        "Kentucky Wildcats": "Kentucky",
        "Georgia Bulldogs": "Georgia",
        "Saint Louis Billikens": "Saint Louis",
        "Santa Clara Broncos": "Santa Clara",
        "SMU Mustangs": "SMU",
        "Akron Zips": "Akron",
        "Hofstra Pride": "Hofstra",
        "Wright State Raiders": "Wright St.",
        "Tennessee State Tigers": "Tenn. State",
        "UMBC Retrievers": "UMBC",
    }

    def sn(name):
        return short_names.get(name, name.split()[-1])

    games_js = []

    # First Four (placeholder - not in our 68-team bracket)
    games_js.append(
        '{id:101,r:0,g:1,sa:16,a:"UMBC",pa:65,sb:16,b:"Howard",pb:63,w:"UMBC",wp:.51,t:128,sp:0}'
    )
    games_js.append(
        '{id:102,r:0,g:2,sa:11,a:"Texas",pa:75,sb:11,b:"NC State",pb:72,w:"Texas",wp:.55,t:147,sp:3}'
    )
    games_js.append(
        '{id:103,r:0,g:3,sa:16,a:"Prairie View",pa:62,sb:16,b:"Lehigh",pb:64,w:"Lehigh",wp:.53,t:126,sp:-2}'
    )
    games_js.append(
        '{id:104,r:0,g:4,sa:11,a:"SMU",pa:76,sb:11,b:"NC State",pb:73,w:"SMU",wp:.54,t:149,sp:3}'
    )

    # R64 games
    for slot_id, agg in sim_data["r64_aggs"].items():
        avg_a = agg["score_a"] / n_sims
        avg_b = agg["score_b"] / n_sims
        wp = agg["wins_a"] / n_sims
        total = avg_a + avg_b
        spread = avg_a - avg_b
        w = agg["a"] if wp > 0.5 else agg["b"]
        games_js.append(
            f'{{id:{slot_id},r:1,g:{slot_id % 100},sa:{agg["sa"]},a:"{sn(agg["a"])}",pa:{avg_a:.1f},sb:{agg["sb"]},b:"{sn(agg["b"])}",pb:{avg_b:.1f},w:"{sn(w)}",wp:{wp:.3f},t:{total:.0f},sp:{spread:.1f}}}'
        )

    # R32 games
    for slot_id, agg in sim_data["r32_matchups"].items():
        avg_a = agg["score_a"] / n_sims
        avg_b = agg["score_b"] / n_sims
        wp = agg["wins_a"] / n_sims
        total = avg_a + avg_b
        spread = avg_a - avg_b
        w = agg["a"] if wp > 0.5 else agg["b"]
        games_js.append(
            f'{{id:{slot_id},r:2,g:{slot_id % 100},sa:{agg["sa"]},a:"{sn(agg["a"])}",pa:{avg_a:.1f},sb:{agg["sb"]},b:"{sn(agg["b"])}",pb:{avg_b:.1f},w:"{sn(w)}",wp:{wp:.3f},t:{total:.0f},sp:{spread:.1f}}}'
        )

    # S16 games
    for slot_id, agg in sim_data["s16_matchups"].items():
        avg_a = agg["score_a"] / n_sims
        avg_b = agg["score_b"] / n_sims
        wp = agg["wins_a"] / n_sims
        total = avg_a + avg_b
        spread = avg_a - avg_b
        w = agg["a"] if wp > 0.5 else agg["b"]
        games_js.append(
            f'{{id:{slot_id},r:3,g:{slot_id % 100},sa:{agg["sa"]},a:"{sn(agg["a"])}",pa:{avg_a:.1f},sb:{agg["sb"]},b:"{sn(agg["b"])}",pb:{avg_b:.1f},w:"{sn(w)}",wp:{wp:.3f},t:{total:.0f},sp:{spread:.1f}}}'
        )

    # E8 games
    for slot_id, agg in sim_data["e8_matchups"].items():
        avg_a = agg["score_a"] / n_sims
        avg_b = agg["score_b"] / n_sims
        wp = agg["wins_a"] / n_sims
        total = avg_a + avg_b
        spread = avg_a - avg_b
        w = agg["a"] if wp > 0.5 else agg["b"]
        games_js.append(
            f'{{id:{slot_id},r:4,g:{slot_id % 100},sa:{agg["sa"]},a:"{sn(agg["a"])}",pa:{avg_a:.1f},sb:{agg["sb"]},b:"{sn(agg["b"])}",pb:{avg_b:.1f},w:"{sn(w)}",wp:{wp:.3f},t:{total:.0f},sp:{spread:.1f}}}'
        )

    # F4 games
    for slot_id, agg in sim_data["ff_matchups"].items():
        avg_a = agg["score_a"] / n_sims
        avg_b = agg["score_b"] / n_sims
        wp = agg["wins_a"] / n_sims
        total = avg_a + avg_b
        spread = avg_a - avg_b
        w = agg["a"] if wp > 0.5 else agg["b"]
        games_js.append(
            f'{{id:{slot_id},r:5,g:{slot_id % 100},sa:{agg["sa"]},a:"{sn(agg["a"])}",pa:{avg_a:.1f},sb:{agg["sb"]},b:"{sn(agg["b"])}",pb:{avg_b:.1f},w:"{sn(w)}",wp:{wp:.3f},t:{total:.0f},sp:{spread:.1f}}}'
        )

    # Championship
    for slot_id, agg in sim_data["champ_matchup"].items():
        avg_a = agg["score_a"] / n_sims
        avg_b = agg["score_b"] / n_sims
        wp = agg["wins_a"] / n_sims
        total = avg_a + avg_b
        spread = avg_a - avg_b
        w = agg["a"] if wp > 0.5 else agg["b"]
        games_js.append(
            f'{{id:{slot_id},r:6,g:1,sa:{agg["sa"]},a:"{sn(agg["a"])}",pa:{avg_a:.1f},sb:{agg["sb"]},b:"{sn(agg["b"])}",pb:{avg_b:.1f},w:"{sn(w)}",wp:{wp:.3f},t:{total:.0f},sp:{spread:.1f}}}'
        )

    return ",\n".join(games_js)


def generate_adv_js(sim_data):
    """Generate advancement table data."""
    n_sims = sim_data["n_sims"]
    advancement = sim_data["advancement"]
    championships = sim_data["championships"]
    team_idx = sim_data["team_idx"]
    seed_map = sim_data["seed_map"]
    team_names = sim_data["team_names"]

    results = []
    for name in team_names:
        idx = team_idx.get(name, -1)
        if idx >= 0:
            r64 = advancement[idx, 1] / n_sims * 100
            r32 = advancement[idx, 2] / n_sims * 100
            s16 = advancement[idx, 3] / n_sims * 100
            e8 = advancement[idx, 4] / n_sims * 100
            f4 = advancement[idx, 5] / n_sims * 100
            champ = championships[idx] / n_sims * 100
            ew = sum(advancement[idx, 1:7]) / n_sims
            results.append(
                {
                    "name": name,
                    "seed": seed_map.get(name, 99),
                    "r64": r64,
                    "r32": r32,
                    "s16": s16,
                    "e8": e8,
                    "f4": f4,
                    "ch": champ,
                    "ew": ew,
                }
            )

    results.sort(key=lambda x: -x["ch"])

    adv_lines = []
    for r in results[:20]:
        adv_lines.append(
            f'{{n:"{r["name"]}",r64:"{r["r64"]:.1f}",r32:"{r["r32"]:.1f}",s16:"{r["s16"]:.1f}",e8:"{r["e8"]:.1f}",f4:"{r["f4"]:.1f}",ch:"{r["ch"]:.1f}",ew:"{r["ew"]:.2f}"}}'
        )

    return ",\n".join(adv_lines)


def main():
    print("=" * 60)
    print("  GENERATING LIVE BRACKET DASHBOARD")
    print("=" * 60)

    sim_data = run_simulation(1_000_000)

    # Find champion
    champ_idx = np.argmax(sim_data["championships"])
    champion_name = sim_data["team_names"][champ_idx]
    champ_pct = sim_data["championships"][champ_idx] / sim_data["n_sims"] * 100

    print(f"\nChampion: {champion_name} ({champ_pct:.1f}%)")

    # Generate JS data
    print("Generating bracket data...")
    games_js = generate_game_js(sim_data)
    adv_js = generate_adv_js(sim_data)

    # Read template and replace data sections
    template_path = Path(__file__).parent.parent / "apps" / "dashboard" / "bracket.html"
    with open(template_path, "r") as f:
        html = f.read()

    # Update header with regex for flexible matching
    import re

    html = re.sub(
        r"[\d,]+ Monte Carlo Simulations", f"{sim_data['n_sims']:,} Monte Carlo Simulations", html
    )
    html = re.sub(r'<div class="cn">[^<]+</div>', f'<div class="cn">{champion_name}</div>', html)
    html = re.sub(r"[\d.]+% Title Probability", f"{champ_pct:.1f}% Title Probability", html)
    html = re.sub(
        r'<div class="sv">[\d.]+%</div><div class="sl">Title Prob</div>',
        f'<div class="sv">{champ_pct:.1f}%</div><div class="sl">Title Prob</div>',
        html,
    )
    html = re.sub(
        r'<div class="sv">\w+</div><div class="sl">Champion</div>',
        f'<div class="sv">{champion_name.split()[0]}</div><div class="sl">Champion</div>',
        html,
    )

    # Replace G array
    g_start = html.find("const G=[")
    g_end = html.find("];", g_start) + 2
    if g_start > 0 and g_end > g_start:
        html = html[:g_start] + "const G=[\n" + games_js + "\n];\n" + html[g_end:]

    # Replace ADV array
    adv_start = html.find("const ADV=[")
    adv_end = html.find("];", adv_start) + 2
    if adv_start > 0 and adv_end > adv_start:
        html = html[:adv_start] + "const ADV=[\n" + adv_js + "\n];\n" + html[adv_end:]

    # Write updated file
    with open(template_path, "w") as f:
        f.write(html)

    print(f"\nDashboard updated: {template_path}")
    print(f"Champion: {champion_name} ({champ_pct:.1f}%)")

    # Save JSON summary
    json_path = (
        Path(__file__).parent.parent / "predictions" / "tournament_2026" / "sim_results.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name in sim_data["team_names"]:
        idx = sim_data["team_idx"].get(name, -1)
        if idx >= 0:
            results.append(
                {
                    "name": name,
                    "seed": sim_data["seed_map"].get(name, 99),
                    "champ_pct": round(
                        sim_data["championships"][idx] / sim_data["n_sims"] * 100, 1
                    ),
                }
            )
    results.sort(key=lambda x: -x["champ_pct"])

    with open(json_path, "w") as f:
        json.dump(
            {
                "champion": champion_name,
                "champ_pct": round(champ_pct, 1),
                "top10": results[:10],
                "n_sims": sim_data["n_sims"],
                "gpu": USE_GPU,
            },
            f,
            indent=2,
        )

    print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
