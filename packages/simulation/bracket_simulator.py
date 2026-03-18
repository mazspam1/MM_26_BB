"""
Interactive Bracket Simulator.
Handles state management and round-by-round simulation.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scripts.run_march_madness_v2 import (
    load_teams,
    find_team,
    REGIONS,
    R64_MATCHUPS,
    TeamData,
    calc_game,
)

try:
    import cupy as cp

    HAS_GPU = True
except:
    HAS_GPU = False

SHORT_NAMES = {
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
    "Hawai'i Rainbow Warriors": "Hawai'i",
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

REGION_ORDER = ["East", "South", "West", "Midwest"]

R64_SLOTS = {
    "East": list(range(201, 209)),
    "South": list(range(209, 217)),
    "West": list(range(217, 225)),
    "Midwest": list(range(225, 233)),
}
R32_SLOTS = {
    "East": list(range(301, 305)),
    "South": list(range(305, 309)),
    "West": list(range(309, 313)),
    "Midwest": list(range(313, 317)),
}
S16_SLOTS = {"East": [401, 402], "South": [403, 404], "West": [405, 406], "Midwest": [407, 408]}
E8_SLOTS = {"East": 501, "South": 502, "West": 503, "Midwest": 504}
F4_SLOTS = [601, 602]
CHAMP_SLOT = 701


def sn(name):
    return SHORT_NAMES.get(name, name.split()[-1] if name else "")


class BracketSimulator:
    def __init__(self):
        self.team_data = {}
        self.seed_map = {}
        self.region_map = {}
        self._load_teams()
        self.state = {
            "r64": {},
            "r32": {},
            "s16": {},
            "e8": {},
            "f4": [],
            "championship": None,
            "champion": None,
        }
        self.completed = set()
        self._init_r64()

    def _load_teams(self):
        teams = load_teams()
        for region, seed_teams in REGIONS.items():
            for seed, name in seed_teams.items():
                t = find_team(name, teams)
                if t:
                    self.team_data[name] = t
                    self.seed_map[name] = seed
                    self.region_map[name] = region
                else:
                    print(
                        f"    WARNING: Missing team data for {name} (seed {seed}, region {region})"
                    )

    def _init_r64(self):
        for region in REGION_ORDER:
            seed_teams = REGIONS[region]
            for i, (hs, ls) in enumerate(R64_MATCHUPS):
                hn = seed_teams.get(hs, "")
                ln = seed_teams.get(ls, "")
                if hn and ln and i < len(R64_SLOTS[region]):
                    sid = R64_SLOTS[region][i]
                    self.state["r64"][sid] = {
                        "id": sid,
                        "a": hn,
                        "sa": hs,
                        "b": ln,
                        "sb": ls,
                        "winner": None,
                    }

    def _sim_game(self, ta_name, tb_name, n_sims):
        """
        Run N simulations of a game and sample ONE outcome.

        Returns stats from all sims plus a randomly sampled winner
        weighted by the actual simulation results.
        """
        ta, tb = self.team_data.get(ta_name), self.team_data.get(tb_name)
        if not ta or not tb:
            return None

        if HAS_GPU:
            rng = cp.random.default_rng()
            rand = rng.standard_normal((n_sims, 5), dtype=cp.float32)
            rand_np = cp.asnumpy(rand)
        else:
            rand_np = np.random.default_rng().standard_normal((n_sims, 5)).astype(np.float32)

        # Store all simulation outcomes
        scores_a = np.zeros(n_sims, dtype=np.float32)
        scores_b = np.zeros(n_sims, dtype=np.float32)
        wins_a = 0

        for s in range(n_sims):
            sa, sb, wc, _, _, _ = calc_game(ta, tb, rand_np[s])
            scores_a[s] = sa
            scores_b[s] = sb
            if wc == "A":
                wins_a += 1

        # Stats from all simulations
        avg_a = float(np.mean(scores_a))
        avg_b = float(np.mean(scores_b))
        win_prob_a = wins_a / n_sims

        # Determine the "displayed" favorite
        if win_prob_a > 0.5:
            display_wp = win_prob_a
            display_wp_team = "a"
            fav_name = ta_name
        else:
            display_wp = 1.0 - win_prob_a
            display_wp_team = "b"
            fav_name = tb_name

        # ===== SAMPLE ONE ACTUAL OUTCOME =====
        # Randomly pick a winner weighted by simulation results
        actual_winner_is_a = np.random.random() < win_prob_a

        if actual_winner_is_a:
            # Pick a random simulation where A won
            a_win_indices = np.where(scores_a > scores_b)[0]
            if len(a_win_indices) == 0:
                a_win_indices = np.where(scores_a >= scores_b)[0]
            idx = np.random.choice(a_win_indices)
            sampled_winner = ta_name
            sampled_loser = tb_name
        else:
            # Pick a random simulation where B won
            b_win_indices = np.where(scores_b > scores_a)[0]
            if len(b_win_indices) == 0:
                b_win_indices = np.where(scores_b >= scores_a)[0]
            idx = np.random.choice(b_win_indices)
            sampled_winner = tb_name
            sampled_loser = ta_name

        # The actual scores for THIS specific game outcome
        actual_score_a = float(scores_a[idx])
        actual_score_b = float(scores_b[idx])

        return {
            # Stats from all 100k sims (for display/analysis)
            "pa": round(avg_a, 1),
            "pb": round(avg_b, 1),
            "wp": round(display_wp, 3),
            "wp_team": display_wp_team,
            "t": round(avg_a + avg_b),
            "sp": round(avg_a - avg_b, 1),
            # The ACTUAL sampled outcome (determines who advances)
            "winner": sampled_winner,
            "loser": sampled_loser,
            "actual_sa": round(actual_score_a),
            "actual_sb": round(actual_score_b),
            "win_prob_a": round(win_prob_a, 3),
            "n_sims": n_sims,
        }

    def simulate_round(self, rnd, n_sims=100000):
        result = {"round": rnd, "games": []}

        if rnd == "r64":
            for sid, g in self.state["r64"].items():
                if g.get("winner"):
                    continue
                r = self._sim_game(g["a"], g["b"], n_sims)
                if r:
                    g["winner"] = r["winner"]
                    g["actual_sa"] = r["actual_sa"]
                    g["actual_sb"] = r["actual_sb"]
                    result["games"].append(
                        {
                            "id": sid,
                            "a": sn(g["a"]),
                            "b": sn(g["b"]),
                            "sa": g["sa"],
                            "sb": g["sb"],
                            **r,
                            "winner": sn(r["winner"]),
                        }
                    )
            self.completed.add("r64")
            self._advance_r64()

        elif rnd == "r32":
            for sid, g in self.state["r32"].items():
                if not g or g.get("winner"):
                    continue
                r = self._sim_game(g["a"], g["b"], n_sims)
                if r:
                    g["winner"] = r["winner"]
                    g["actual_sa"] = r["actual_sa"]
                    g["actual_sb"] = r["actual_sb"]
                    result["games"].append(
                        {
                            "id": sid,
                            "a": sn(g["a"]),
                            "b": sn(g["b"]),
                            "sa": g["sa"],
                            "sb": g["sb"],
                            **r,
                            "winner": sn(r["winner"]),
                        }
                    )
            self.completed.add("r32")
            self._advance_r32()

        elif rnd == "s16":
            for sid, g in self.state["s16"].items():
                if not g or g.get("winner"):
                    continue
                r = self._sim_game(g["a"], g["b"], n_sims)
                if r:
                    g["winner"] = r["winner"]
                    g["actual_sa"] = r["actual_sa"]
                    g["actual_sb"] = r["actual_sb"]
                    result["games"].append(
                        {
                            "id": sid,
                            "a": sn(g["a"]),
                            "b": sn(g["b"]),
                            "sa": g["sa"],
                            "sb": g["sb"],
                            **r,
                            "winner": sn(r["winner"]),
                        }
                    )
            self.completed.add("s16")
            self._advance_s16()

        elif rnd == "e8":
            for sid, g in self.state["e8"].items():
                if not g or g.get("winner"):
                    continue
                r = self._sim_game(g["a"], g["b"], n_sims)
                if r:
                    g["winner"] = r["winner"]
                    g["actual_sa"] = r["actual_sa"]
                    g["actual_sb"] = r["actual_sb"]
                    result["games"].append(
                        {
                            "id": sid,
                            "a": sn(g["a"]),
                            "b": sn(g["b"]),
                            "sa": g["sa"],
                            "sb": g["sb"],
                            **r,
                            "winner": sn(r["winner"]),
                        }
                    )
            self.completed.add("e8")
            self._advance_e8()

        elif rnd == "f4":
            for i, g in enumerate(self.state["f4"]):
                if not g or g.get("winner"):
                    continue
                r = self._sim_game(g["a"], g["b"], n_sims)
                if r:
                    g["winner"] = r["winner"]
                    g["actual_sa"] = r["actual_sa"]
                    g["actual_sb"] = r["actual_sb"]
                    result["games"].append(
                        {
                            "id": F4_SLOTS[i],
                            "a": sn(g["a"]),
                            "b": sn(g["b"]),
                            "sa": g["sa"],
                            "sb": g["sb"],
                            **r,
                            "winner": sn(r["winner"]),
                        }
                    )
            self.completed.add("f4")
            self._advance_f4()

        elif rnd == "championship":
            g = self.state["championship"]
            if g and not g.get("winner"):
                r = self._sim_game(g["a"], g["b"], n_sims)
                if r:
                    g["winner"] = r["winner"]
                    g["actual_sa"] = r["actual_sa"]
                    g["actual_sb"] = r["actual_sb"]
                    self.state["champion"] = r["winner"]
                    result["games"].append(
                        {
                            "id": CHAMP_SLOT,
                            "a": sn(g["a"]),
                            "b": sn(g["b"]),
                            "sa": g["sa"],
                            "sb": g["sb"],
                            **r,
                            "winner": sn(r["winner"]),
                        }
                    )
            self.completed.add("championship")

        return result

    def _advance_r64(self):
        for region in REGION_ORDER:
            r64 = R64_SLOTS[region]
            r32 = R32_SLOTS[region]
            for i in range(0, len(r64), 2):
                if i // 2 < len(r32):
                    ga = self.state["r64"].get(r64[i], {})
                    gb = self.state["r64"].get(r64[i + 1], {})
                    wa, wb = ga.get("winner"), gb.get("winner")
                    if wa and wb:
                        # Use actual simulated scores
                        sa = ga.get("actual_sa", ga["sa"])
                        sb = gb.get("actual_sb", gb["sb"])
                        self.state["r32"][r32[i // 2]] = {
                            "id": r32[i // 2],
                            "a": wa,
                            "sa": sa,
                            "b": wb,
                            "sb": sb,
                            "winner": None,
                        }

    def _advance_r32(self):
        for region in REGION_ORDER:
            r32 = R32_SLOTS[region]
            s16 = S16_SLOTS[region]
            for i in range(0, len(r32), 2):
                if i // 2 < len(s16):
                    ga = self.state["r32"].get(r32[i], {})
                    gb = self.state["r32"].get(r32[i + 1], {})
                    wa, wb = ga.get("winner"), gb.get("winner")
                    if wa and wb:
                        sa = ga.get("actual_sa", ga["sa"])
                        sb = gb.get("actual_sb", gb["sb"])
                        self.state["s16"][s16[i // 2]] = {
                            "id": s16[i // 2],
                            "a": wa,
                            "sa": sa,
                            "b": wb,
                            "sb": sb,
                            "winner": None,
                        }

    def _advance_s16(self):
        for region in REGION_ORDER:
            s16 = S16_SLOTS[region]
            e8 = E8_SLOTS[region]
            if len(s16) == 2:
                ga = self.state["s16"].get(s16[0], {})
                gb = self.state["s16"].get(s16[1], {})
                wa, wb = ga.get("winner"), gb.get("winner")
                if wa and wb:
                    sa = ga.get("actual_sa", ga["sa"])
                    sb = gb.get("actual_sb", gb["sb"])
                    self.state["e8"][e8] = {
                        "id": e8,
                        "a": wa,
                        "sa": sa,
                        "b": wb,
                        "sb": sb,
                        "winner": None,
                    }

    def _advance_e8(self):
        winners = []
        for region in REGION_ORDER:
            g = self.state["e8"].get(E8_SLOTS[region], {})
            w = g.get("winner")
            if w:
                # Use winner's actual simulated score
                if w == g["a"]:
                    s = g.get("actual_sa", g["sa"])
                else:
                    s = g.get("actual_sb", g["sb"])
                winners.append((w, s))
        if len(winners) >= 4:
            self.state["f4"] = [
                {
                    "id": F4_SLOTS[0],
                    "a": winners[0][0],
                    "sa": winners[0][1],
                    "b": winners[1][0],
                    "sb": winners[1][1],
                    "winner": None,
                },
                {
                    "id": F4_SLOTS[1],
                    "a": winners[2][0],
                    "sa": winners[2][1],
                    "b": winners[3][0],
                    "sb": winners[3][1],
                    "winner": None,
                },
            ]

    def _advance_f4(self):
        winners = []
        for g in self.state["f4"]:
            w = g.get("winner")
            if w:
                if w == g["a"]:
                    s = g.get("actual_sa", g["sa"])
                else:
                    s = g.get("actual_sb", g["sb"])
                winners.append((w, s))
        if len(winners) >= 2:
            self.state["championship"] = {
                "id": CHAMP_SLOT,
                "a": winners[0][0],
                "sa": winners[0][1],
                "b": winners[1][0],
                "sb": winners[1][1],
                "winner": None,
            }

    def simulate_all(self, n_sims=100000):
        result = {"rounds": []}
        for rnd in ["r64", "r32", "s16", "e8", "f4", "championship"]:
            if rnd not in self.completed:
                result["rounds"].append(self.simulate_round(rnd, n_sims))
        return result

    def get_state(self):
        """Return displayable state."""
        s = {
            "r64": {},
            "r32": {},
            "s16": {},
            "e8": {},
            "f4": [],
            "championship": None,
            "champion": self.state.get("champion"),
            "completed": list(self.completed),
        }

        for rnd in ["r64", "r32", "s16", "e8"]:
            for sid, g in self.state[rnd].items():
                if g and g.get("winner"):
                    s[rnd][str(sid)] = {
                        "a": sn(g["a"]),
                        "b": sn(g["b"]),
                        "sa": g["sa"],
                        "sb": g["sb"],
                        **{k: g.get(k, 0) for k in ["pa", "pb", "wp", "t", "sp"]},
                        "winner": sn(g["winner"]),
                    }

        for g in self.state.get("f4", []):
            if g and g.get("winner"):
                s["f4"].append(
                    {
                        "id": g["id"],
                        "a": sn(g["a"]),
                        "b": sn(g["b"]),
                        "sa": g["sa"],
                        "sb": g["sb"],
                        **{k: g.get(k, 0) for k in ["pa", "pb", "wp", "t", "sp"]},
                        "winner": sn(g["winner"]),
                    }
                )

        g = self.state.get("championship")
        if g and g.get("winner"):
            s["championship"] = {
                "id": CHAMP_SLOT,
                "a": sn(g["a"]),
                "b": sn(g["b"]),
                "sa": g["sa"],
                "sb": g["sb"],
                **{k: g.get(k, 0) for k in ["pa", "pb", "wp", "t", "sp"]},
                "winner": sn(g["winner"]),
            }

        return s


if __name__ == "__main__":
    sim = BracketSimulator()
    print(f"Loaded {len(sim.team_data)} teams")
    result = sim.simulate_round("r64", 1000)
    print(f"Simulated {len(result['games'])} R64 games")
