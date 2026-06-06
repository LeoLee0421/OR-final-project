"""
Generate 100 test cases across 6 scenarios for performance evaluation.

Scenarios (from Section 6 of the project report):
  base              - medium applicants, uniform pref, medium exp-seniors, balanced gender
  high_competition  - medium applicants, concentrated pref, medium exp-seniors, balanced gender
  leader_shortage   - medium applicants, uniform pref, LOW exp-seniors, balanced gender
  gender_imbalance  - medium applicants, uniform pref, medium exp-seniors, IMBALANCED gender
  applicant_shortage- LOW applicants, uniform pref, medium exp-seniors, balanced gender
  applicant_excess  - HIGH applicants, uniform pref, medium exp-seniors, balanced gender

Output: testcases/<scenario>_NN.json  (total 100 files)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Fixed department parameters (from CSIE camp) ─────────────────────────────
J    = ["Act", "Pbr", "Acd", "Srv", "Ins", "Pho", "Art"]
J_L  = ["Act", "Pbr", "Acd", "Srv", "Art"]
J_S  = ["Act", "Pbr", "Acd", "Srv", "Art"]
J_R  = ["Act", "Pbr", "Acd", "Srv", "Art"]
T    = ["pre", "during"]

L = {"Act": 25, "Pbr": 2, "Acd": 8,  "Srv": 4, "Ins": 20, "Pho": 2, "Art": 4}
U = {"Act": 35, "Pbr": 6, "Acd": 14, "Srv": 8, "Ins": 20, "Pho": 6, "Art": 8}
A = {"Act": 30, "Pbr": 4, "Acd": 11, "Srv": 6, "Ins": 20, "Pho": 4, "Art": 6}

B = {
    "Act": {"pre": 1, "during": 1},
    "Pbr": {"pre": 1, "during": 0},
    "Acd": {"pre": 1, "during": 1},
    "Srv": {"pre": 1, "during": 1},
    "Ins": {"pre": 0, "during": 1},
    "Pho": {"pre": 0, "during": 1},
    "Art": {"pre": 1, "during": 0},
}

INS_MALE_REQ   = 10
INS_FEMALE_REQ = 10

# ── Scenario definitions ──────────────────────────────────────────────────────
SCENARIOS: Dict[str, Dict] = {
    "base": {
        "n_range": (90, 100),
        "pref_mode": "uniform",
        "senior_prob": 0.30,
        "exp_given_senior": 0.65,
        "male_prob": 0.50,
        "count": 100,
    },
    "high_competition": {
        "n_range": (90, 100),
        "pref_mode": "concentrated",
        "senior_prob": 0.30,
        "exp_given_senior": 0.65,
        "male_prob": 0.50,
        "count": 100,
    },
    "leader_shortage": {
        # Very few experienced seniors → only the minimum 5 are guaranteed;
        # random extras have low probability of being experienced seniors.
        "n_range": (90, 100),
        "pref_mode": "uniform",
        "senior_prob": 0.12,
        "exp_given_senior": 0.35,
        "male_prob": 0.50,
        "count": 100,
    },
    "gender_imbalance": {
        # 75 % male → Ins female and Srv female requirements become tight.
        "n_range": (90, 100),
        "pref_mode": "uniform",
        "senior_prob": 0.30,
        "exp_given_senior": 0.65,
        "male_prob": 0.78,
        "count": 100,
    },
    "applicant_shortage": {
        # Fewer applicants → departments harder to fill to minimum sizes.
        "n_range": (70, 78),
        "pref_mode": "uniform",
        "senior_prob": 0.30,
        "exp_given_senior": 0.65,
        "male_prob": 0.50,
        "count": 100,
    },
    "applicant_excess": {
        # Many applicants → more flexibility but also more competing for same slots.
        "n_range": (130, 145),
        "pref_mode": "uniform",
        "senior_prob": 0.30,
        "exp_given_senior": 0.65,
        "male_prob": 0.50,
        "count": 100,
    },
}


def _make_preference(rng: random.Random, mode: str) -> Dict[str, int]:
    """Return a preference dict {dept: rank} (rank 1 = most preferred)."""
    ranks = list(range(1, len(J) + 1))
    rng.shuffle(ranks)
    pref = {j: ranks[k] for k, j in enumerate(J)}

    if mode == "concentrated":
        # 70 % chance: bias top-choice toward Activity or Team Instructor
        if rng.random() < 0.70:
            fav = rng.choice(["Act", "Ins"])
            # Swap whichever dept currently holds rank-1 with fav
            rank1_dept = min(pref, key=lambda j: pref[j])
            pref[fav], pref[rank1_dept] = pref[rank1_dept], pref[fav]

    return pref


def generate_instance(
    scenario_name: str,
    cfg: Dict,
    instance_idx: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    n_target = rng.randint(*cfg["n_range"])

    # ── Guaranteed feasibility pool ───────────────────────────────────────────
    # Every instance has at minimum:
    #   5 experienced seniors  (one leader per dept in J_L)
    #  10 junior males          (Ins male requirement)
    #  10 junior females        (Ins female requirement)
    # These are always included regardless of scenario distributions.
    pool: List[Dict] = []

    for _ in range(5):
        pool.append({"y": 1, "e": 1, "m": rng.choice([0, 1])})

    for _ in range(INS_MALE_REQ):
        pool.append({"y": 0, "e": 0, "m": 1})

    for _ in range(INS_FEMALE_REQ):
        pool.append({"y": 0, "e": 0, "m": 0})

    # ── Scenario-distributed extras ───────────────────────────────────────────
    n_extra = max(0, n_target - len(pool))
    for _ in range(n_extra):
        is_senior = rng.random() < cfg["senior_prob"]
        is_exp    = (rng.random() < cfg["exp_given_senior"]) if is_senior else False
        is_male   = rng.random() < cfg["male_prob"]
        pool.append({"y": int(is_senior), "e": int(is_exp), "m": int(is_male)})

    rng.shuffle(pool)

    I = [f"i{k:03d}" for k in range(len(pool))]
    y = {I[k]: pool[k]["y"] for k in range(len(pool))}
    e = {I[k]: pool[k]["e"] for k in range(len(pool))}
    m = {I[k]: pool[k]["m"] for k in range(len(pool))}
    p = {i: _make_preference(rng, cfg["pref_mode"]) for i in I}

    return {
        "I": I,
        "J": J,
        "J_L": J_L,
        "J_S": J_S,
        "J_R": J_R,
        "T": T,
        "y": y,
        "e": e,
        "m": m,
        "p": p,
        "l": L,
        "u": U,
        "a": A,
        "b": B,
        "w1": 0.4,
        "w2": 0.4,
        "w3": 0.2,
        "ins_male_required": INS_MALE_REQ,
        "ins_female_required": INS_FEMALE_REQ,
        "time_limit": 300,
        "mip_gap": 0.005,
        "log_to_console": False,
        "scenario": scenario_name,
        "instance_id": f"{scenario_name}_{instance_idx:02d}",
    }


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "testcases"
    out_dir.mkdir(exist_ok=True)

    total = 0
    for scenario_name, cfg in SCENARIOS.items():
        for idx in range(1, cfg["count"] + 1):
            # Deterministic but distinct seed per (scenario, instance)
            seed = abs(hash((scenario_name, idx, "camp2026"))) % (2 ** 31)
            data = generate_instance(scenario_name, cfg, idx, seed)

            fname = out_dir / f"{scenario_name}_{idx:02d}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            n = len(data["I"])
            n_senior = sum(data["y"][i] for i in data["I"])
            n_exp_sr = sum(data["e"][i] for i in data["I"] if data["y"][i] == 1)
            n_male   = sum(data["m"][i] for i in data["I"])
            print(
                f"  {fname.name}  n={n}  seniors={n_senior}  "
                f"exp_sr={n_exp_sr}  male={n_male}"
            )
            total += 1

    print(f"\nDone: {total} test cases in {out_dir}/")
