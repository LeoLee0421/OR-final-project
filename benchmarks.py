"""
Three upper-bound benchmark heuristics for the camp allocation problem.

All three satisfy the same hard constraints as test2.py but use different
strategies in the optional-assignment phase:

  1. random_optimal_size_heuristic
     – Randomly fills departments toward their optimal staffing sizes.
       Prioritizes departments most below their target.

  2. preference_satisfaction_heuristic
     – Greedily assigns each applicant to their highest-ranked available
       department (preference-first, ignoring staffing-gap cost).

  3. unrecruited_minimization_heuristic
     – Maximises the number of recruited applicants by giving every
       unassigned person at least one department before allowing dual jobs.

Each function accepts the same JSON-compatible data dict as test2.py and
returns the same solution dict format.
"""

from __future__ import annotations

import random
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

from test2 import (
    CampState,
    _best_candidate,
    _assign_best_for_departments,
    format_solution,
    print_solution,
)


# ── Shared constraint-satisfaction preamble ───────────────────────────────────

def _parse_data(data: Mapping[str, Any]):
    """Extract and coerce all parameters from the input dict."""
    I   = list(data["I"])
    J   = list(data["J"])
    J_L = list(data["J_L"])
    J_S = list(data["J_S"])
    J_R = list(data["J_R"])
    T   = list(data["T"])

    y = {i: int(data["y"][i]) for i in I}
    e = {i: int(data["e"][i]) for i in I}
    m = {i: int(data["m"][i]) for i in I}
    p = {i: {j: float(data["p"][i][j]) for j in J} for i in I}

    l = {j: int(data["l"][j]) for j in J}
    u = {j: int(data["u"][j]) for j in J}
    a = {j: int(data["a"][j]) for j in J}
    b = {j: {t: int(data["b"][j][t]) for t in T} for j in J}

    w1 = float(data["w1"])
    w2 = float(data["w2"])
    w3 = float(data["w3"])
    ins_m = int(data.get("ins_male_required", 10))
    ins_f = int(data.get("ins_female_required", 10))

    return I, J, J_L, J_S, J_R, T, y, e, m, p, l, u, a, b, w1, w2, w3, ins_m, ins_f


def _infeasible_return() -> Dict[str, Any]:
    return {
        "status": "INFEASIBLE",
        "obj_value": None,
        "x": {}, "r": {}, "v": {},
        "gplus": {}, "gminus": {},
    }


def _satisfy_mandatory_constraints(
    state: CampState,
    I: List[str],
) -> bool:
    """
    Run stages 1-5 of test2.py:
      1. Leader placement
      2. Ins exact gender counts
      3. Srv gender requirements
      4. Per-dept senior / junior requirements
      5. Department minimum sizes
    Returns False if any stage fails (infeasible).
    """
    # Stage 1 – leaders
    for j in state.J_L:
        cand = _best_candidate(state, I, j, leader=True)
        if cand is None:
            return False
        state.add_assignment(cand[0], j, leader=True)

    # Stage 2 – Ins gender balance
    if "Ins" in state.J:
        while state.ins_male < state.ins_male_required:
            cand = _best_candidate(state, I, "Ins", require_junior=True, require_male=True)
            if cand is None:
                return False
            state.add_assignment(cand[0], "Ins")
        while state.ins_female < state.ins_female_required:
            cand = _best_candidate(state, I, "Ins", require_junior=True, require_female=True)
            if cand is None:
                return False
            state.add_assignment(cand[0], "Ins")

    # Stage 3 – Srv gender
    if "Srv" in state.J:
        if state.srv_male < 1:
            cand = _best_candidate(
                state, I, "Srv", require_male=True,
                require_junior=(state.junior_j["Srv"] < 1),
            )
            if cand is None:
                return False
            state.add_assignment(cand[0], "Srv")
        if state.srv_female < 1:
            cand = _best_candidate(
                state, I, "Srv", require_female=True,
                require_junior=(state.junior_j["Srv"] < 1),
            )
            if cand is None:
                return False
            state.add_assignment(cand[0], "Srv")

    # Stage 4 – senior / junior per dept
    for j in state.J_S:
        if state.senior_j[j] < 1:
            cand = _best_candidate(state, I, j, require_senior=True)
            if cand is None:
                return False
            state.add_assignment(cand[0], j)
    for j in state.J_R:
        if state.junior_j[j] < 1:
            cand = _best_candidate(state, I, j, require_junior=True)
            if cand is None:
                return False
            state.add_assignment(cand[0], j)

    # Stage 5 – minimum sizes
    return _assign_best_for_departments(state, list(state.J))


# ── Benchmark 1: Random Optimal Size ─────────────────────────────────────────

def random_optimal_size_heuristic(
    data: Mapping[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    After satisfying hard constraints, randomly fill departments up to their
    optimal staffing size a[j].  Departments most below their target are
    processed first; applicants within each department's candidates are
    shuffled randomly.
    """
    I, J, J_L, J_S, J_R, T, y, e, m, p, l, u, a, b, w1, w2, w3, ins_m, ins_f = _parse_data(data)

    if "Ins" in J and not (l["Ins"] <= ins_m + ins_f <= u["Ins"]):
        return _infeasible_return()

    state = CampState(I, J, J_L, J_S, J_R, T, y, e, m, p, l, u, a, b, w1, w2, w3, ins_m, ins_f)

    if not _satisfy_mandatory_constraints(state, I):
        return _infeasible_return()

    # Optional phase: fill toward optimal sizes in random applicant order
    rng = random.Random(seed)
    improved = True
    while improved:
        improved = False
        # Pick the department most below its optimal size
        below = [(a[j] - state.count_j[j], j) for j in J if state.count_j[j] < a[j]]
        if not below:
            break
        below.sort(reverse=True)
        for _, j in below:
            candidates = [i for i in I if state.can_assign(i, j)]
            if not candidates:
                continue
            rng.shuffle(candidates)
            chosen = candidates[0]
            state.add_assignment(chosen, j)
            improved = True
            break   # restart outer loop so dept ranking is recalculated

    return state.get_solution()


# ── Benchmark 2: Preference Satisfaction ─────────────────────────────────────

def preference_satisfaction_heuristic(data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    After satisfying hard constraints, greedily give each applicant their
    highest-ranked available department (ignoring staffing-gap cost).

    Applicants are processed in order of how strongly they prefer their
    current best available choice (ascending preference rank = stronger desire).
    """
    I, J, J_L, J_S, J_R, T, y, e, m, p, l, u, a, b, w1, w2, w3, ins_m, ins_f = _parse_data(data)

    if "Ins" in J and not (l["Ins"] <= ins_m + ins_f <= u["Ins"]):
        return _infeasible_return()

    state = CampState(I, J, J_L, J_S, J_R, T, y, e, m, p, l, u, a, b, w1, w2, w3, ins_m, ins_f)

    if not _satisfy_mandatory_constraints(state, I):
        return _infeasible_return()

    # Optional phase: preference-first greedy
    # Repeatedly find the (applicant, dept) pair with the best preference rank
    # among all currently feasible assignments.
    improved = True
    while improved:
        improved = False
        best: Optional[Tuple[str, str, float]] = None
        for i in I:
            if len(state.assigned[i]) >= state._max_assignments(i):
                continue
            for j in J:
                if j in state.assigned[i]:
                    continue
                if not state.can_assign(i, j):
                    continue
                rank = p[i][j]   # lower = better preference
                if best is None or rank < best[2]:
                    best = (i, j, rank)
        if best is not None:
            state.add_assignment(best[0], best[1])
            improved = True

    return state.get_solution()


# ── Benchmark 3: Unrecruited Minimisation ────────────────────────────────────

def unrecruited_minimization_heuristic(
    data: Mapping[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    After satisfying hard constraints, maximise the number of recruited
    applicants by first giving every unassigned person at least one department
    before assigning second departments.

    Each iteration we prefer assigning to an applicant who has zero current
    assignments, reducing unrecruitment.
    """
    I, J, J_L, J_S, J_R, T, y, e, m, p, l, u, a, b, w1, w2, w3, ins_m, ins_f = _parse_data(data)

    if "Ins" in J and not (l["Ins"] <= ins_m + ins_f <= u["Ins"]):
        return _infeasible_return()

    state = CampState(I, J, J_L, J_S, J_R, T, y, e, m, p, l, u, a, b, w1, w2, w3, ins_m, ins_f)

    if not _satisfy_mandatory_constraints(state, I):
        return _infeasible_return()

    rng = random.Random(seed)

    # Pass 1: give every currently unassigned applicant their first department
    unassigned = [i for i in I if len(state.assigned[i]) == 0]
    rng.shuffle(unassigned)
    for i in unassigned:
        candidates = [j for j in J if state.can_assign(i, j)]
        if not candidates:
            continue
        # Pick the dept with most room below upper bound (ties broken by preference)
        candidates.sort(key=lambda j: (-( u[j] - state.count_j[j]), p[i][j]))
        state.add_assignment(i, candidates[0])

    # Pass 2: assign second departments only if dept still needs more people
    improved = True
    while improved:
        improved = False
        for j in J:
            if state.count_j[j] >= u[j]:
                continue
            candidates = [i for i in I if state.can_assign(i, j)]
            if not candidates:
                continue
            # Prefer applicants who already have 1 assignment (don't re-un-recruit)
            candidates.sort(
                key=lambda i: (len(state.assigned[i]) == 0, p[i][j])
            )
            state.add_assignment(candidates[0], j)
            improved = True

    return state.get_solution()


# ── CLI entry point (single file) ─────────────────────────────────────────────

def main(argv=None):
    import json, sys
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python benchmarks.py INPUT.json [b1|b2|b3]", file=sys.stderr)
        sys.exit(1)

    with open(argv[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    which = argv[1] if len(argv) > 1 else "all"

    if which in ("b1", "all"):
        print("=== Benchmark 1: Random Optimal Size ===")
        print_solution(random_optimal_size_heuristic(data))
    if which in ("b2", "all"):
        print("=== Benchmark 2: Preference Satisfaction ===")
        print_solution(preference_satisfaction_heuristic(data))
    if which in ("b3", "all"):
        print("=== Benchmark 3: Unrecruited Minimisation ===")
        print_solution(unrecruited_minimization_heuristic(data))


if __name__ == "__main__":
    main()
