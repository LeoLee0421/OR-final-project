"""
Heuristic solver for the camp staff recruitment and allocation problem.

This module reads the same JSON input format as `gurobi_solve.py` and
produces a feasible assignment that respects all model constraints.

The heuristic is greedy with problem-aware stages:
- leader placement
- exact team instructor gender balancing
- senior/junior and gender requirements
- department minimum staffing
- optional assignments when they improve the objective

Usage:
    python test2.py INPUT.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


def _require_keys(data: Mapping[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in input data: {missing}")


def _abs_diff(a: float, b: float) -> float:
    return abs(a - b)


class CampState:
    def __init__(
        self,
        I: List[str],
        J: List[str],
        J_L: List[str],
        J_S: List[str],
        J_R: List[str],
        T: List[str],
        y: Mapping[str, int],
        e: Mapping[str, int],
        m: Mapping[str, int],
        p: Mapping[str, Mapping[str, float]],
        l: Mapping[str, int],
        u: Mapping[str, int],
        a: Mapping[str, int],
        b: Mapping[str, Mapping[str, int]],
        w1: float,
        w2: float,
        w3: float,
        ins_male_required: int,
        ins_female_required: int,
    ):
        self.I = I
        self.J = J
        self.J_L = set(J_L)
        self.J_S = set(J_S)
        self.J_R = set(J_R)
        self.T = T

        self.y = y
        self.e = e
        self.m = m
        self.p = p
        self.l = l
        self.u = u
        self.a = a
        self.b = b

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.ins_male_required = ins_male_required
        self.ins_female_required = ins_female_required
        self.ins_total_required = ins_male_required + ins_female_required

        self.assigned: Dict[str, Set[str]] = {i: set() for i in I}
        self.leader: Dict[str, Optional[str]] = {j: None for j in J}
        self.is_leader: Dict[str, bool] = {i: False for i in I}
        self.count_j: Counter[str] = Counter({j: 0 for j in J})
        self.senior_j: Counter[str] = Counter({j: 0 for j in J})
        self.junior_j: Counter[str] = Counter({j: 0 for j in J})
        self.busy_count: Dict[str, Dict[str, int]] = {
            i: {t: 0 for t in T} for i in I
        }
        self.srv_male = 0
        self.srv_female = 0
        self.ins_male = 0
        self.ins_female = 0

        self.denom_gap = 1.0 + sum(
            max(float(a[j]) - float(l[j]), float(u[j]) - float(a[j])) for j in J
        )
        self.abs_I = float(len(I))
        self.denom_pref = float(len(J)) * sum(float(u[j]) for j in J)

    def _max_assignments(self, applicant: str) -> int:
        return 1 if self.is_leader[applicant] else 2

    def _busy_conflict(self, applicant: str, department: str) -> bool:
        for t in self.T:
            if self.b[department][t] == 1 and self.busy_count[applicant][t] >= 1:
                return True
        return False

    def _allowed_department(self, applicant: str, department: str) -> bool:
        if department == "Ins" and self.y[applicant] == 1:
            return False
        return True

    def _ins_capacity_ok(self, department: str, applicant: str) -> bool:
        if department != "Ins":
            return True
        if self.count_j["Ins"] >= self.ins_total_required:
            return False
        if self.m[applicant] == 1 and self.ins_male >= self.ins_male_required:
            return False
        if self.m[applicant] == 0 and self.ins_female >= self.ins_female_required:
            return False
        return True

    def can_assign(self, applicant: str, department: str, leader: bool = False) -> bool:
        if department in self.assigned[applicant]:
            return False
        if not self._allowed_department(applicant, department):
            return False
        if self.count_j[department] >= self.u[department]:
            return False
        if len(self.assigned[applicant]) >= self._max_assignments(applicant):
            return False
        if self._busy_conflict(applicant, department):
            return False
        if leader:
            if department not in self.J_L:
                return False
            if self.y[applicant] != 1 or self.e[applicant] != 1:
                return False
            if len(self.assigned[applicant]) != 0:
                return False
        if department == "Ins" and self.y[applicant] == 1:
            return False
        if not self._ins_capacity_ok(department, applicant):
            return False
        return True

    def add_assignment(self, applicant: str, department: str, leader: bool = False) -> None:
        self.assigned[applicant].add(department)
        self.count_j[department] += 1

        if self.y[applicant] == 1:
            self.senior_j[department] += 1
        else:
            self.junior_j[department] += 1

        for t in self.T:
            if self.b[department][t] == 1:
                self.busy_count[applicant][t] += 1

        if department == "Srv":
            if self.m[applicant] == 1:
                self.srv_male += 1
            else:
                self.srv_female += 1

        if department == "Ins":
            if self.m[applicant] == 1:
                self.ins_male += 1
            else:
                self.ins_female += 1

        if leader:
            self.leader[department] = applicant
            self.is_leader[applicant] = True

    def gap_term_delta(self, department: str) -> float:
        before = abs(self.count_j[department] - self.a[department])
        after = abs((self.count_j[department] + 1) - self.a[department])
        return after - before

    def objective_delta(self, applicant: str, department: str) -> float:
        if not self.can_assign(applicant, department):
            return float("inf")

        gap_cost = self.gap_term_delta(department) / self.denom_gap
        pref_cost = float(self.p[applicant][department]) / self.denom_pref
        unrecruit_cost = -1.0 / self.abs_I if len(self.assigned[applicant]) == 0 else 0.0
        return self.w1 * gap_cost + self.w2 * pref_cost + self.w3 * unrecruit_cost

    def objective_value(self) -> float:
        gap_value = sum(abs(self.count_j[j] - self.a[j]) for j in self.J) / self.denom_gap
        pref_value = sum(
            float(self.p[i][j]) for i in self.I for j in self.assigned[i]
        ) / self.denom_pref
        unrecruit_value = sum(1 for i in self.I if len(self.assigned[i]) == 0) / self.abs_I
        return self.w1 * gap_value + self.w2 * pref_value + self.w3 * unrecruit_value

    def is_feasible(self) -> bool:
        if any(self.count_j[j] < self.l[j] or self.count_j[j] > self.u[j] for j in self.J):
            return False
        if any(len(self.assigned[i]) > self._max_assignments(i) for i in self.I):
            return False
        if any(self.busy_count[i][t] > 1 for i in self.I for t in self.T):
            return False
        if any(self.count_j[j] < 1 for j in self.J_S):
            return False
        if any(
            self.count_j[j] < 1 if j in self.J_R else False
            for j in self.J_R
        ):
            return False
        if "Srv" in self.J and (self.srv_male < 1 or self.srv_female < 1):
            return False
        if "Ins" in self.J and (
            self.ins_male != self.ins_male_required or self.ins_female != self.ins_female_required
        ):
            return False
        if any(self.leader[j] is None for j in self.J_L):
            return False
        for j in self.J_S:
            if self.senior_j[j] < 1:
                return False
        for j in self.J_R:
            if self.junior_j[j] < 1:
                return False
        return True

    def get_solution(self) -> Dict[str, Any]:
        x_solution: Dict[str, Dict[str, int]] = {
            i: {j: int(j in self.assigned[i]) for j in self.J} for i in self.I
        }
        r_solution: Dict[str, Dict[str, int]] = {
            i: {j: int(self.leader[j] == i) for j in self.J_L} for i in self.I
        }
        v_solution: Dict[str, int] = {i: int(len(self.assigned[i]) == 0) for i in self.I}
        gplus: Dict[str, float] = {}
        gminus: Dict[str, float] = {}
        for j in self.J:
            diff = self.count_j[j] - self.a[j]
            gplus[j] = float(max(diff, 0))
            gminus[j] = float(max(-diff, 0))

        solution = {
            "status": "FEASIBLE" if self.is_feasible() else "INFEASIBLE",
            "obj_value": self.objective_value(),
            "x": x_solution,
            "r": r_solution,
            "v": v_solution,
            "gplus": gplus,
            "gminus": gminus,
        }

        pref_raw = sum(
            float(self.p[i][j]) for i in self.I for j in self.assigned[i]
        )
        pref_norm = None
        if self.denom_pref > 0:
            pref_norm = pref_raw / self.denom_pref
        solution["pref_raw"] = pref_raw
        solution["pref_dissatisfaction"] = pref_norm
        return solution


def _best_candidate(
    state: CampState,
    applicants: Iterable[str],
    department: str,
    require_senior: bool = False,
    require_junior: bool = False,
    require_male: bool = False,
    require_female: bool = False,
    leader: bool = False,
) -> Optional[Tuple[str, float]]:
    best: Optional[Tuple[str, float]] = None
    for i in applicants:
        if require_senior and state.y[i] != 1:
            continue
        if require_junior and state.y[i] != 0:
            continue
        if require_male and state.m[i] != 1:
            continue
        if require_female and state.m[i] != 0:
            continue
        if leader and department not in state.J_L:
            continue
        if not state.can_assign(i, department, leader=leader):
            continue
        score = state.objective_delta(i, department)
        if best is None or score < best[1] or (
            score == best[1] and float(state.p[i][department]) < float(state.p[best[0]][department])
        ):
            best = (i, score)
    return best


def _assign_best_for_departments(state: CampState, departments: List[str]) -> bool:
    for j in departments:
        while state.count_j[j] < state.l[j]:
            candidate = _best_candidate(state, state.I, j)
            if candidate is None:
                return False
            state.add_assignment(candidate[0], j)
    return True


def solve_camp_problem_heuristic(data: Mapping[str, Any]) -> Dict[str, Any]:
    _require_keys(
        data,
        [
            "I",
            "J",
            "J_L",
            "J_S",
            "J_R",
            "T",
            "y",
            "e",
            "m",
            "p",
            "l",
            "u",
            "a",
            "b",
            "w1",
            "w2",
            "w3",
        ],
    )

    I: List[str] = list(data["I"])
    J: List[str] = list(data["J"])
    J_L: List[str] = list(data["J_L"])
    J_S: List[str] = list(data["J_S"])
    J_R: List[str] = list(data["J_R"])
    T: List[str] = list(data["T"])

    y: Mapping[str, int] = {i: int(data["y"][i]) for i in I}
    e: Mapping[str, int] = {i: int(data["e"][i]) for i in I}
    m: Mapping[str, int] = {i: int(data["m"][i]) for i in I}
    p: Mapping[str, Mapping[str, float]] = {
        i: {j: float(data["p"][i][j]) for j in J} for i in I
    }

    l: Mapping[str, int] = {j: int(data["l"][j]) for j in J}
    u: Mapping[str, int] = {j: int(data["u"][j]) for j in J}
    a: Mapping[str, int] = {j: int(data["a"][j]) for j in J}
    b: Mapping[str, Mapping[str, int]] = {
        j: {t: int(data["b"][j][t]) for t in T} for j in J
    }

    w1 = float(data["w1"])
    w2 = float(data["w2"])
    w3 = float(data["w3"])

    ins_male_required = int(data.get("ins_male_required", 10))
    ins_female_required = int(data.get("ins_female_required", 10))

    if "Ins" in J:
        if not (l["Ins"] <= ins_male_required + ins_female_required <= u["Ins"]):
            return {
                "status": "INFEASIBLE",
                "obj_value": None,
                "x": {},
                "r": {},
                "v": {},
                "gplus": {},
                "gminus": {},
            }

    state = CampState(
        I,
        J,
        J_L,
        J_S,
        J_R,
        T,
        y,
        e,
        m,
        p,
        l,
        u,
        a,
        b,
        w1,
        w2,
        w3,
        ins_male_required,
        ins_female_required,
    )

    # Stage 1: assign leaders first.
    for j in J_L:
        candidate = _best_candidate(state, I, j, leader=True)
        if candidate is None:
            return {
                "status": "INFEASIBLE",
                "obj_value": None,
                "x": {},
                "r": {},
                "v": {},
                "gplus": {},
                "gminus": {},
            }
        state.add_assignment(candidate[0], j, leader=True)

    # Stage 2: satisfy exact Ins gender counts when Ins exists.
    if "Ins" in J:
        while state.ins_male < ins_male_required:
            candidate = _best_candidate(state, I, "Ins", require_junior=True, require_male=True)
            if candidate is None:
                return {
                    "status": "INFEASIBLE",
                    "obj_value": None,
                    "x": {},
                    "r": {},
                    "v": {},
                    "gplus": {},
                    "gminus": {},
                }
            state.add_assignment(candidate[0], "Ins")
        while state.ins_female < ins_female_required:
            candidate = _best_candidate(state, I, "Ins", require_junior=True, require_female=True)
            if candidate is None:
                return {
                    "status": "INFEASIBLE",
                    "obj_value": None,
                    "x": {},
                    "r": {},
                    "v": {},
                    "gplus": {},
                    "gminus": {},
                }
            state.add_assignment(candidate[0], "Ins")

    # Stage 3: satisfy Srv male/female count requirements before general senior/junior assignment.
    if "Srv" in J:
        if state.srv_male < 1:
            candidate = _best_candidate(
                state,
                I,
                "Srv",
                require_male=True,
                require_junior=(state.junior_j["Srv"] < 1),
            )
            if candidate is None:
                return {
                    "status": "INFEASIBLE",
                    "obj_value": None,
                    "x": {},
                    "r": {},
                    "v": {},
                    "gplus": {},
                    "gminus": {},
                }
            state.add_assignment(candidate[0], "Srv")
        if state.srv_female < 1:
            candidate = _best_candidate(
                state,
                I,
                "Srv",
                require_female=True,
                require_junior=(state.junior_j["Srv"] < 1),
            )
            if candidate is None:
                return {
                    "status": "INFEASIBLE",
                    "obj_value": None,
                    "x": {},
                    "r": {},
                    "v": {},
                    "gplus": {},
                    "gminus": {},
                }
            state.add_assignment(candidate[0], "Srv")

    # Stage 4: satisfy senior and junior requirements.
    for j in J_S:
        if state.senior_j[j] < 1:
            candidate = _best_candidate(state, I, j, require_senior=True)
            if candidate is None:
                return {
                    "status": "INFEASIBLE",
                    "obj_value": None,
                    "x": {},
                    "r": {},
                    "v": {},
                    "gplus": {},
                    "gminus": {},
                }
            state.add_assignment(candidate[0], j)
    for j in J_R:
        if state.junior_j[j] < 1:
            candidate = _best_candidate(state, I, j, require_junior=True)
            if candidate is None:
                return {
                    "status": "INFEASIBLE",
                    "obj_value": None,
                    "x": {},
                    "r": {},
                    "v": {},
                    "gplus": {},
                    "gminus": {},
                }
            state.add_assignment(candidate[0], j)

    # Stage 5: satisfy minimum sizes for all departments.
    if not _assign_best_for_departments(state, J):
        return {
            "status": "INFEASIBLE",
            "obj_value": None,
            "x": {},
            "r": {},
            "v": {},
            "gplus": {},
            "gminus": {},
        }

    # Stage 6: add optional assignments that improve the objective.
    improvement = True
    while improvement:
        improvement = False
        best_add: Optional[Tuple[str, str, float]] = None
        for i in I:
            if len(state.assigned[i]) >= state._max_assignments(i):
                continue
            for j in J:
                if j in state.assigned[i]:
                    continue
                delta = state.objective_delta(i, j)
                if delta == float("inf"):
                    continue
                if best_add is None or delta < best_add[2]:
                    best_add = (i, j, delta)
        if best_add is not None and best_add[2] < 0.0:
            state.add_assignment(best_add[0], best_add[1])
            improvement = True

    # Stage 7: try to give every unassigned applicant a first assignment if it is beneficial.
    for i in I:
        if len(state.assigned[i]) == 0:
            best_choice: Optional[Tuple[str, float]] = None
            for j in J:
                candidate_score = state.objective_delta(i, j)
                if candidate_score == float("inf"):
                    continue
                if best_choice is None or candidate_score < best_choice[1]:
                    best_choice = (j, candidate_score)
            if best_choice is not None and best_choice[1] < 0.0:
                state.add_assignment(i, best_choice[0])

    return state.get_solution()


def format_solution(solution: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Status: {solution.get('status', 'UNKNOWN')}")

    obj_val = solution.get('obj_value')
    if obj_val is None:
        lines.append("Objective value: n/a")
    else:
        lines.append(f"Objective value: {float(obj_val):.6f}")

    r = solution.get('r', {})
    if r:
        lines.append("\nLeaders:")
        for i in sorted(r):
            leaders = [j for j, val in sorted(r[i].items()) if int(val) == 1]
            if leaders:
                lines.append(f"  {i}: {', '.join(leaders)}")

    x = solution.get('x', {})
    if x:
        lines.append("\nAssignments:")
        for i in sorted(x):
            assigned = [j for j, val in sorted(x[i].items()) if int(val) == 1]
            if assigned:
                lines.append(f"  {i}: {', '.join(assigned)}")
            else:
                lines.append(f"  {i}: none")

    gplus = solution.get('gplus', {})
    gminus = solution.get('gminus', {})
    if gplus or gminus:
        lines.append("\nStaffing gaps:")
        for j in sorted(set(list(gplus) + list(gminus))):
            lines.append(f"  {j}: +{gplus.get(j, 0)} -{gminus.get(j, 0)}")

    pref_norm = solution.get('pref_dissatisfaction')
    pref_raw = solution.get('pref_raw')
    if pref_norm is not None or pref_raw is not None:
        lines.append("\nPreference dissatisfaction:")
        if pref_norm is not None:
            lines.append(f"  Normalized: {float(pref_norm):.6f}")
        if pref_raw is not None:
            lines.append(f"  Raw total: {float(pref_raw):.2f}")

    v = solution.get('v', {})
    if v:
        unrecruited = [i for i, val in sorted(v.items()) if int(val) == 1]
        lines.append("\nUnrecruited applicants:")
        if unrecruited:
            lines.append(f"  {', '.join(unrecruited)}")
        else:
            lines.append("  None")

    return "\n".join(lines)


def print_solution(solution: Mapping[str, Any]) -> None:
    print(format_solution(solution))


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print("Usage: python test2.py INPUT.json", file=sys.stderr)
        sys.exit(1)

    input_path = argv[0]
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solution = solve_camp_problem_heuristic(data)
    print_solution(solution)


if __name__ == "__main__":
    main()
