"""
Gurobi model for the camp staff recruitment and allocation problem.

This script builds and solves the mathematical formulation described in camp.typ
using Gurobi. It exposes a function `solve_camp_problem(data)` that accepts all
sets and parameters as a Python dictionary, and a simple CLI interface that reads
JSON input.

Expected input format (JSON or Python dict):

{
  "I": ["i1", "i2", ...],                      # list of applicant IDs
  "J": ["Act", "Pbr", "Acd", "Srv", "Ins", "Pho", "Art"],  # department IDs
  "J_L": ["Act", "Pbr", "Acd", "Srv", "Art"],              # departments requiring a leader
  "J_S": ["Act", "Pbr", "Acd", "Srv", "Art"],              # departments requiring at least one senior
  "J_R": ["Act", "Pbr", "Acd", "Srv", "Art"],              # departments requiring at least one junior
  "T": ["pre", "during"],                     # workload phases

  # Parameters indexed by applicants
  "y": { "i1": 1, "i2": 0, ... },             # 1 if senior, 0 otherwise
  "e": { "i1": 1, "i2": 0, ... },             # 1 if has experience, 0 otherwise
  "m": { "i1": 1, "i2": 0, ... },             # 1 if male, 0 otherwise

  # Preference ranking p[i][j] (smaller = better)
  "p": {
    "i1": {"Act": 1, "Pbr": 2, ...},
    "i2": {"Act": 3, "Pbr": 1, ...},
    ...
  },

  # Department size parameters
  "l": { "Act": 25, "Pbr": 2, ... },          # lower bounds
  "u": { "Act": 35, "Pbr": 6, ... },          # upper bounds
  "a": { "Act": 30, "Pbr": 4, ... },          # optimal sizes (l[j] <= a[j] <= u[j])

  # Busy indicator b[j][t]
  "b": {
    "Act": {"pre": 1, "during": 1},
    "Pbr": {"pre": 1, "during": 0},
    ...
  },

  # Objective weights
  "w1": 0.5,
  "w2": 0.3,
  "w3": 0.2,

  # (Optional) Gurobi options
  "time_limit": 60,        # seconds
  "mip_gap": 0.0,          # default: 0 (optimal)
  "log_to_console": true   # default: True
}

The solution dictionary contains:
- "obj_value": optimal objective value
- "x": {i: {j: 0/1, ...}, ...}
- "r": {i: {j: 0/1, ...}, ...}
- "v": {i: 0/1, ...}
- "gplus": {j: >=0, ...}
- "gminus": {j: >=0, ...}
- "status": Gurobi status string
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Mapping, Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "gurobipy is required to run this script. Please install it with `pip install gurobipy` "
        "and ensure you have a valid Gurobi license."
    ) from exc


def _require_keys(data: Mapping[str, Any], keys: List[str]) -> None:
    """Raise a clear error if required keys are missing from input."""
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in input data: {missing}")


def solve_camp_problem(data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build and solve the camp allocation MIP using Gurobi.

    Parameters
    ----------
    data : Mapping[str, Any]
        Dictionary containing sets and parameters as documented in module docstring.

    Returns
    -------
    Dict[str, Any]
        Solution dictionary with variable values and objective, or a dict with only
        status if the model is infeasible or otherwise not solved to optimality.
    """
    # --- Validate and extract basic sets ---
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

    y: Mapping[str, int] = data["y"]
    e: Mapping[str, int] = data["e"]
    m: Mapping[str, int] = data["m"]

    p: Mapping[str, Mapping[str, float]] = data["p"]

    l: Mapping[str, int] = data["l"]
    u: Mapping[str, int] = data["u"]
    a: Mapping[str, int] = data["a"]

    b: Mapping[str, Mapping[str, int]] = data["b"]

    w1: float = float(data["w1"])
    w2: float = float(data["w2"])
    w3: float = float(data["w3"])

    # Optional: exact numbers of males/females required in Ins; default to 10 if not provided.
    ins_male_required: float = float(data.get("ins_male_required", 10))
    ins_female_required: float = float(data.get("ins_female_required", 10))

    time_limit: Optional[float] = data.get("time_limit")
    mip_gap: Optional[float] = data.get("mip_gap")
    log_to_console: bool = bool(data.get("log_to_console", True))

    # Basic size checks
    if not I or not J:
        raise ValueError("Sets I and J must be non-empty.")

    # --- Create model ---
    env = None
    if not log_to_console:
        # Suppress Gurobi output if requested
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("camp_allocation", env=env)
    else:
        model = gp.Model("camp_allocation")

    # --- Decision variables ---

    # x_(i j) in {0,1}
    x = model.addVars(I, J, vtype=GRB.BINARY, name="x")

    # r_(i j) in {0,1} only for j in J_L
    r = model.addVars(I, J_L, vtype=GRB.BINARY, name="r")

    # v_i in {0,1}
    v_var = model.addVars(I, vtype=GRB.BINARY, name="v")

    # g_j^+, g_j^- >= 0 (integer or continuous; we keep them as integer per formulation)
    gplus = model.addVars(J, vtype=GRB.INTEGER, lb=0.0, name="gplus")
    gminus = model.addVars(J, vtype=GRB.INTEGER, lb=0.0, name="gminus")

    # --- Constraints ---

    # Staffing gap definition:
    # g_j^+ - g_j^- = sum_i x_(i j) - a_j,  forall j in J
    for j in J:
        model.addConstr(
            gplus[j] - gminus[j]
            == gp.quicksum(x[i, j] for i in I) - float(a[j]),
            name=f"gap_def[{j}]",
        )

    # Unrecruited if no department assigned:
    # v_i + sum_j x_(i j) >= 1, forall i in I
    for i in I:
        model.addConstr(
            v_var[i] + gp.quicksum(x[i, j] for j in J) >= 1.0,
            name=f"unrecruit_def[{i}]",
        )

    # Size requirement:
    # l_j <= sum_i x_(i j) <= u_j, forall j in J
    for j in J:
        model.addConstr(
            gp.quicksum(x[i, j] for i in I) >= float(l[j]),
            name=f"size_lb[{j}]",
        )
        model.addConstr(
            gp.quicksum(x[i, j] for i in I) <= float(u[j]),
            name=f"size_ub[{j}]",
        )

    # Seniority requirements:

    # x_(i, Ins) <= 1 - y_i, forall i (team instructor restricted to juniors only)
    if "Ins" in J:
        for i in I:
            model.addConstr(
                x[i, "Ins"] <= 1.0 - float(y[i]),
                name=f"ins_junior_only[{i}]",
            )

    # sum_i y_i x_(i j) >= 1, forall j in J_S (at least one senior)
    for j in J_S:
        model.addConstr(
            gp.quicksum(float(y[i]) * x[i, j] for i in I) >= 1.0,
            name=f"senior_req[{j}]",
        )

    # sum_i (1 - y_i) x_(i j) >= 1, forall j in J_R (at least one junior)
    for j in J_R:
        model.addConstr(
            gp.quicksum((1.0 - float(y[i])) * x[i, j] for i in I) >= 1.0,
            name=f"junior_req[{j}]",
        )

    # Gender requirements:

    # At least one male in Services
    if "Srv" in J:
        model.addConstr(
            gp.quicksum(float(m[i]) * x[i, "Srv"] for i in I) >= 1.0,
            name="srv_male_at_least_one",
        )
        # At least one female in Services
        model.addConstr(
            gp.quicksum((1.0 - float(m[i])) * x[i, "Srv"] for i in I) >= 1.0,
            name="srv_female_at_least_one",
        )

    # Exactly specified numbers of males and females in Team Instructor
    if "Ins" in J:
        model.addConstr(
            gp.quicksum(float(m[i]) * x[i, "Ins"] for i in I) == ins_male_required,
            name="ins_male_exact",
        )
        model.addConstr(
            gp.quicksum((1.0 - float(m[i])) * x[i, "Ins"] for i in I) == ins_female_required,
            name="ins_female_exact",
        )

    # Leader requirements:

    # Exactly one leader per department in J_L
    for j in J_L:
        model.addConstr(
            gp.quicksum(r[i, j] for i in I) == 1.0,
            name=f"leader_exact_one[{j}]",
        )

    # Leader must be assigned to their department: r_(i j) <= x_(i j)
    for i in I:
        for j in J_L:
            model.addConstr(
                r[i, j] <= x[i, j],
                name=f"leader_assigned[{i},{j}]",
            )

    # Leader must be a senior: r_(i j) <= y_i
    for i in I:
        for j in J_L:
            model.addConstr(
                r[i, j] <= float(y[i]),
                name=f"leader_senior[{i},{j}]",
            )

    # Leader must have prior experience: r_(i j) <= e_i
    for i in I:
        for j in J_L:
            model.addConstr(
                r[i, j] <= float(e[i]),
                name=f"leader_experience[{i},{j}]",
            )

    # Leader cannot hold a concurrent assignment:
    # sum_j x_(i j) <= 2 - sum_{j in J_L} r_(i j), forall i in I
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, j] for j in J)
            <= 2.0 - gp.quicksum(r[i, j] for j in J_L),
            name=f"leader_no_dual_job[{i}]",
        )

    # Loading requirement:

    # Each applicant assigned to at most two departments:
    # sum_j x_(i j) <= 2, forall i in I
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, j] for j in J) <= 2.0,
            name=f"dual_job_limit[{i}]",
        )

    # No concurrent Busy assignments across phases:
    # sum_j b_(j t) x_(i j) <= 1, forall i in I, forall t in T
    for i in I:
        for t in T:
            model.addConstr(
                gp.quicksum(float(b[j][t]) * x[i, j] for j in J) <= 1.0,
                name=f"busy_conflict[{i},{t}]",
            )

    # --- Objective function ---

    # Component 1: normalized staffing gap
    # sum_j (g_j^+ + g_j^-) / (1 + sum_j max{a_j - l_j, u_j - a_j})
    denom_gap = 1.0 + sum(
        max(float(a[j]) - float(l[j]), float(u[j]) - float(a[j])) for j in J
    )
    gap_term = (1.0 / denom_gap) * gp.quicksum(gplus[j] + gminus[j] for j in J)

    # Component 2: normalized preference dissatisfaction
    # sum_i sum_j p_(i j) x_(i j) / (|J| * sum_j u_j)
    abs_J = float(len(J))
    denom_pref = abs_J * sum(float(u[j]) for j in J)
    pref_term = (1.0 / denom_pref) * gp.quicksum(
        float(p[i][j]) * x[i, j] for i in I for j in J
    )

    # Component 3: normalized unrecruitment
    # sum_i v_i / |I|
    abs_I = float(len(I))
    unrecruit_term = (1.0 / abs_I) * gp.quicksum(v_var[i] for i in I)

    # Final weighted objective:
    obj_expr = w1 * gap_term + w2 * pref_term + w3 * unrecruit_term
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # --- Solver parameters ---
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    # Optimize
    model.optimize()

    status = model.Status
    # Map common status codes to readable strings; fall back to raw code otherwise.
    if status == GRB.OPTIMAL:
        status_str = "OPTIMAL"
    elif status == GRB.INFEASIBLE:
        status_str = "INFEASIBLE"
    elif status == GRB.UNBOUNDED:
        status_str = "UNBOUNDED"
    elif status == GRB.INF_OR_UNBD:
        status_str = "INF_OR_UNBD"
    elif status == GRB.TIME_LIMIT:
        status_str = "TIME_LIMIT"
    elif status == GRB.INTERRUPTED:
        status_str = "INTERRUPTED"
    else:
        status_str = str(status)

    # If not optimal/feasible, return status only
    if status not in (GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT):
        # If infeasible, compute and print IIS for debugging
        if status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...", file=sys.stderr)
            model.computeIIS()
            print("IIS constraints:", file=sys.stderr)
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  {c.ConstrName}", file=sys.stderr)
            print("IIS variables:", file=sys.stderr)
            for v in model.getVars():
                if v.IISLB:
                    print(f"  {v.VarName} >= {v.LB}", file=sys.stderr)
                if v.IISUB:
                    print(f"  {v.VarName} <= {v.UB}", file=sys.stderr)
        
        # Could also handle feasible-but-not-optimal solutions if available.
        return {
            "status": status_str,
            "obj_value": None,
            "x": {},
            "r": {},
            "v": {},
            "gplus": {},
            "gminus": {},
        }

    # --- Extract solution ---
    solution: Dict[str, Any] = {"status": status_str}

    if model.SolCount == 0:
        solution.update(
            {
                "obj_value": None,
                "x": {},
                "r": {},
                "v": {},
                "gplus": {},
                "gminus": {},
            }
        )
        return solution

    solution["obj_value"] = model.ObjVal

    # x[i][j]
    x_sol: Dict[str, Dict[str, int]] = {}
    for i in I:
        x_sol[i] = {}
        for j in J:
            val = x[i, j].X
            x_sol[i][j] = int(round(val))

    # r[i][j]
    r_sol: Dict[str, Dict[str, int]] = {}
    for i in I:
        r_sol[i] = {}
        for j in J_L:
            val = r[i, j].X
            r_sol[i][j] = int(round(val))

    # v[i]
    v_sol: Dict[str, int] = {}
    for i in I:
        v_sol[i] = int(round(v_var[i].X))

    # gplus[j], gminus[j]
    gplus_sol: Dict[str, float] = {}
    gminus_sol: Dict[str, float] = {}
    for j in J:
        gplus_sol[j] = gplus[j].X
        gminus_sol[j] = gminus[j].X

    solution["x"] = x_sol
    solution["r"] = r_sol
    solution["v"] = v_sol
    solution["gplus"] = gplus_sol
    solution["gminus"] = gminus_sol

    # Compute preference dissatisfaction (raw and normalized) from the solution
    try:
        pref_raw = 0.0
        for i in I:
            for j in J:
                pref_raw += float(p[i][j]) * float(x_sol[i].get(j, 0))
        pref_norm = None
        if denom_pref > 0:
            pref_norm = float(pref_raw) / float(denom_pref)
    except Exception:
        pref_raw = None
        pref_norm = None

    solution["pref_raw"] = pref_raw
    solution["pref_dissatisfaction"] = pref_norm
    return solution


def main(argv: Optional[List[str]] = None) -> None:
    """
    Simple CLI: python gurobi.py input.json > output.json

    The input should be a JSON file matching the structure described in the
    module docstring. The output is a JSON document with the solution.
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1:
        print("Usage: python gurobi.py INPUT.json", file=sys.stderr)
        sys.exit(1)

    input_path = argv[0]
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solution = solve_camp_problem(data)
    print_solution(solution)


def format_solution(solution: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Status: {solution.get('status', 'UNKNOWN')}")

    # Objective value first
    obj_val = solution.get('obj_value')
    if obj_val is None:
        lines.append("Objective value: n/a")
    else:
        lines.append(f"Objective value: {float(obj_val):.6f}")

    # Leaders next
    r = solution.get('r', {})
    if r:
        lines.append("\nLeaders:")
        for i in sorted(r):
            leaders = [j for j, val in sorted(r[i].items()) if int(val) == 1]
            if leaders:
                lines.append(f"  {i}: {', '.join(leaders)}")

    # Then assignments
    x = solution.get('x', {})
    if x:
        lines.append("\nAssignments:")
        for i in sorted(x):
            assigned = [j for j, val in sorted(x[i].items()) if int(val) == 1]
            if assigned:
                lines.append(f"  {i}: {', '.join(assigned)}")
            else:
                lines.append(f"  {i}: none")

    # Staffing gaps
    gplus = solution.get('gplus', {})
    gminus = solution.get('gminus', {})
    if gplus or gminus:
        lines.append("\nStaffing gaps:")
        for j in sorted(set(list(gplus) + list(gminus))):
            lines.append(f"  {j}: +{gplus.get(j, 0)} -{gminus.get(j, 0)}")

    # Preference dissatisfaction (normalized and raw)
    pref_norm = solution.get('pref_dissatisfaction')
    pref_raw = solution.get('pref_raw')
    if pref_norm is not None or pref_raw is not None:
        lines.append("\nPreference dissatisfaction:")
        if pref_norm is not None:
            lines.append(f"  Normalized: {float(pref_norm):.6f}")
        if pref_raw is not None:
            lines.append(f"  Raw total: {float(pref_raw):.2f}")

    # Unrecruited applicants last
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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()