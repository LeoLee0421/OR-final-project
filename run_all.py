"""
Entry point: run all 5 solvers on one or all test cases and write results to CSV.

Solvers
-------
  gurobi   – groubi_solve.py  (exact MIP, lower bound reference)
  test2    – test2.py          (greedy heuristic, proposed algorithm)
  bench1   – Random Optimal Size benchmark
  bench2   – Preference Satisfaction benchmark
  bench3   – Unrecruited Minimisation benchmark

Usage
-----
  # Single file
  python run_all.py testcases/base_01.json

  # All JSON files in testcases/
  python run_all.py --all

  # Custom output path
  python run_all.py --all --output my_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# ── Import solvers ─────────────────────────────────────────────────────────────
try:
    from groubi_solve import solve_camp_problem as gurobi_solve
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("[WARNING] gurobipy not available – Gurobi solver will be skipped.", file=sys.stderr)

from test2 import solve_camp_problem_heuristic as test2_solve
from benchmarks import (
    random_optimal_size_heuristic     as bench1_solve,
    preference_satisfaction_heuristic as bench2_solve,
    unrecruited_minimization_heuristic as bench3_solve,
)

# ── CSV schema ─────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "instance_id",
    "scenario",
    "n_applicants",
    "n_seniors",
    "n_exp_seniors",
    "n_male",
    "solver",
    "status",
    "obj_value",
    "obj_gap",             # w1 * gap_norm  (staffing-gap component)
    "obj_pref",            # w2 * pref_norm  (preference component)
    "obj_unrecruit",       # w3 * unrecruit_norm  (unrecruitment component)
    "staffing_gap_raw",    # sum |count_j - a_j|
    "pref_raw",            # sum p[i][j] x[i][j]
    "n_unrecruited",       # sum v[i]
    "runtime_sec",
]


def _extract_stats(
    solution: Dict[str, Any],
    data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute derived statistics from a solution dict."""
    stats: Dict[str, Any] = {
        "status":           solution.get("status", "UNKNOWN"),
        "obj_value":        solution.get("obj_value"),
        "obj_gap":          None,
        "obj_pref":         None,
        "obj_unrecruit":    None,
        "staffing_gap_raw": None,
        "pref_raw":         solution.get("pref_raw"),
        "n_unrecruited":    None,
    }

    J = list(data["J"])
    a = {j: int(data["a"][j]) for j in J}
    l = {j: int(data["l"][j]) for j in J}
    u = {j: int(data["u"][j]) for j in J}
    I = list(data["I"])
    w1 = float(data.get("w1", 1.0))
    w2 = float(data.get("w2", 1.0))
    w3 = float(data.get("w3", 1.0))

    x_sol = solution.get("x", {})
    v_sol = solution.get("v", {})
    gplus = solution.get("gplus", {})
    gminus = solution.get("gminus", {})

    if gplus or gminus:
        stats["staffing_gap_raw"] = sum(
            gplus.get(j, 0) + gminus.get(j, 0) for j in J
        )
    elif x_sol:
        stats["staffing_gap_raw"] = sum(
            abs(sum(int(x_sol.get(i, {}).get(j, 0)) for i in I) - a[j])
            for j in J
        )

    if v_sol:
        stats["n_unrecruited"] = sum(int(v) for v in v_sol.values())

    # Compute per-component objective contributions
    if stats["staffing_gap_raw"] is not None:
        denom_gap = 1.0 + sum(max(float(a[j]) - float(l[j]), float(u[j]) - float(a[j])) for j in J)
        stats["obj_gap"] = round(w1 * stats["staffing_gap_raw"] / denom_gap, 8)

    pref_raw = stats["pref_raw"]
    if pref_raw is None and x_sol:
        p = data.get("p", {})
        pref_raw = sum(float(p.get(i, {}).get(j, 0)) * int(x_sol.get(i, {}).get(j, 0))
                       for i in I for j in J)
        stats["pref_raw"] = pref_raw
    if pref_raw is not None:
        denom_pref = float(len(J)) * sum(float(u[j]) for j in J)
        if denom_pref > 0:
            stats["obj_pref"] = round(w2 * pref_raw / denom_pref, 8)

    if stats["n_unrecruited"] is not None and len(I) > 0:
        stats["obj_unrecruit"] = round(w3 * stats["n_unrecruited"] / len(I), 8)

    return stats


def run_one(input_path: Path, data: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Run all solvers on one instance; return list of result rows."""
    I = list(data["I"])
    scenario   = data.get("scenario", input_path.stem.rsplit("_", 1)[0])
    instance_id = data.get("instance_id", input_path.stem)

    common = {
        "instance_id":   instance_id,
        "scenario":      scenario,
        "n_applicants":  len(I),
        "n_seniors":     sum(int(data["y"][i]) for i in I),
        "n_exp_seniors": sum(int(data["e"][i]) for i in I if int(data["y"][i]) == 1),
        "n_male":        sum(int(data["m"][i]) for i in I),
    }

    rows: list[Dict[str, Any]] = []

    solvers = [
        ("test2",  lambda d: test2_solve(d)),
        ("bench1", lambda d: bench1_solve(d, seed=42)),
        ("bench2", lambda d: bench2_solve(d)),
        ("bench3", lambda d: bench3_solve(d, seed=42)),
    ]
    if GUROBI_AVAILABLE:
        solvers = [("gurobi", lambda d: gurobi_solve(d))] + solvers

    for solver_name, solver_fn in solvers:
        print(f"  [{solver_name}] ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            sol = solver_fn(data)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sol = {"status": f"ERROR: {exc}", "obj_value": None}
        elapsed = time.perf_counter() - t0
        print(f"{sol.get('status', '?')}  obj={sol.get('obj_value'):.6f}"
              if sol.get("obj_value") is not None
              else f"{sol.get('status', '?')}")

        stats = _extract_stats(sol, data)
        row = {**common, "solver": solver_name, "runtime_sec": round(elapsed, 3), **stats}
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run all camp-allocation solvers on test cases and save CSV."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Single JSON test file.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all JSON files in testcases/.",
    )
    parser.add_argument(
        "--testcases-dir",
        default="testcases",
        help="Directory of test cases (used with --all). Default: testcases/",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV path. Default: results.csv",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting.",
    )
    args = parser.parse_args()

    if not args.all and not args.input:
        parser.error("Provide a JSON file or --all.")

    if args.all:
        tc_dir = Path(args.testcases_dir)
        if not tc_dir.exists():
            sys.exit(f"[ERROR] Test-cases directory not found: {tc_dir}")
        files = sorted(tc_dir.glob("*.json"))
        if not files:
            sys.exit(f"[ERROR] No JSON files found in {tc_dir}")
    else:
        files = [Path(args.input)]

    output_path = Path(args.output)
    mode = "a" if args.append else "w"
    write_header = (mode == "w") or (not output_path.exists())

    all_rows: list[Dict[str, Any]] = []

    for idx, fpath in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] {fpath.name}")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = run_one(fpath, data)
        all_rows.extend(rows)

    with open(output_path, mode, newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {output_path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
