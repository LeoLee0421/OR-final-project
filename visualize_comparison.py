"""
Runs both the greedy heuristic (test2.py) and Gurobi optimal solver (groubi_solve.py)
on every test_*.json file in the same directory, then visualises the results.

Panels:
  1. Objective value comparison (grouped bar)
  2. Optimality gap %  [(heuristic - optimal) / optimal * 100]
  3. Objective component breakdown  (gap / pref / unrecruit)
  4. Staffing-gap component detail  per instance
  5. Unrecruited applicants count
  6. Solve-time comparison

Usage:
    python visualize_comparison.py
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Mapping

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── solver imports ────────────────────────────────────────────────────────────
from test2 import solve_camp_problem_heuristic
from groubi_solve import solve_camp_problem as solve_gurobi


# ── helpers ───────────────────────────────────────────────────────────────────

def _denom_gap(data: Mapping[str, Any]) -> float:
    J, l, u, a = data["J"], data["l"], data["u"], data["a"]
    return 1.0 + sum(max(float(a[j]) - float(l[j]), float(u[j]) - float(a[j])) for j in J)


def _denom_pref(data: Mapping[str, Any]) -> float:
    J, u = data["J"], data["u"]
    return float(len(J)) * sum(float(u[j]) for j in J)


def compute_components(solution: Mapping[str, Any], data: Mapping[str, Any]) -> Dict[str, float]:
    """Return the three normalised objective components (before weighting)."""
    J = data["J"]
    I = data["I"]
    w1, w2, w3 = float(data["w1"]), float(data["w2"]), float(data["w3"])

    dg = _denom_gap(data)
    gplus = solution.get("gplus", {})
    gminus = solution.get("gminus", {})
    gap_norm = sum(float(gplus.get(j, 0)) + float(gminus.get(j, 0)) for j in J) / dg

    pref_norm = float(solution.get("pref_dissatisfaction") or 0.0)

    v = solution.get("v", {})
    unrecruit_norm = sum(int(v.get(i, 0)) for i in I) / float(len(I))

    return {
        "gap_norm":      gap_norm,
        "pref_norm":     pref_norm,
        "unrecruit_norm": unrecruit_norm,
        "gap_weighted":       w1 * gap_norm,
        "pref_weighted":      w2 * pref_norm,
        "unrecruit_weighted": w3 * unrecruit_norm,
        "n_unrecruited": sum(int(v.get(i, 0)) for i in I),
    }


def run_instance(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    label = os.path.splitext(os.path.basename(path))[0]
    n = len(data["I"])

    # Heuristic
    t0 = time.perf_counter()
    h_sol = solve_camp_problem_heuristic(data)
    h_time = time.perf_counter() - t0

    # Gurobi  (suppress console output regardless of JSON flag)
    data_quiet = dict(data, log_to_console=False)
    t0 = time.perf_counter()
    g_sol = solve_gurobi(data_quiet)
    g_time = time.perf_counter() - t0

    h_obj = h_sol.get("obj_value")
    g_obj = g_sol.get("obj_value")
    gap_pct = None
    if h_obj is not None and g_obj is not None and g_obj > 1e-12:
        gap_pct = (h_obj - g_obj) / g_obj * 100.0

    h_comp = compute_components(h_sol, data) if h_obj is not None else None
    g_comp = compute_components(g_sol, data) if g_obj is not None else None

    return dict(
        label=label,
        n=n,
        h_obj=h_obj,   g_obj=g_obj,
        h_time=h_time, g_time=g_time,
        gap_pct=gap_pct,
        h_comp=h_comp, g_comp=g_comp,
        h_status=h_sol.get("status"), g_status=g_sol.get("status"),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(
        glob.glob(os.path.join(base, "test_*.json")),
        key=lambda p: json.load(open(p))["I"].__len__(),
    )
    if not paths:
        print("No test_*.json files found.")
        return

    print(f"Found {len(paths)} test instances. Running solvers …\n")
    results: List[Dict[str, Any]] = []
    for p in paths:
        print(f"  {os.path.basename(p)} …", end=" ", flush=True)
        r = run_instance(p)
        results.append(r)
        g_lbl = f"{r['g_obj']:.4f}" if r["g_obj"] is not None else r["g_status"]
        h_lbl = f"{r['h_obj']:.4f}" if r["h_obj"] is not None else r["h_status"]
        gap_s = f"{r['gap_pct']:+.2f}%" if r["gap_pct"] is not None else "n/a"
        print(f"gurobi={g_lbl}  heuristic={h_lbl}  gap={gap_s}  "
              f"({r['g_time']:.1f}s / {r['h_time']:.3f}s)")

    labels = [r["label"].replace("test_", "") for r in results]
    x = np.arange(len(results))

    # ── colour palette ────────────────────────────────────────────────────────
    C_GRB  = "#2563EB"   # blue   – Gurobi
    C_HEU  = "#EA580C"   # orange – Heuristic
    C_GAP  = "#DC2626"   # red    – staffing gap component
    C_PRF  = "#16A34A"   # green  – preference component
    C_UNR  = "#7C3AED"   # purple – unrecruited component

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Heuristic vs Gurobi Optimal – Objective Comparison", fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.38)

    # ── Panel 1: objective values ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    w = 0.35
    g_objs = [r["g_obj"] if r["g_obj"] is not None else 0 for r in results]
    h_objs = [r["h_obj"] if r["h_obj"] is not None else 0 for r in results]
    b1 = ax1.bar(x - w/2, g_objs, width=w, color=C_GRB, label="Gurobi (optimal)", zorder=3)
    b2 = ax1.bar(x + w/2, h_objs, width=w, color=C_HEU, label="Heuristic", zorder=3)
    ax1.bar_label(b1, fmt="%.4f", fontsize=7, padding=2)
    ax1.bar_label(b2, fmt="%.4f", fontsize=7, padding=2)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Objective value (lower = better)")
    ax1.set_title("Objective Values")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_ylim(0, max(max(g_objs), max(h_objs)) * 1.22)

    # ── Panel 2: optimality gap % ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    gaps = [r["gap_pct"] if r["gap_pct"] is not None else float("nan") for r in results]
    colors_g = [C_HEU if g >= 0 else C_GRB for g in gaps]
    bars = ax2.bar(x, gaps, color=colors_g, zorder=3)
    ax2.bar_label(bars, fmt="%.2f%%", fontsize=7, padding=2)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("Gap (%) = (H − G) / G × 100")
    ax2.set_title("Optimality Gap")
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    # ── Panel 3: objective component breakdown – Gurobi ───────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    bar_w = 0.18
    valid = [r for r in results if r["g_comp"] and r["h_comp"]]
    xv = np.arange(len(valid))
    lv = [r["label"].replace("test_", "") for r in valid]

    def _comp(r, key, side):
        comp = r["g_comp"] if side == "g" else r["h_comp"]
        return comp[key] if comp else 0.0

    g_gap  = [_comp(r, "gap_weighted",       "g") for r in valid]
    g_prf  = [_comp(r, "pref_weighted",      "g") for r in valid]
    g_unr  = [_comp(r, "unrecruit_weighted", "g") for r in valid]
    h_gap  = [_comp(r, "gap_weighted",       "h") for r in valid]
    h_prf  = [_comp(r, "pref_weighted",      "h") for r in valid]
    h_unr  = [_comp(r, "unrecruit_weighted", "h") for r in valid]

    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

    def stacked_bar(ax, xi, bot_a, bot_b, bot_c, offset, alpha=1.0, edge="none"):
        bot = np.zeros(len(xi))
        a = ax.bar(xi + offset, bot_a, bar_w, bottom=bot, color=C_GAP, alpha=alpha, edgecolor=edge)
        bot = np.array(bot_a, dtype=float)
        b = ax.bar(xi + offset, bot_b, bar_w, bottom=bot, color=C_PRF, alpha=alpha, edgecolor=edge)
        bot += np.array(bot_b)
        c = ax.bar(xi + offset, bot_c, bar_w, bottom=bot, color=C_UNR, alpha=alpha, edgecolor=edge)
        return a, b, c

    stacked_bar(ax3, xv, g_gap, g_prf, g_unr, offsets[0])
    stacked_bar(ax3, xv, h_gap, h_prf, h_unr, offsets[1])

    # overlay total height labels
    for i, r in enumerate(valid):
        if r["g_obj"] is not None:
            ax3.text(xv[i] + offsets[0], r["g_obj"] + 0.002, f"{r['g_obj']:.3f}",
                     ha="center", va="bottom", fontsize=6.5, color=C_GRB)
        if r["h_obj"] is not None:
            ax3.text(xv[i] + offsets[1], r["h_obj"] + 0.002, f"{r['h_obj']:.3f}",
                     ha="center", va="bottom", fontsize=6.5, color=C_HEU)

    ax3.set_xticks(xv + (offsets[0]+offsets[1])/2)
    ax3.set_xticklabels(lv, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("Weighted objective component")
    ax3.set_title("Objective Breakdown per Instance  (left bar = Gurobi · right bar = Heuristic)")
    legend_patches = [
        mpatches.Patch(facecolor=C_GAP, label="w₁ · gap term"),
        mpatches.Patch(facecolor=C_PRF, label="w₂ · pref term"),
        mpatches.Patch(facecolor=C_UNR, label="w₃ · unrecruited term"),
    ]
    ax3.legend(handles=legend_patches, fontsize=8, ncol=5, loc="upper left")
    ax3.grid(axis="y", alpha=0.3)

    # ── Panel 4: unrecruited count ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    g_unr_n = [r["g_comp"]["n_unrecruited"] if r["g_comp"] else 0 for r in results]
    h_unr_n = [r["h_comp"]["n_unrecruited"] if r["h_comp"] else 0 for r in results]
    ax4.bar(x - w/2, g_unr_n, width=w, color=C_GRB, label="Gurobi")
    ax4.bar(x + w/2, h_unr_n, width=w, color=C_HEU, label="Heuristic")
    ax4.set_xticks(x); ax4.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax4.set_ylabel("Count")
    ax4.set_title("Unrecruited Applicants")
    ax4.legend(fontsize=9); ax4.grid(axis="y", alpha=0.3)
    ax4.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    # ── Panel 5: solve time ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    g_times = [r["g_time"] for r in results]
    h_times = [r["h_time"] for r in results]
    ax5.bar(x - w/2, g_times, width=w, color=C_GRB, label="Gurobi")
    ax5.bar(x + w/2, h_times, width=w, color=C_HEU, label="Heuristic")
    ax5.set_yscale("log")
    ax5.set_xticks(x); ax5.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax5.set_ylabel("Seconds (log scale)")
    ax5.set_title("Solve Time")
    ax5.legend(fontsize=9); ax5.grid(axis="y", alpha=0.3)

    # ── Summary table printed to console ─────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Instance':<22} {'n':>4}  {'Gurobi':>9}  {'Heuristic':>10}  "
          f"{'Gap %':>7}  {'G-time':>7}  {'H-time':>7}  {'G-unrec':>8}  {'H-unrec':>8}")
    print("-" * 90)
    for r in results:
        g_s = f"{r['g_obj']:.5f}" if r["g_obj"] is not None else "n/a     "
        h_s = f"{r['h_obj']:.5f}" if r["h_obj"] is not None else "n/a     "
        gp  = f"{r['gap_pct']:+.2f}%" if r["gap_pct"] is not None else "  n/a"
        gu  = r["g_comp"]["n_unrecruited"] if r["g_comp"] else "-"
        hu  = r["h_comp"]["n_unrecruited"] if r["h_comp"] else "-"
        print(f"  {r['label']:<20} {r['n']:>4}  {g_s:>9}  {h_s:>10}  "
              f"{gp:>7}  {r['g_time']:>6.2f}s  {r['h_time']:>6.3f}s  {str(gu):>8}  {str(hu):>8}")
    print("=" * 90)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
