"""
Visualise results from results.csv.

Plots produced
--------------
1. Objective value distribution (box plots) per scenario, per solver
2. Optimality gap vs Gurobi lower bound per scenario
3. Improvement over each upper-bound benchmark per scenario
4. Summary statistics table (mean ± std per solver × scenario)

Usage
-----
  python visualize.py                      # reads results.csv, saves figures/
  python visualize.py --input my.csv       # custom CSV
  python visualize.py --no-save            # show plots interactively only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 130,
})

# ── Constants ──────────────────────────────────────────────────────────────────
SCENARIO_ORDER = [
    "base",
    "high_competition",
    "leader_shortage",
    "gender_imbalance",
    "applicant_shortage",
    "applicant_excess",
]
SCENARIO_LABELS = {
    "base":               "Base",
    "high_competition":   "High\nCompetition",
    "leader_shortage":    "Leader\nShortage",
    "gender_imbalance":   "Gender\nImbalance",
    "applicant_shortage": "Applicant\nShortage",
    "applicant_excess":   "Applicant\nExcess",
}
SOLVER_ORDER  = ["gurobi", "test2", "bench1", "bench2", "bench3"]
SOLVER_LABELS = {
    "gurobi": "Gurobi",
    "test2":  "Heuristic\n(test2)",
    "bench1": "B1: Rnd\nOpt-Size",
    "bench2": "B2: Pref\nSatisf.",
    "bench3": "B3: Unrecruit\nMin.",
}
SOLVER_COLORS = {
    "gurobi": "#2196F3",
    "test2":  "#4CAF50",
    "bench1": "#FF9800",
    "bench2": "#E91E63",
    "bench3": "#9C27B0",
}

UPPER_BOUNDS = ["bench1", "bench2", "bench3"]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep only feasible rows for numeric comparisons
    df["obj_value"] = pd.to_numeric(df["obj_value"], errors="coerce")
    df["is_feasible"] = df["status"].isin(["OPTIMAL", "FEASIBLE", "TIME_LIMIT", "INTERRUPTED"])
    return df


def _scenarios_present(df: pd.DataFrame) -> List[str]:
    return [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]


def _solvers_present(df: pd.DataFrame) -> List[str]:
    return [s for s in SOLVER_ORDER if s in df["solver"].unique()]


# ── Plot 1: Objective value box plots ─────────────────────────────────────────

def plot_objective_boxplots(df: pd.DataFrame, save_dir: Optional[Path]) -> None:
    scenarios = _scenarios_present(df)
    solvers   = _solvers_present(df)
    n_sc = len(scenarios)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    fig.suptitle("Objective Value Distribution per Scenario and Solver", fontweight="bold")

    for ax, sc in zip(axes.flat, scenarios):
        sub = df[(df["scenario"] == sc) & df["is_feasible"]]
        data_per_solver = [
            sub[sub["solver"] == sv]["obj_value"].dropna().values
            for sv in solvers
        ]
        labels = [SOLVER_LABELS.get(sv, sv).replace("\n", " ") for sv in solvers]
        colors = [SOLVER_COLORS.get(sv, "#888888") for sv in solvers]

        bps = ax.boxplot(
            data_per_solver, labels=labels, patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bps["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(SCENARIO_LABELS.get(sc, sc))
        ax.set_ylabel("Objective value")
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes.flat[n_sc:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "1_objective_boxplots.png", bbox_inches="tight")
        print(f"  Saved: {save_dir}/1_objective_boxplots.png")
    else:
        plt.show()
    plt.close(fig)


# ── Plot 2: Optimality gap vs Gurobi ─────────────────────────────────────────

def plot_optimality_gap(df: pd.DataFrame, save_dir: Optional[Path]) -> None:
    if "gurobi" not in df["solver"].values:
        print("  [skip] optimality gap plot: no Gurobi results in CSV.")
        return

    scenarios = _scenarios_present(df)
    solvers_to_gap = [sv for sv in _solvers_present(df) if sv != "gurobi"]

    # Merge gurobi reference
    gurobi_ref = (
        df[df["solver"] == "gurobi"][["instance_id", "obj_value"]]
        .rename(columns={"obj_value": "gurobi_obj"})
    )
    merged = df[df["solver"].isin(solvers_to_gap)].merge(gurobi_ref, on="instance_id", how="inner")
    merged = merged[merged["is_feasible"] & merged["gurobi_obj"].notna() & (merged["gurobi_obj"] > 0)]
    merged["gap_pct"] = (merged["obj_value"] - merged["gurobi_obj"]) / merged["gurobi_obj"] * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(scenarios))
    n_sv = len(solvers_to_gap)
    width = 0.8 / n_sv

    for k, sv in enumerate(solvers_to_gap):
        means, errs = [], []
        for sc in scenarios:
            vals = merged[(merged["solver"] == sv) & (merged["scenario"] == sc)]["gap_pct"].dropna()
            means.append(vals.mean() if len(vals) else float("nan"))
            errs.append(vals.std() if len(vals) else 0.0)
        offset = (k - n_sv / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=errs,
            label=SOLVER_LABELS.get(sv, sv).replace("\n", " "),
            color=SOLVER_COLORS.get(sv, "#888"),
            alpha=0.82, capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s).replace("\n", " ") for s in scenarios])
    ax.set_ylabel("Optimality gap vs Gurobi (%)")
    ax.set_title("Optimality Gap vs Gurobi Lower Bound (mean ± std)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "2_optimality_gap.png", bbox_inches="tight")
        print(f"  Saved: {save_dir}/2_optimality_gap.png")
    else:
        plt.show()
    plt.close(fig)


# ── Plot 3: Improvement of test2 over each upper-bound benchmark ──────────────

def plot_improvement_over_benchmarks(df: pd.DataFrame, save_dir: Optional[Path]) -> None:
    if "test2" not in df["solver"].values:
        print("  [skip] improvement plot: no test2 results.")
        return

    scenarios = _scenarios_present(df)
    benchmarks_present = [b for b in UPPER_BOUNDS if b in df["solver"].values]
    if not benchmarks_present:
        print("  [skip] improvement plot: no benchmark results.")
        return

    fig, axes = plt.subplots(1, len(benchmarks_present), figsize=(6 * len(benchmarks_present), 6), sharey=True)
    if len(benchmarks_present) == 1:
        axes = [axes]

    fig.suptitle("Improvement of Heuristic (test2) over Upper-Bound Benchmarks\n(positive = test2 is better)", fontweight="bold")

    test2_ref = (
        df[df["solver"] == "test2"][["instance_id", "obj_value"]]
        .rename(columns={"obj_value": "test2_obj"})
    )

    for ax, bench in zip(axes, benchmarks_present):
        bench_df = df[df["solver"] == bench][["instance_id", "scenario", "obj_value"]].rename(
            columns={"obj_value": "bench_obj"}
        )
        merged = bench_df.merge(test2_ref, on="instance_id", how="inner")
        merged = merged[merged["bench_obj"].notna() & merged["test2_obj"].notna() & (merged["bench_obj"] > 0)]
        merged["improvement_pct"] = (merged["bench_obj"] - merged["test2_obj"]) / merged["bench_obj"] * 100

        x = np.arange(len(scenarios))
        means, errs = [], []
        for sc in scenarios:
            vals = merged[merged["scenario"] == sc]["improvement_pct"].dropna()
            means.append(vals.mean() if len(vals) else float("nan"))
            errs.append(vals.std() if len(vals) else 0.0)

        colors = [
            SOLVER_COLORS.get(bench, "#888") if m >= 0 else "#f44336"
            for m in means
        ]
        ax.bar(x, means, yerr=errs, color=colors, alpha=0.80, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s).replace("\n", " ") for s in scenarios], rotation=15, ha="right")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"vs {SOLVER_LABELS.get(bench, bench).replace(chr(10), ' ')}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    axes[0].set_ylabel("Improvement (%)")
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "3_improvement_over_benchmarks.png", bbox_inches="tight")
        print(f"  Saved: {save_dir}/3_improvement_over_benchmarks.png")
    else:
        plt.show()
    plt.close(fig)


# ── Plot 4: Summary statistics table ─────────────────────────────────────────

def plot_summary_table(df: pd.DataFrame, save_dir: Optional[Path]) -> None:
    scenarios = _scenarios_present(df)
    solvers   = _solvers_present(df)

    records = []
    for sc in scenarios:
        for sv in solvers:
            sub = df[(df["scenario"] == sc) & (df["solver"] == sv) & df["is_feasible"]]["obj_value"].dropna()
            n_total = len(df[(df["scenario"] == sc) & (df["solver"] == sv)])
            n_feas  = len(sub)
            records.append({
                "Scenario": SCENARIO_LABELS.get(sc, sc).replace("\n", " "),
                "Solver":   SOLVER_LABELS.get(sv, sv).replace("\n", " "),
                "N feasible": f"{n_feas}/{n_total}",
                "Mean obj": f"{sub.mean():.4f}" if n_feas else "–",
                "Std obj":  f"{sub.std():.4f}"  if n_feas > 1 else "–",
                "Min obj":  f"{sub.min():.4f}"  if n_feas else "–",
                "Max obj":  f"{sub.max():.4f}"  if n_feas else "–",
            })

    table_df = pd.DataFrame(records)

    # Print to stdout
    print("\n=== Summary Statistics ===")
    print(table_df.to_string(index=False))

    # Save as CSV
    if save_dir:
        table_df.to_csv(save_dir / "4_summary_statistics.csv", index=False)
        print(f"  Saved: {save_dir}/4_summary_statistics.csv")

    # Also render as a matplotlib table figure
    fig, ax = plt.subplots(figsize=(18, max(4, len(records) * 0.35 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(table_df.columns))))
    # Header styling
    for j in range(len(table_df.columns)):
        tbl[(0, j)].set_facecolor("#1565C0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Alternating row colours
    for i in range(1, len(records) + 1):
        color = "#E3F2FD" if i % 2 == 0 else "white"
        for j in range(len(table_df.columns)):
            tbl[(i, j)].set_facecolor(color)

    ax.set_title("Summary Statistics: Objective Values per Scenario × Solver",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "4_summary_table.png", bbox_inches="tight")
        print(f"  Saved: {save_dir}/4_summary_table.png")
    else:
        plt.show()
    plt.close(fig)


# ── Plot 5: Component breakdown bar chart ─────────────────────────────────────

def plot_component_breakdown(df: pd.DataFrame, save_dir: Optional[Path]) -> None:
    """Mean staffing gap, preference raw, and unrecruitment count per solver (averaged over all scenarios)."""
    solvers = _solvers_present(df)
    metrics = ["staffing_gap_raw", "pref_raw", "n_unrecruited"]
    labels  = ["Staffing Gap\n(sum |count-opt|)", "Pref. Dissatisf.\n(raw sum)", "# Unrecruited"]

    feas = df[df["is_feasible"]].copy()
    for col in metrics:
        feas[col] = pd.to_numeric(feas[col], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Mean Objective Component Values per Solver (all scenarios, feasible only)", fontweight="bold")

    x = np.arange(len(solvers))
    for ax, metric, label in zip(axes, metrics, labels):
        means, errs = [], []
        for sv in solvers:
            vals = feas[feas["solver"] == sv][metric].dropna()
            means.append(vals.mean() if len(vals) else float("nan"))
            errs.append(vals.std()  if len(vals) > 1 else 0.0)
        colors = [SOLVER_COLORS.get(sv, "#888") for sv in solvers]
        ax.bar(x, means, yerr=errs, color=colors, alpha=0.82, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SOLVER_LABELS.get(sv, sv).replace("\n", " ") for sv in solvers],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_title(label)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "5_component_breakdown.png", bbox_inches="tight")
        print(f"  Saved: {save_dir}/5_component_breakdown.png")
    else:
        plt.show()
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise camp-allocation benchmark results.")
    parser.add_argument("--input",   default="results.csv", help="Input CSV (default: results.csv)")
    parser.add_argument("--output",  default="figures",     help="Output directory for figures (default: figures/)")
    parser.add_argument("--no-save", action="store_true",   help="Show plots interactively instead of saving")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}\nRun `python run_all.py --all` first.")

    save_dir: Optional[Path] = None
    if not args.no_save:
        save_dir = Path(args.output)
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} …")
    df = load(csv_path)
    print(f"  {len(df)} rows, scenarios: {sorted(df['scenario'].unique())}")
    print(f"  Solvers: {sorted(df['solver'].unique())}")

    print("\nGenerating plots …")
    plot_objective_boxplots(df, save_dir)
    plot_optimality_gap(df, save_dir)
    plot_improvement_over_benchmarks(df, save_dir)
    plot_summary_table(df, save_dir)
    plot_component_breakdown(df, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
