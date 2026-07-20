"""Publication figures: dynamic suite comparison + ILP optimality gap."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# Consistent algorithm colors for the paper
ALGO_COLORS = {
    "Random": "#9e9e9e",
    "Greedy": "#757575",
    "Cap-Greedy": "#546e7a",
    "Swarm-GAP": "#00897b",
    "CBBA": "#1565c0",
    "CBBA-Replan": "#0277bd",
    "Hungarian": "#6a1b9a",
    "ILP-Oracle": "#c62828",
    "TBTA": "#e65100",
    "TBTA-D3": "#bf360c",
    "TBTA-E3": "#e65100",
    "TBTA-E4c": "#ff8f00",
}

PREFERRED_ORDER = [
    "Random",
    "Greedy",
    "Cap-Greedy",
    "Swarm-GAP",
    "CBBA",
    "CBBA-Replan",
    "Hungarian",
    "TBTA",
    "TBTA-D3",
    "ILP-Oracle",
]


def _style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _order(algos):
    rank = {a: i for i, a in enumerate(PREFERRED_ORDER)}
    return sorted(algos, key=lambda a: rank.get(a, 100))


def plot_dynamic_panel(df: pd.DataFrame, out: Path):
    """Bar charts for D1/D2/D3 F_Reward across algorithms."""
    dyn = df[df["case"].astype(str).str.startswith("D")].copy()
    if dyn.empty:
        print("No dynamic rows; skip dynamic panel")
        return
    cases = [c for c in ["D1_attrition", "D2_popup_threats", "D3_combined"] if c in set(dyn["case"])]
    algos = _order(dyn["algorithm"].unique().tolist())
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4.2), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        sub = dyn[dyn["case"] == case]
        label = sub["label"].iloc[0] if "label" in sub.columns else case
        means, stds, colors = [], [], []
        for a in algos:
            row = sub[sub["algorithm"] == a]
            if row.empty:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(float(row["mean_F_Reward"].iloc[0]))
                stds.append(float(row["std_F_Reward"].iloc[0]))
            colors.append(ALGO_COLORS.get(a, "#333333"))
        x = np.arange(len(algos))
        ax.bar(x, means, yerr=stds, color=colors, capsize=3, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha="right")
        ax.set_title(label)
        ax.set_ylabel("F_Reward" if case == cases[0] else "")
        ax.axhline(0, color="#ccc", linewidth=0.5)
    fig.suptitle("Dynamic task allocation stress tests", y=1.02)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}")


def plot_static_panel(df: pd.DataFrame, out: Path):
    st = df[~df["case"].astype(str).str.startswith("D")].copy()
    if st.empty:
        return
    cases = [c for c in ["static_strike", "recon_strike_mix", "agent_scaling_mid"] if c in set(st["case"])]
    if not cases:
        cases = sorted(st["case"].unique())
    algos = _order(st["algorithm"].unique().tolist())
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4.2), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        sub = st[st["case"] == case]
        label = sub["label"].iloc[0] if "label" in sub.columns else case
        means, stds, colors = [], [], []
        for a in algos:
            row = sub[sub["algorithm"] == a]
            if row.empty:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(float(row["mean_F_Reward"].iloc[0]))
                stds.append(float(row.get("std_F_Reward", pd.Series([0])).iloc[0]))
            colors.append(ALGO_COLORS.get(a, "#333333"))
        x = np.arange(len(algos))
        ax.bar(x, means, yerr=stds, color=colors, capsize=3, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha="right")
        ax.set_title(label)
        ax.set_ylabel("F_Reward" if case == cases[0] else "")
    fig.suptitle("Static comparison (mean ± std)", y=1.02)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}")


def plot_multimetric(df: pd.DataFrame, out: Path):
    """Makespan / reallocations / decision time on recon_strike_mix or D3."""
    preferred = "D3_combined" if "D3_combined" in set(df["case"]) else None
    if preferred is None:
        for c in ["recon_strike_mix", "static_strike"]:
            if c in set(df["case"]):
                preferred = c
                break
    if preferred is None:
        return
    sub = df[df["case"] == preferred].copy()
    algos = _order(sub["algorithm"].unique().tolist())
    metrics = [
        ("mean_makespan", "Makespan"),
        ("mean_reallocations", "# Reallocations"),
        ("mean_decision_ms", "Decision time (ms)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (col, title) in zip(axes, metrics):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        vals = []
        colors = []
        for a in algos:
            row = sub[sub["algorithm"] == a]
            vals.append(float(row[col].iloc[0]) if not row.empty else np.nan)
            colors.append(ALGO_COLORS.get(a, "#333"))
        ax.bar(np.arange(len(algos)), vals, color=colors, edgecolor="white")
        ax.set_xticks(np.arange(len(algos)))
        ax.set_xticklabels(algos, rotation=45, ha="right")
        ax.set_title(title)
    fig.suptitle(f"Secondary metrics — {preferred}", y=1.03)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}")


def plot_ilp_gap(gap_path: Path, eval_path: Path, out: Path):
    if not gap_path.exists():
        print(f"No ILP gap file at {gap_path}")
        return
    gap = pd.read_csv(gap_path)
    # Mean ILP F_Reward
    ilp_f = float(gap["F_Reward"].mean()) if "F_Reward" in gap.columns else float("nan")
    ilp_obj = float(gap["ilp_objective"].mean()) if "ilp_objective" in gap.columns else float("nan")

    # Compare against algorithms on static_strike from paper_eval
    means = {"ILP-Oracle": ilp_f}
    if eval_path.exists():
        df = pd.read_csv(eval_path)
        st = df[df["case"] == "static_strike"]
        for _, row in st.iterrows():
            means[row["algorithm"]] = float(row["mean_F_Reward"])

    algos = _order(list(means.keys()))
    vals = [means[a] for a in algos]
    gaps = [100.0 * (ilp_f - v) / max(ilp_f, 1e-6) if np.isfinite(ilp_f) else np.nan for v in vals]
    colors = [ALGO_COLORS.get(a, "#333") for a in algos]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(np.arange(len(algos)), vals, color=colors, edgecolor="white")
    axes[0].set_xticks(np.arange(len(algos)))
    axes[0].set_xticklabels(algos, rotation=45, ha="right")
    axes[0].set_ylabel("F_Reward")
    axes[0].set_title("Static Strike reward")
    if np.isfinite(ilp_obj):
        axes[0].text(0.02, 0.98, f"ILP obj={ilp_obj:.2f}", transform=axes[0].transAxes, va="top", fontsize=9)

    axes[1].bar(np.arange(len(algos)), gaps, color=colors, edgecolor="white")
    axes[1].set_xticks(np.arange(len(algos)))
    axes[1].set_xticklabels(algos, rotation=45, ha="right")
    axes[1].set_ylabel("Gap to ILP (%)")
    axes[1].set_title("Optimality gap (↑ worse)")
    axes[1].axhline(0, color="#888", linewidth=0.8)
    fig.suptitle("Small-instance ILP reference", y=1.02)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}")


def plot_scaling(df: pd.DataFrame, out: Path, scaling_csv: Path | None = None):
    """Plot agent/task scaling curves when available; else static trio."""
    scale_df = df
    if scaling_csv and scaling_csv.exists():
        scale_df = pd.read_csv(scaling_csv)
    agent_cases = sorted([c for c in scale_df["case"].unique() if str(c).startswith("scale_agents_")])
    task_cases = sorted([c for c in scale_df["case"].unique() if str(c).startswith("scale_tasks_")])
    if not agent_cases and not task_cases:
        scale_cases = [c for c in ["static_strike", "recon_strike_mix", "agent_scaling_mid"] if c in set(df["case"])]
        if len(scale_cases) < 2:
            return
        _plot_scale_line(df, scale_cases, out, "Scaling / scenario difficulty")
        return
    n = int(bool(agent_cases)) + int(bool(task_cases))
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4))
    if n == 1:
        axes = [axes]
    i = 0
    if agent_cases:
        _plot_scale_line(scale_df, agent_cases, None, "Agent scaling", ax=axes[i])
        i += 1
    if task_cases:
        _plot_scale_line(scale_df, task_cases, None, "Task scaling", ax=axes[i])
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}")


def _plot_scale_line(df, scale_cases, out, title, ax=None):
    key = [a for a in ["Cap-Greedy", "CBBA", "CBBA-Replan", "Hungarian", "TBTA", "TBTA-D3"] if a in set(df["algorithm"])]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(scale_cases))
    for a in key:
        ys, es = [], []
        for c in scale_cases:
            row = df[(df["case"] == c) & (df["algorithm"] == a)]
            ys.append(float(row["mean_F_Reward"].iloc[0]) if not row.empty else np.nan)
            es.append(float(row["std_F_Reward"].iloc[0]) if not row.empty else 0)
        ax.errorbar(x, ys, yerr=es, marker="o", label=a, color=ALGO_COLORS.get(a, None), capsize=3)
    ax.set_xticks(x)
    labels = []
    for c in scale_cases:
        sub = df[df["case"] == c]
        labels.append(sub["label"].iloc[0] if "label" in sub.columns and not sub.empty else c)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("F_Reward")
    ax.set_title(title)
    ax.legend(frameon=False)
    if own:
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(RESULTS / "paper_eval.csv"))
    parser.add_argument("--ilp", default=str(RESULTS / "ilp_gap.csv"))
    args = parser.parse_args()
    _style()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Missing {args.csv}; run paper_eval first.")

    df = pd.read_csv(args.csv)
    # Keep latest exp rows if duplicates: prefer last occurrence per (case, algorithm)
    df = df.drop_duplicates(subset=["case", "algorithm"], keep="last")

    plot_static_panel(df, FIGS / "fig_paper_static")
    plot_dynamic_panel(df, FIGS / "fig_paper_dynamic")
    plot_multimetric(df, FIGS / "fig_paper_metrics")
    plot_scaling(df, FIGS / "fig_paper_scaling", scaling_csv=RESULTS / "scaling_curves.csv")
    plot_ilp_gap(Path(args.ilp), Path(args.csv), FIGS / "fig_paper_ilp_gap")
    print(f"Figures in {FIGS}")


if __name__ == "__main__":
    main()
