"""Pareto / tradeoff plots: F_Reward vs #replans and decision time."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Hungarian": "#6a1b9a",
    "CBBA-Replan": "#0277bd",
    "CBBA": "#1565c0",
    "Cap-Greedy": "#546e7a",
    "TBTA": "#e65100",
    "TBTA-D3": "#bf360c",
    "RG-DQN": "#2e7d32",
    "RA-DQN": "#00897b",
    "Random": "#9e9e9e",
    "Greedy": "#757575",
    "Swarm-GAP": "#00897b",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(RESULTS / "hybrid_eval.csv"))
    parser.add_argument("--fallback", default=str(RESULTS / "paper_eval.csv"))
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        path = Path(args.fallback)
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["case", "algorithm"], keep="last")
    dyn = df[df["case"].astype(str).str.startswith("D")].copy()
    if dyn.empty:
        dyn = df.copy()

    replan_col = "mean_algo_replans" if "mean_algo_replans" in dyn.columns else "mean_reallocations"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for case in sorted(dyn["case"].unique()):
        sub = dyn[dyn["case"] == case]
        for _, row in sub.iterrows():
            algo = row["algorithm"]
            c = COLORS.get(algo, "#333")
            axes[0].scatter(
                float(row.get(replan_col, 0)),
                float(row["mean_F_Reward"]),
                color=c,
                s=70,
                label=f"{algo}|{case}" if case == sorted(dyn["case"].unique())[0] else None,
                alpha=0.85,
            )
            axes[1].scatter(
                float(row["mean_decision_ms"]),
                float(row["mean_F_Reward"]),
                color=c,
                s=70,
                alpha=0.85,
            )
            axes[0].annotate(algo, (float(row.get(replan_col, 0)), float(row["mean_F_Reward"])), fontsize=7, alpha=0.8)
            axes[1].annotate(algo, (float(row["mean_decision_ms"]), float(row["mean_F_Reward"])), fontsize=7, alpha=0.8)

    axes[0].set_xlabel("# Algo replans (mean)")
    axes[0].set_ylabel("F_Reward")
    axes[0].set_title("Quality vs replan count")
    axes[1].set_xlabel("Decision time (ms)")
    axes[1].set_ylabel("F_Reward")
    axes[1].set_title("Quality vs compute")
    fig.suptitle("Hybrid / dynamic tradeoffs", y=1.02)
    fig.tight_layout()
    out = FIGS / "fig_paper_pareto"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
