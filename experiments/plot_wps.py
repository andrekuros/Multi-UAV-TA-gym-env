"""WPS paper figures: S_WPS (primary), on-time, oracle gap, ablations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Local-Cap-Greedy": "#546e7a",
    "Local-Hungarian": "#6a1b9a",
    "Local-CBBA-Replan": "#0277bd",
    "Global-Hungarian": "#c62828",
    "RAH": "#2e7d32",
    "MLP-RAH": "#2e7d32",
    "RAH-no-reserve": "#81c784",
    "Att-RAH": "#1565c0",
    "Att-RAH-no-reserve": "#64b5f6",
    "Att-RAH-no-priority": "#90caf9",
}

ORDER = [
    "Local-Cap-Greedy",
    "Local-CBBA-Replan",
    "Local-Hungarian",
    "MLP-RAH",
    "Att-RAH",
    "Att-RAH-no-reserve",
    "Att-RAH-no-priority",
    "Global-Hungarian",
]


def _algos(sub):
    present = set(sub["algorithm"])
    return [a for a in ORDER if a in present]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(RESULTS / "wps_final_eval.csv"))
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    df = pd.read_csv(path).drop_duplicates(["case", "algorithm"], keep="last")
    cases = [c for c in ["WPS_easy", "WPS_hard", "WPS_burst"] if c in set(df["case"])]
    if not cases:
        cases = sorted(df["case"].unique())

    swps_col = "mean_S_WPS" if "mean_S_WPS" in df.columns else "mean_F_Reward"
    swps_std = "std_S_WPS" if "std_S_WPS" in df.columns else "std_F_Reward"

    # Panel 1: S_WPS
    fig, axes = plt.subplots(1, len(cases), figsize=(4.8 * len(cases), 4.2), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        sub = df[df["case"] == case]
        algos = _algos(sub)
        means = [float(sub[sub.algorithm == a][swps_col].iloc[0]) for a in algos]
        stds = [float(sub[sub.algorithm == a][swps_std].iloc[0]) for a in algos]
        ax.bar(np.arange(len(algos)), means, yerr=stds, color=[COLORS.get(a, "#333") for a in algos], capsize=3)
        ax.set_xticks(np.arange(len(algos)))
        ax.set_xticklabels(algos, rotation=35, ha="right")
        ax.set_title(sub["label"].iloc[0] if "label" in sub.columns else case)
        ax.set_ylabel(r"$S_{\mathrm{WPS}}$" if case == cases[0] else "")
        ax.axhline(0, color="#888", linewidth=0.6)
    fig.suptitle(r"WPS primary score $S_{\mathrm{WPS}}$", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_wps_swps.png", bbox_inches="tight", dpi=200)
    fig.savefig(FIGS / "fig_wps_swps.pdf", bbox_inches="tight")
    plt.close(fig)

    # Panel 2: on-time
    fig, axes = plt.subplots(1, len(cases), figsize=(4.8 * len(cases), 4.0), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        sub = df[df["case"] == case]
        algos = _algos(sub)
        vals = [float(sub[sub.algorithm == a]["mean_on_time_rate"].iloc[0]) for a in algos]
        ax.bar(np.arange(len(algos)), vals, color=[COLORS.get(a, "#333") for a in algos])
        ax.set_xticks(np.arange(len(algos)))
        ax.set_xticklabels(algos, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(case)
        ax.set_ylabel("On-time rate" if case == cases[0] else "")
    fig.suptitle("On-time completion rate", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_wps_ontime.png", bbox_inches="tight", dpi=200)
    fig.savefig(FIGS / "fig_wps_ontime.pdf", bbox_inches="tight")
    plt.close(fig)

    # Misses
    fig, axes = plt.subplots(1, len(cases), figsize=(4.8 * len(cases), 4.0), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        sub = df[df["case"] == case]
        algos = _algos(sub)
        vals = [float(sub[sub.algorithm == a]["mean_missed_windows"].iloc[0]) for a in algos]
        ax.bar(np.arange(len(algos)), vals, color=[COLORS.get(a, "#333") for a in algos])
        ax.set_xticks(np.arange(len(algos)))
        ax.set_xticklabels(algos, rotation=35, ha="right")
        ax.set_title(case)
        ax.set_ylabel("Missed windows" if case == cases[0] else "")
    fig.suptitle("Missed hard windows (lower is better)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_wps_misses.png", bbox_inches="tight", dpi=200)
    fig.savefig(FIGS / "fig_wps_misses.pdf", bbox_inches="tight")
    plt.close(fig)

    # Oracle gap on WPS_hard
    hard = df[df["case"] == "WPS_hard"] if "WPS_hard" in set(df["case"]) else df
    if not hard.empty and "Global-Hungarian" in set(hard["algorithm"]):
        oracle = float(hard[hard.algorithm == "Global-Hungarian"][swps_col].iloc[0])
        algos = [a for a in _algos(hard) if a != "Global-Hungarian"]
        gaps = []
        for a in algos:
            f = float(hard[hard.algorithm == a][swps_col].iloc[0])
            gaps.append(100.0 * (oracle - f) / max(abs(oracle), 1e-6))
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.bar(np.arange(len(algos)), gaps, color=[COLORS.get(a, "#333") for a in algos])
        ax.set_xticks(np.arange(len(algos)))
        ax.set_xticklabels(algos, rotation=30, ha="right")
        ax.set_ylabel(r"Gap to Global-Hungarian on $S_{\mathrm{WPS}}$ (%)")
        ax.set_title("Oracle gap (lower is better)")
        ax.axhline(0, color="#888", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_wps_oracle_gap.png", bbox_inches="tight", dpi=200)
        fig.savefig(FIGS / "fig_wps_oracle_gap.pdf", bbox_inches="tight")
        plt.close(fig)

    # Ablation panel
    if "Att-RAH" in set(hard["algorithm"]):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        abl = ["Local-Hungarian", "Att-RAH-no-priority", "Att-RAH-no-reserve", "Att-RAH", "MLP-RAH", "Global-Hungarian"]
        abl = [a for a in abl if a in set(hard["algorithm"])]
        for ax, col, ylab in zip(
            axes,
            [swps_col, "mean_on_time_rate"],
            [r"$S_{\mathrm{WPS}}$", "On-time rate"],
        ):
            vals = [float(hard[hard.algorithm == a][col].iloc[0]) for a in abl]
            ax.bar(np.arange(len(abl)), vals, color=[COLORS.get(a, "#555") for a in abl])
            ax.set_xticks(np.arange(len(abl)))
            ax.set_xticklabels(abl, rotation=25, ha="right")
            ax.set_ylabel(ylab)
            ax.set_title(ylab)
        fig.suptitle("Ablation / comparison (WPS_hard)", y=1.02)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_wps_ablation.png", bbox_inches="tight", dpi=200)
        fig.savefig(FIGS / "fig_wps_ablation.pdf", bbox_inches="tight")
        plt.close(fig)

    # Legacy F if present
    if "mean_F_Reward" in df.columns:
        fig, axes = plt.subplots(1, len(cases), figsize=(4.8 * len(cases), 4.0), sharey=True)
        if len(cases) == 1:
            axes = [axes]
        for ax, case in zip(axes, cases):
            sub = df[df["case"] == case]
            algos = _algos(sub)
            means = [float(sub[sub.algorithm == a]["mean_F_Reward"].iloc[0]) for a in algos]
            stds = [float(sub[sub.algorithm == a]["std_F_Reward"].iloc[0]) for a in algos]
            ax.bar(np.arange(len(algos)), means, yerr=stds, color=[COLORS.get(a, "#333") for a in algos], capsize=3)
            ax.set_xticks(np.arange(len(algos)))
            ax.set_xticklabels(algos, rotation=35, ha="right")
            ax.set_title(case)
            ax.set_ylabel("F_Reward (legacy)" if case == cases[0] else "")
        fig.suptitle("Legacy F_Reward (appendix)", y=1.02)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_wps_freward.png", bbox_inches="tight", dpi=200)
        fig.savefig(FIGS / "fig_wps_freward.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote figures under {FIGS}")


if __name__ == "__main__":
    main()
