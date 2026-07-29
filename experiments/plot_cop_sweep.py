"""Plot COP-quality sweep: sense radius + cueing delay panels."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PAPER_FIGS = ROOT / "paper" / "figures"
RESULTS_FIGS = RESULTS / "figs"


def main():
    path = RESULTS / "wps_attn_cop_sweep_summary.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run summarize_cop_sweep.py first")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        for k in ("R", "d", "Local", "Urgency", "Global", "gap_GL"):
            try:
                r[k] = float(r[k]) if r[k] not in ("", "None") else np.nan
            except ValueError:
                r[k] = np.nan

    sense = sorted([r for r in rows if r["axis"] == "sense"], key=lambda x: x["R"])
    cue = sorted([r for r in rows if r["axis"] == "cue"], key=lambda x: x["d"])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)

    def _panel(ax, pts, xkey, xlabel, title):
        if not pts:
            ax.set_title(title + " (missing)")
            return
        xs = [p[xkey] for p in pts]
        for lab, col, mk in (
            ("Local-Hungarian", "#6a1b9a", "o"),
            ("Urgency-Pair", "#2e7d32", "s"),
            ("Global-Hungarian", "#c62828", "^"),
        ):
            key = {"Local-Hungarian": "Local", "Urgency-Pair": "Urgency", "Global-Hungarian": "Global"}[lab]
            ax.plot(xs, [p[key] for p in pts], marker=mk, color=col, label=lab, linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _panel(axes[0], sense, "R", r"$R_{\mathrm{sense}}$", r"Local sensing ($d{=}18$, no share)")
    _panel(axes[1], cue, "d", r"cueing delay $d$", r"Team cueing ($R{=}0$, share)")
    axes[0].set_ylabel(r"$S_{\mathrm{WPS}}$")
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle("COP quality closes the Local–Global gap", y=1.02, fontsize=11)
    fig.tight_layout()

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGS.mkdir(parents=True, exist_ok=True)
    for dest in (PAPER_FIGS / "fig_cop_sweep.png", RESULTS_FIGS / "fig_cop_sweep.png"):
        fig.savefig(dest, bbox_inches="tight", dpi=200)
        print(f"Wrote {dest}")
    fig.savefig(PAPER_FIGS / "fig_cop_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
