"""Generate publication-style figures from experiments/results/eval_log.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent / "results"
CSV_PATH = ROOT / "eval_log.csv"

# Professional palette (cool slate + single accent)
C = {
    "ink": "#1a1f2e",
    "muted": "#5c6578",
    "grid": "#e6e8ee",
    "spine": "#c5cad6",
    "Random": "#8b93a7",
    "Greedy": "#c4a35a",
    "Swarm-GAP": "#3d8b7a",
    "CBBA": "#2f5d8c",
    "TBTA": "#c45c26",
    "TBTA_soft": "#e8a07a",
    "line_mixed": "#2f5d8c",
    "line_none": "#c45c26",
    "line_agents": "#3d8b7a",
}

CASE_LABELS = {
    "train_mixed": "Train-mixed\n(F1/R1 + Att/Rec)",
    "scal_None": "Fixed small\n(F2×2, Att×15)",
    "scal_Agents_mid": "Agent scaling\n(F1×3, R1×6)",
}

CASE_SHORT = {
    "train_mixed": "Train-mixed",
    "scal_None": "Fixed small",
    "scal_Agents_mid": "Agent scaling",
}

EXP_LABELS = {
    "E1": "E1\nScenario",
    "E2": "E2\nEarly stop",
    "E3": "E3\nMasks",
    "E4c": "E4c\nRewards",
    "E5a": "E5a\nε-decay",
}


def load_data(path: Path):
    mean, std = {}, {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["exp"], r["case"], r["algorithm"])
            mean[key] = float(r["mean_F_Reward"])
            std[key] = float(r["std_F_Reward"])
    return mean, std


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": C["spine"],
            "axes.labelcolor": C["ink"],
            "axes.titlecolor": C["ink"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": C["muted"],
            "ytick.color": C["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": C["ink"],
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "grid.color": C["grid"],
            "grid.linewidth": 0.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def style_axes(ax, ylabel="Mean final reward  $F_{\\mathrm{Reward}}$"):
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(50))
    ax.set_ylim(0, 560)
    ax.grid(axis="y", which="major", linestyle="-", alpha=1.0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(C["spine"])


def fig_baselines_vs_best(mean, std, out: Path):
    cases = ["train_mixed", "scal_None", "scal_Agents_mid"]
    algos = ["Random", "Greedy", "Swarm-GAP", "CBBA", "TBTA"]
    # best TBTA per case among E1..E5a (and E1_none for scal_None)
    best = {}
    for case in cases:
        cands = []
        for exp in ["E1", "E2", "E3", "E4c", "E5a", "E1_none"]:
            k = (exp, case, "TBTA")
            if k in mean:
                cands.append((exp, mean[k], std[k]))
        best[case] = max(cands, key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(cases))
    n = len(algos)
    width = 0.15
    offsets = (np.arange(n) - (n - 1) / 2) * width

    for i, algo in enumerate(algos):
        vals, errs = [], []
        for case in cases:
            if algo == "TBTA":
                _, m, s = best[case]
                vals.append(m)
                errs.append(s)
            else:
                vals.append(mean[("BASE", case, algo)])
                errs.append(std[("BASE", case, algo)])
        color = C[algo]
        bars = ax.bar(
            x + offsets[i],
            vals,
            width * 0.92,
            yerr=errs,
            label="TBTA (best)" if algo == "TBTA" else algo,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            error_kw={"ecolor": C["muted"], "elinewidth": 0.9, "capsize": 2.5},
            zorder=3,
        )
        if algo == "TBTA":
            for j, case in enumerate(cases):
                exp_id = best[case][0]
                ax.text(
                    x[j] + offsets[i],
                    vals[j] + errs[j] + 12,
                    exp_id,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=C["TBTA"],
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[c] for c in cases])
    ax.set_title("Task allocation quality across evaluation scenarios", pad=10)
    style_axes(ax)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.2)
    fig.text(
        0.01,
        -0.02,
        "Error bars: ±1 std over 5 episodes. TBTA labels show the best ablation variant per scenario.",
        fontsize=7.5,
        color=C["muted"],
    )
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_ablation(mean, std, out: Path):
    exps = ["E1", "E2", "E3", "E4c", "E5a"]
    cases = ["train_mixed", "scal_None", "scal_Agents_mid"]
    styles = {
        "train_mixed": (C["line_mixed"], "o", "-"),
        "scal_None": (C["line_none"], "s", "-"),
        "scal_Agents_mid": (C["line_agents"], "^", "-"),
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = np.arange(len(exps))

    # CBBA references
    ax.axhline(
        mean[("BASE", "train_mixed", "CBBA")],
        color=C["line_mixed"],
        ls="--",
        lw=1.1,
        alpha=0.55,
        label="CBBA (train-mixed)",
    )
    ax.axhline(
        mean[("BASE", "scal_None", "CBBA")],
        color=C["line_none"],
        ls=":",
        lw=1.3,
        alpha=0.55,
        label="CBBA (fixed small)",
    )

    for case in cases:
        ys = [mean[(e, case, "TBTA")] for e in exps]
        es = [std[(e, case, "TBTA")] for e in exps]
        color, marker, ls = styles[case]
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            color=color,
            marker=marker,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.6,
            linewidth=2.0,
            linestyle=ls,
            capsize=3,
            elinewidth=0.9,
            label=CASE_SHORT[case],
            zorder=4,
        )

    # Annotate mask jump
    ax.annotate(
        "Masks enabled (E3)",
        xy=(2, mean[("E3", "train_mixed", "TBTA")]),
        xytext=(2.35, 210),
        fontsize=8,
        color=C["muted"],
        arrowprops=dict(arrowstyle="->", color=C["muted"], lw=0.9),
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([EXP_LABELS[e] for e in exps])
    ax.set_xlabel("Ablation stage")
    ax.set_title("TBTA ablation study: effect of successive improvements", pad=10)
    style_axes(ax)
    ax.legend(loc="lower right", ncol=2)
    fig.text(
        0.01,
        -0.02,
        "Shaded intentionally omitted (n=5). Markers show mean ± std. Horizontal lines: CBBA baselines.",
        fontsize=7.5,
        color=C["muted"],
    )
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_per_case(mean, std, out: Path):
    cases = ["train_mixed", "scal_None", "scal_Agents_mid"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0), sharey=True)

    for ax, case in zip(axes, cases):
        entries = [
            ("CBBA", "BASE", "CBBA", C["CBBA"]),
            ("Swarm-GAP", "BASE", "Swarm-GAP", C["Swarm-GAP"]),
            ("E1", "E1", "TBTA", C["TBTA_soft"]),
            ("E2", "E2", "TBTA", C["TBTA_soft"]),
            ("E3", "E3", "TBTA", C["TBTA"]),
            ("E4c", "E4c", "TBTA", C["TBTA"]),
            ("E5a", "E5a", "TBTA", C["TBTA_soft"]),
        ]
        if case == "scal_None":
            entries.append(("E1$_\\mathrm{none}$", "E1_none", "TBTA", C["TBTA_soft"]))

        labels, vals, errs, colors = [], [], [], []
        for lab, exp, algo, color in entries:
            k = (exp, case, algo)
            if k not in mean:
                continue
            labels.append(lab)
            vals.append(mean[k])
            errs.append(std[k])
            colors.append(color)

        xpos = np.arange(len(vals))
        ax.bar(
            xpos,
            vals,
            yerr=errs,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            width=0.78,
            error_kw={"ecolor": C["muted"], "elinewidth": 0.85, "capsize": 2},
            zorder=3,
        )
        # value labels on strong bars
        for i, (v, e) in enumerate(zip(vals, errs)):
            if labels[i] in ("E3", "E4c", "CBBA"):
                ax.text(i, v + e + 8, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=C["ink"])

        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_title(CASE_SHORT[case], pad=8)
        style_axes(ax)
        if ax is not axes[0]:
            ax.set_ylabel("")

    axes[0].legend(
        handles=[
            Patch(facecolor=C["CBBA"], edgecolor="white", label="CBBA"),
            Patch(facecolor=C["Swarm-GAP"], edgecolor="white", label="Swarm-GAP"),
            Patch(facecolor=C["TBTA"], edgecolor="white", label="TBTA (key)"),
            Patch(facecolor=C["TBTA_soft"], edgecolor="white", label="TBTA (other)"),
        ],
        loc="upper left",
        fontsize=7.5,
    )
    fig.suptitle("Per-scenario comparison of baselines and TBTA ablations", y=1.02, fontsize=12, fontweight="bold")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_summary_panel(mean, std, out: Path):
    """Single-page figure suitable for slides/paper."""
    fig = plt.figure(figsize=(10.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.28)

    # --- Top: grouped bars ---
    ax = fig.add_subplot(gs[0, :])
    cases = ["train_mixed", "scal_None", "scal_Agents_mid"]
    algos = ["Random", "Greedy", "Swarm-GAP", "CBBA", "TBTA"]
    best = {}
    for case in cases:
        cands = [(e, mean[(e, case, "TBTA")], std[(e, case, "TBTA")]) for e in ["E1", "E2", "E3", "E4c", "E5a"] if (e, case, "TBTA") in mean]
        if ("E1_none", case, "TBTA") in mean:
            cands.append(("E1_none", mean[("E1_none", case, "TBTA")], std[("E1_none", case, "TBTA")]))
        best[case] = max(cands, key=lambda t: t[1])

    x = np.arange(len(cases))
    width = 0.15
    offsets = (np.arange(len(algos)) - 2) * width
    for i, algo in enumerate(algos):
        vals, errs = [], []
        for case in cases:
            if algo == "TBTA":
                vals.append(best[case][1])
                errs.append(best[case][2])
            else:
                vals.append(mean[("BASE", case, algo)])
                errs.append(std[("BASE", case, algo)])
        ax.bar(
            x + offsets[i],
            vals,
            width * 0.92,
            yerr=errs,
            label="TBTA (best)" if algo == "TBTA" else algo,
            color=C[algo],
            edgecolor="white",
            linewidth=0.5,
            error_kw={"ecolor": C["muted"], "elinewidth": 0.8, "capsize": 2},
            zorder=3,
        )
    for j, case in enumerate(cases):
        ax.text(x[j] + offsets[-1], best[case][1] + best[case][2] + 10, best[case][0], ha="center", fontsize=7.5, color=C["TBTA"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_SHORT[c] for c in cases])
    ax.set_title("(a) Baselines vs best TBTA policy", loc="left", fontsize=11)
    style_axes(ax)
    ax.legend(loc="upper left", ncol=5, fontsize=7.5)

    # --- Bottom left: ablation ---
    ax2 = fig.add_subplot(gs[1, 0])
    exps = ["E1", "E2", "E3", "E4c", "E5a"]
    xs = np.arange(len(exps))
    for case, color, marker in [
        ("train_mixed", C["line_mixed"], "o"),
        ("scal_None", C["line_none"], "s"),
        ("scal_Agents_mid", C["line_agents"], "^"),
    ]:
        ys = [mean[(e, case, "TBTA")] for e in exps]
        es = [std[(e, case, "TBTA")] for e in exps]
        ax2.errorbar(
            xs, ys, yerr=es, color=color, marker=marker, markersize=6,
            markerfacecolor="white", markeredgewidth=1.4, linewidth=1.8, capsize=2.5, label=CASE_SHORT[case],
        )
    ax2.axhline(mean[("BASE", "train_mixed", "CBBA")], color=C["line_mixed"], ls="--", lw=1, alpha=0.5)
    ax2.axhline(mean[("BASE", "scal_None", "CBBA")], color=C["line_none"], ls=":", lw=1.2, alpha=0.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(exps)
    ax2.set_xlabel("Ablation stage")
    ax2.set_title("(b) Ablation trajectory", loc="left", fontsize=11)
    style_axes(ax2)
    ax2.legend(loc="lower right", fontsize=7.5)

    # --- Bottom right: delta vs CBBA ---
    ax3 = fig.add_subplot(gs[1, 1])
    focus = ["E3", "E4c", "E5a"]
    x3 = np.arange(len(cases))
    w = 0.24
    delta_colors = {"E3": "#a84a1c", "E4c": C["TBTA"], "E5a": C["TBTA_soft"]}
    for i, exp in enumerate(focus):
        deltas = [mean[(exp, c, "TBTA")] - mean[("BASE", c, "CBBA")] for c in cases]
        ax3.bar(
            x3 + (i - 1) * w,
            deltas,
            w * 0.92,
            label=exp,
            color=delta_colors[exp],
            edgecolor="white",
            zorder=3,
        )
        for j, d in enumerate(deltas):
            ax3.text(x3[j] + (i - 1) * w, d + (6 if d >= 0 else -14), f"{d:+.0f}", ha="center", fontsize=7, color=C["ink"])

    ax3.axhline(0, color=C["ink"], lw=0.9, alpha=0.7)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([CASE_SHORT[c] for c in cases])
    ax3.set_ylabel(r"$\Delta F_{\mathrm{Reward}}$ vs CBBA")
    ax3.set_ylim(-90, 90)
    ax3.set_title("(c) TBTA − CBBA gap", loc="left", fontsize=11)
    ax3.grid(axis="y", linestyle="-", alpha=1.0)
    ax3.set_axisbelow(True)
    ax3.legend(loc="lower left", fontsize=7.5)
    for spine in ("left", "bottom"):
        ax3.spines[spine].set_linewidth(0.9)
        ax3.spines[spine].set_color(C["spine"])

    fig.suptitle("Multi-UAV Task Allocation — RL Experiment Results", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005, "Mean ± std over 5 evaluation episodes  ·  Metric: final completed capacity  $F_{\\mathrm{Reward}}$", ha="center", fontsize=7.5, color=C["muted"])
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main():
    apply_style()
    mean, std = load_data(CSV_PATH)
    ROOT.mkdir(parents=True, exist_ok=True)

    fig_baselines_vs_best(mean, std, ROOT / "fig1_baselines_vs_tbta.png")
    fig_ablation(mean, std, ROOT / "fig2_tbta_ablation.png")
    fig_per_case(mean, std, ROOT / "fig3_per_case.png")
    fig_summary_panel(mean, std, ROOT / "fig_summary_panel.png")

    # Keep legacy names pointing to new professional figures
    import shutil

    shutil.copyfile(ROOT / "fig1_baselines_vs_tbta.png", ROOT / "final_baselines_vs_tbta.png")
    shutil.copyfile(ROOT / "fig2_tbta_ablation.png", ROOT / "final_tbta_ablation.png")
    shutil.copyfile(ROOT / "fig3_per_case.png", ROOT / "final_per_case_comparison.png")

    print("Wrote professional figures to", ROOT)
    for p in sorted(ROOT.glob("fig*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
