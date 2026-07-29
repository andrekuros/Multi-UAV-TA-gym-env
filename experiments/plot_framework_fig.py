"""Generate grayscale-safe framework stack figure for the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"


def _box(ax, xy, w, h, title, body="", fc="#f7f7f7", ec="#222222", title_size=8.2, body_size=6.8):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.15,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color="#333333",
            linespacing=1.15,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color="#111111",
        )
    return x + w / 2, y, x + w / 2, y + h, x, x + w


def _arrow(ax, start, end, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=1.15,
            color="#333333",
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # Background lanes
    ax.add_patch(Rectangle((0.15, 2.55), 11.9, 1.55, facecolor="#fafafa", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.15, 0.85), 11.9, 1.45, facecolor="#f3f3f3", edgecolor="none", zorder=0))

    ax.text(0.25, 3.98, "End-to-end eval pipeline (identical visibility contract)", fontsize=7.2, color="#555555", ha="left")
    ax.text(0.25, 2.12, "Geometry assist", fontsize=7.2, color="#555555", ha="left")
    ax.text(0.25, 0.18, "Optional UI  ·  headless training/eval", fontsize=7.2, color="#555555", ha="left")

    # Main pipeline (top lane)
    y, h = 2.75, 1.15
    b1 = _box(ax, (0.35, y), 1.85, h, "1. WPS registry", "locked suites\nattn / AWACS / hard")
    b2 = _box(ax, (2.45, y), 2.15, h, "2. MultiUAVEnv", "PettingZoo + dynamics\nwindows / arrivals", fc="#f0f0f0")
    b3 = _box(ax, (4.85, y), 2.15, h, "3. Visibility / COP", r"$\mathcal{K}_{a,t}$ via $R$, $d$, share" + "\nagent_known_ids", fc="#ebebeb")
    b4 = _box(ax, (7.25, y), 2.35, h, "4. Allocators", "Local/Global Hung.\nCBBA / PI / hybrids", fc="#f0f0f0")
    b5 = _box(ax, (9.85, y), 1.95, h, "5. Scores", r"$S_{\mathrm{WPS}}$, $S_{\mathrm{ESC}}$" + "\neval harness", fc="#e8e8e8")

    # Geometry assist under env
    b_core = _box(ax, (2.55, 1.05), 1.95, 0.85, "core_sim", "Rust geometry assist", fc="#ffffff", title_size=7.5, body_size=6.4)

    # Optional UI
    b_ui = _box(
        ax,
        (4.85, 0.35),
        2.15,
        0.85,
        "3D replay UI",
        "qualitative only\n(not used for metrics)",
        fc="#ffffff",
        ec="#666666",
        title_size=7.5,
        body_size=6.2,
    )

    # Training note
    b_train = _box(
        ax,
        (7.55, 0.35),
        3.9,
        0.85,
        "Training / eval (headless)",
        "IL / RL scripts · paired CIs · same $\\mathcal{K}_{a,t}$ contract",
        fc="#ffffff",
        title_size=7.5,
        body_size=6.2,
    )

    # Pipeline arrows
    mid = y + h / 2
    _arrow(ax, (b1[4] + 1.85, mid), (b2[4], mid))
    _arrow(ax, (b2[4] + 2.15, mid), (b3[4], mid))
    _arrow(ax, (b3[4] + 2.15, mid), (b4[4], mid))
    _arrow(ax, (b4[4] + 2.35, mid), (b5[4], mid))
    # core -> env
    _arrow(ax, (b_core[0], b_core[3]), (b2[0], b2[1]))
    # env -> UI (optional)
    _arrow(ax, (b2[0], b2[1]), (b_ui[0], b_ui[3]))
    # scores -> training note
    _arrow(ax, (b5[0], b5[1]), (b_train[0] + 0.6, b_train[3]))

    ax.text(
        6.1,
        4.85,
        "Open WPS simulation stack",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        6.1,
        4.50,
        "Scenario registry → masked PettingZoo env → classical/hybrid allocators → scores  |  UI optional",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#444444",
    )

    fig.tight_layout(pad=0.15)
    png = OUT_DIR / "fig_framework.png"
    pdf = OUT_DIR / "fig_framework.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
