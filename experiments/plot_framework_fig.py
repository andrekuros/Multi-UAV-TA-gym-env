"""Generate grayscale-safe framework stack figure for the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"


def _box(ax, xy, w, h, text, fc="#f5f5f5", ec="#333333", fontsize=8.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111111",
        wrap=True,
    )
    return (x + w / 2, y, x + w / 2, y + h)  # cx, bottom, cx, top


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color="#444444",
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Row layout (bottom → top flow left-to-right)
    # y≈3.2 top row UI; y≈1.6 main pipeline
    b_sc = _box(ax, (0.25, 1.55), 1.7, 1.1, "WPS scenario\nregistry")
    b_env = _box(ax, (2.3, 1.55), 1.9, 1.1, "MultiUAVEnv\n(PettingZoo)")
    b_core = _box(ax, (2.4, 0.15), 1.7, 0.95, "core_sim\n(Rust geometry)", fc="#eeeeee")
    b_vis = _box(ax, (4.55, 1.55), 1.85, 1.1, "Visibility map\n(agent_known_ids)")
    b_alloc = _box(ax, (6.7, 1.55), 1.55, 1.1, "Allocators\nLocal/Global\nCBBA/PI/Hybrid")
    b_score = _box(ax, (8.5, 1.55), 1.25, 1.1, r"$S_{\mathrm{WPS}}$" + "\n/ $S_{ESC}$")
    b_ui = _box(ax, (4.55, 3.35), 1.85, 1.05, "3D replay UI\n(qualitative)", fc="#f0f0f0")

    # Main arrows
    _arrow(ax, (b_sc[0] + 0.85, 2.1), (b_env[0] - 0.95, 2.1))
    _arrow(ax, (b_env[0] + 0.95, 2.1), (b_vis[0] - 0.92, 2.1))
    _arrow(ax, (b_vis[0] + 0.92, 2.1), (b_alloc[0] - 0.78, 2.1))
    _arrow(ax, (b_alloc[0] + 0.78, 2.1), (b_score[0] - 0.62, 2.1))
    # core_sim up into env
    _arrow(ax, (b_core[0], b_core[3]), (b_env[0], b_env[1]))
    # env to UI (side path)
    _arrow(ax, (b_env[0], b_env[3]), (b_ui[0] - 0.2, b_ui[1]))

    ax.text(
        5.0,
        4.75,
        "Open WPS simulation stack (training/eval headless; UI optional)",
        ha="center",
        va="center",
        fontsize=10,
        color="#111111",
    )

    fig.tight_layout(pad=0.2)
    png = OUT_DIR / "fig_framework.png"
    pdf = OUT_DIR / "fig_framework.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
