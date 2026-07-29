"""Summarize WPS_attn final comparison (Local/CBBA/PI/Urgency/Att/MLP/GNN/Global)."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "experiments", "results")

ALGOS = [
    "Local-Hungarian",
    "Local-CBBA-Replan",
    "Local-PI",
    "Urgency-Pair",
    "Att-ContextPair",
    "MLP-ContextPair",
    "GNN-ContextPair",
    "Global-Hungarian",
]


def load(path: str):
    by = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by[r["algorithm"]][int(r["seed"])] = float(r["S_WPS"])
    return by


def paired_ci(a, b, rng, n=4000):
    keys = sorted(set(a) & set(b))
    d = np.array([a[x] - b[x] for x in keys])
    boots = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(n)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(keys)


def main():
    rng = np.random.RandomState(0)
    # Prefer multi-seed GNN pooled file, else single final episodes
    paths = []
    for s in (0, 1, 2):
        p = os.path.join(RESULTS, f"wps_attn_final_s{s}_episodes.csv")
        if os.path.isfile(p):
            paths.append((s, p))
    single = os.path.join(RESULTS, "wps_attn_final_episodes.csv")
    if not paths and os.path.isfile(single):
        paths = [(0, single)]
    if not paths:
        print("No final episode CSVs found")
        sys.exit(1)

    data = {s: load(p) for s, p in paths}
    lines = [
        "# WPS_attn final comparison",
        "",
        "Suite: WPS_attn, N=100 eval episodes. Learned methods: raw ContextPair IL",
        "(Att/MLP from ablation Raw seeds; GNN from Gnn seeds). Classical: Local-Hungarian,",
        "Local-CBBA-Replan, Local-PI (shared agent_known_ids), Urgency-Pair, Global-Hungarian.",
        "",
        "## Mean S_WPS",
        "",
        "| Method | " + " | ".join(f"seed{s}" for s, _ in paths) + " | mean |",
        "|---|" + "|".join(["---"] * (len(paths) + 1)) + "|",
    ]
    means = {}
    for algo in ALGOS:
        vals = []
        for s, _ in paths:
            if algo not in data[s]:
                vals.append(float("nan"))
            else:
                vals.append(float(np.mean(list(data[s][algo].values()))))
        finite = [v for v in vals if np.isfinite(v)]
        means[algo] = float(np.mean(finite)) if finite else float("nan")
        cell = " | ".join(f"{v:.1f}" if np.isfinite(v) else "-" for v in vals)
        lines.append(f"| {algo} | {cell} | {means[algo]:.1f} |")

    # Pool GNN across seeds if multiple; classical/Att/MLP use seed0 or average available
    base = data[paths[0][0]]
    local = base["Local-Hungarian"]

    def pooled(algo):
        if algo == "GNN-ContextPair" and len(paths) > 1:
            keys = sorted(base[algo])
            return {
                k: float(np.mean([data[s][algo][k] for s, _ in paths if algo in data[s]]))
                for k in keys
            }
        # Att/MLP: average available seeds
        if algo in ("Att-ContextPair", "MLP-ContextPair") and len(paths) > 1:
            keys = sorted(base[algo])
            return {
                k: float(np.mean([data[s][algo][k] for s, _ in paths if algo in data[s]]))
                for k in keys
            }
        return base[algo]

    lines += [
        "",
        "## Paired vs Local-Hungarian (pooled where multi-seed)",
        "",
        "| Method | mean delta | 95% CI | sig |",
        "|---|---|---|---|",
    ]
    for algo in ALGOS:
        if algo == "Local-Hungarian":
            continue
        m, lo, hi, n = paired_ci(pooled(algo), local, rng)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        lines.append(f"| {algo} - Local | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {sig} |")

    lines += [
        "",
        "## Secondary (pooled)",
        "",
        "| Contrast | mean | 95% CI | sig |",
        "|---|---|---|---|",
    ]
    gnn = pooled("GNN-ContextPair")
    for a, b, lab in (
        (gnn, pooled("MLP-ContextPair"), "GNN - MLP"),
        (gnn, pooled("Att-ContextPair"), "GNN - Att"),
        (gnn, pooled("Urgency-Pair"), "GNN - Urgency"),
        (pooled("Att-ContextPair"), pooled("MLP-ContextPair"), "Att - MLP"),
    ):
        m, lo, hi, _ = paired_ci(a, b, rng)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        lines.append(f"| {lab} | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {sig} |")

    # Gate-style honesty: GNN beats Urgency and MLP?
    m_u, lo_u, hi_u, _ = paired_ci(gnn, pooled("Urgency-Pair"), rng)
    m_m, lo_m, hi_m, _ = paired_ci(gnn, pooled("MLP-ContextPair"), rng)
    beat_urg = lo_u > 0
    beat_mlp = lo_m > 0
    lines += [
        "",
        "## Claim check",
        "",
        f"GNN beats Urgency: {'YES' if beat_urg else 'NO'} ({m_u:+.1f} [{lo_u:+.1f}, {hi_u:+.1f}])",
        f"GNN beats MLP: {'YES' if beat_mlp else 'NO'} ({m_m:+.1f} [{lo_m:+.1f}, {hi_m:+.1f}])",
        "",
    ]
    out = os.path.join(RESULTS, "WPS_ATTN_FINAL.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
