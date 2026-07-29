"""Summarize WPS_attn scale-transfer episode CSVs into WPS_ATTN_SCALE.md."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "experiments", "results")


def load_seed(path: str):
    """case -> algorithm -> {eval_seed: S_WPS}"""
    by = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by[r["case"]][r["algorithm"]][int(r["seed"])] = float(r["S_WPS"])
    return by


def paired_ci(a, b, rng, n=4000):
    keys = sorted(set(a) & set(b))
    d = np.array([a[x] - b[x] for x in keys])
    boots = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(n)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    rng = np.random.RandomState(0)
    suites = ["WPS_attn", "WPS_attn_L", "WPS_attn_XL"]
    train_seeds = [0, 1, 2]
    data = {}  # (suite, train_seed) -> algorithm -> {ep: S_WPS}
    missing = []
    for s in train_seeds:
        path = os.path.join(RESULTS, f"wps_scale_s{s}_episodes.csv")
        if not os.path.isfile(path):
            missing.append(path)
            continue
        by_case = load_seed(path)
        for suite in suites:
            if suite not in by_case:
                missing.append(f"{path}:{suite}")
                continue
            data[(suite, s)] = by_case[suite]
    if missing:
        print("Missing:", *missing, sep="\n  ")
        sys.exit(1)

    lines = [
        "# WPS_attn scale + transfer (ContextPair raw, pad 48/64)",
        "",
        "Train: IL on WPS_attn (12 agents), `--raw --context`, max_agents=48, max_tasks=64,",
        "900 episodes, il-batch=8, warmup=50, seeds 0–2.",
        "Eval: N=100 zero-shot transfer to WPS_attn / WPS_attn_L (~30) / WPS_attn_XL (~40).",
        "",
        "## Mean S_WPS by training seed",
        "",
        "| Suite | Method | seed0 | seed1 | seed2 | mean | sd |",
        "|---|---|---|---|---|---|---|",
    ]
    for suite in suites:
        for algo in ("Local-Hungarian", "Urgency-Pair", "Att-ContextPair", "MLP-ContextPair"):
            vals = [float(np.mean(list(data[(suite, s)][algo].values()))) for s in train_seeds]
            lines.append(
                f"| {suite} | {algo} | {vals[0]:.1f} | {vals[1]:.1f} | {vals[2]:.1f} | "
                f"{np.mean(vals):.1f} | {np.std(vals, ddof=1):.1f} |"
            )

    lines += [
        "",
        "## Paired Att - MLP (N=100 per training seed)",
        "",
        "| Suite | seed | mean | 95% CI | sig |",
        "|---|---|---|---|---|",
    ]
    gate_pass = True
    for suite in suites:
        for s in train_seeds:
            by = data[(suite, s)]
            m, lo, hi = paired_ci(by["Att-ContextPair"], by["MLP-ContextPair"], rng)
            sig = lo > 0 or hi < 0
            if suite == "WPS_attn_XL" and not (lo > 0):
                gate_pass = False
            lines.append(
                f"| {suite} | {s} | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {'yes' if sig else 'no'} |"
            )

    lines += [
        "",
        "## Pooled over training seeds (average 3 nets, then pair)",
        "",
        "| Suite | Contrast | mean | 95% CI | sig |",
        "|---|---|---|---|---|",
    ]
    for suite in suites:
        att = {
            ep: float(np.mean([data[(suite, s)]["Att-ContextPair"][ep] for s in train_seeds]))
            for ep in data[(suite, 0)]["Att-ContextPair"]
        }
        mlp = {
            ep: float(np.mean([data[(suite, s)]["MLP-ContextPair"][ep] for s in train_seeds]))
            for ep in data[(suite, 0)]["MLP-ContextPair"]
        }
        base = data[(suite, 0)]
        for a, b, lab in (
            (att, base["Local-Hungarian"], "Att - Local"),
            (mlp, base["Local-Hungarian"], "MLP - Local"),
            (att, base["Urgency-Pair"], "Att - Urgency"),
            (mlp, base["Urgency-Pair"], "MLP - Urgency"),
            (att, mlp, "Att - MLP"),
        ):
            m, lo, hi = paired_ci(a, b, rng)
            sig = lo > 0 or hi < 0
            if suite == "WPS_attn_XL" and lab == "Att - MLP" and not (lo > 0):
                gate_pass = False
            lines.append(
                f"| {suite} | {lab} | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {'yes' if sig else 'no'} |"
            )

    # Both Att and MLP should still beat Local on XL (pooled)
    att_xl = {
        ep: float(np.mean([data[("WPS_attn_XL", s)]["Att-ContextPair"][ep] for s in train_seeds]))
        for ep in data[("WPS_attn_XL", 0)]["Att-ContextPair"]
    }
    mlp_xl = {
        ep: float(np.mean([data[("WPS_attn_XL", s)]["MLP-ContextPair"][ep] for s in train_seeds]))
        for ep in data[("WPS_attn_XL", 0)]["MLP-ContextPair"]
    }
    loc_xl = data[("WPS_attn_XL", 0)]["Local-Hungarian"]
    for lab, a in (("Att", att_xl), ("MLP", mlp_xl)):
        m, lo, hi = paired_ci(a, loc_xl, rng)
        if lo <= 0:
            gate_pass = False
            lines.append(f"| WPS_attn_XL | {lab} - Local gate check | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] | fail |")

    lines += [
        "",
        "## Gate",
        "",
        "Primary gate: on WPS_attn_XL, Att - MLP 95% CI excludes 0 in Att's favor "
        "(and both beat Local).",
        "",
        f"**Gate {'PASSED' if gate_pass else 'FAILED'}.**",
        "",
    ]
    out = os.path.join(RESULTS, "WPS_ATTN_SCALE.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("GATE_PASS" if gate_pass else "GATE_FAIL")


if __name__ == "__main__":
    main()
