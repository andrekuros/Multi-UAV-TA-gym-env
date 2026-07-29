"""Summarize WPS_attn COP-quality sweeps (sense radius + cueing delay)."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "experiments", "results")

ALGOS = ["Local-Hungarian", "Urgency-Pair", "Global-Hungarian"]


def load_episodes(*paths: str):
    by = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        if not os.path.isfile(path):
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            case, algo, seed = r["case"], r["algorithm"], int(r["seed"])
            by[case][algo][seed] = float(r["S_WPS"])
    return by


def paired_ci(a, b, rng, n=4000):
    keys = sorted(set(a) & set(b))
    d = np.array([a[x] - b[x] for x in keys])
    boots = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(n)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def case_params(case: str):
    if "_COP_cue_d" in case:
        d = int(case.split("_COP_cue_d")[1])
        return "cue", 0, d
    if "_COP_R" in case:
        r = int(case.split("_COP_R")[1])
        return "sense", r, 18
    if "_COP_d" in case:
        d = int(case.split("_COP_d")[1])
        return "delay_local", 90, d
    return "other", None, None


def main():
    paths = [
        os.path.join(RESULTS, "wps_attn_cop_sweep_episodes.csv"),
        os.path.join(RESULTS, "wps_attn_cop_cue_episodes.csv"),
    ]
    data = load_episodes(*paths)
    if not data:
        print("Missing COP episode CSVs")
        sys.exit(1)
    rng = np.random.RandomState(0)

    lines = [
        "# WPS_attn COP-quality sweep",
        "",
        "Base mission = WPS_attn (12 agents).",
        "",
        "**Sense sweep** (`share_knowledge=False`): R in {60,90,150,250} at d=18.",
        "Under local-only sensing, `threat_delay` does not change the COP (discovery is radius-only);",
        "the local-delay panel is therefore flat and reported as a negative control.",
        "",
        "**Cueing sweep** (`share_knowledge=True`, R=0): team-wide reveal after delay d in {0,6,12,18}.",
        "d=0 is the AWACS endpoint (Local = Global).",
        "",
        "Methods: Local-Hungarian, Urgency-Pair, Global-Hungarian. N=100.",
        "",
        "## Mean S_WPS by cell",
        "",
        "| Case | axis | R | d | Local | Urgency | Global | Global-Local |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    rows_plot = []
    for case in sorted(data.keys()):
        axis, r, d = case_params(case)
        means = {}
        for algo in ALGOS:
            vals = list(data[case].get(algo, {}).values())
            means[algo] = float(np.mean(vals)) if vals else float("nan")
        gap = means["Global-Hungarian"] - means["Local-Hungarian"]
        lines.append(
            f"| {case} | {axis} | {r} | {d} | {means['Local-Hungarian']:.1f} | "
            f"{means['Urgency-Pair']:.1f} | {means['Global-Hungarian']:.1f} | {gap:+.1f} |"
        )
        rows_plot.append(
            {
                "case": case,
                "axis": axis,
                "R": r if r is not None else "",
                "d": d if d is not None else "",
                "Local": means["Local-Hungarian"],
                "Urgency": means["Urgency-Pair"],
                "Global": means["Global-Hungarian"],
                "gap_GL": gap,
            }
        )

    lines += [
        "",
        "## Sense: Local vs baseline R=90",
        "",
        "| Case | Local - Local@R90 | 95% CI |",
        "|---|---|---|",
    ]
    base_case = "WPS_attn_COP_R90"
    if base_case in data:
        local_base = data[base_case]["Local-Hungarian"]
        for case in sorted(c for c in data if case_params(c)[0] == "sense"):
            if case == base_case:
                continue
            m, lo, hi = paired_ci(data[case]["Local-Hungarian"], local_base, rng)
            lines.append(f"| {case} | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")

    lines += [
        "",
        "## Cueing: Local vs Global gap",
        "",
        "| Case | Global - Local | 95% CI |",
        "|---|---|---|",
    ]
    for case in sorted(c for c in data if case_params(c)[0] == "cue"):
        m, lo, hi = paired_ci(
            data[case]["Global-Hungarian"], data[case]["Local-Hungarian"], rng
        )
        lines.append(f"| {case} | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")

    lines += [
        "",
        "## Urgency vs Local (sense + cue cells)",
        "",
        "| Case | mean delta | 95% CI | sig |",
        "|---|---|---|---|",
    ]
    for case in sorted(data.keys()):
        axis = case_params(case)[0]
        if axis not in ("sense", "cue"):
            continue
        if "Urgency-Pair" not in data[case] or "Local-Hungarian" not in data[case]:
            continue
        m, lo, hi = paired_ci(data[case]["Urgency-Pair"], data[case]["Local-Hungarian"], rng)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        lines.append(f"| {case} | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {sig} |")

    lines += [
        "",
        "## Takeaway",
        "",
        "- Increasing R_sense closes Local toward Global; Global is flat (oracle).",
        "- Cueing delay under team broadcast: shorter d shrinks the Local-Global gap; d=0 => Local=Global.",
        "- Local-only delay panel is flat (negative control: delay unused without sharing).",
        "- Urgency stays >= Local on incomplete sense cells; at large R it can exceed myopic Global.",
        "",
    ]
    out = os.path.join(RESULTS, "WPS_ATTN_COP_SWEEP.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    plot_csv = os.path.join(RESULTS, "wps_attn_cop_sweep_summary.csv")
    with open(plot_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_plot[0].keys()))
        w.writeheader()
        w.writerows(rows_plot)
    print(f"Wrote {out}")
    print(f"Wrote {plot_csv}")


if __name__ == "__main__":
    main()
