"""
Recompute S_WPS under alternate weights from episode CSVs.

Usage:
  python experiments/wps_metric_sensitivity.py \\
    --csv experiments/results/wps_attn_final_s0_episodes.csv \\
    --case WPS_attn --algo Urgency-Pair --baseline Local-Hungarian \\
    --out experiments/results/WPS_ATTN_SENS.md
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def s_wps(n_on: float, n_miss: float, dist: float, max_coord: float, b: float, p: float, c: float) -> float:
    return b * n_on - p * n_miss - c * (dist / max(max_coord, 1e-6))


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.RandomState(seed)
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True))) for _ in range(n_boot)]
    return float(np.mean(diffs)), float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )


def load_rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(ROOT, "experiments", "results", "wps_attn_final_s0_episodes.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "WPS_ATTN_SENS.md"))
    parser.add_argument("--case", default="WPS_attn")
    parser.add_argument("--algo", default="Urgency-Pair", help="Treatment algorithm")
    parser.add_argument("--baseline", default="Local-Hungarian")
    args = parser.parse_args()

    variants = [
        ("Default", 12.0, 30.0, 0.01),
        ("Miss $-20\\%$", 12.0, 24.0, 0.01),
        ("Miss $+20\\%$", 12.0, 36.0, 0.01),
        ("No distance", 12.0, 30.0, 0.0),
    ]

    if not os.path.isfile(args.csv):
        raise SystemExit(f"Missing {args.csv}")

    rows = load_rows(args.csv)
    if rows and "case" in rows[0]:
        rows = [r for r in rows if r.get("case", args.case) == args.case]

    needed = {"n_on_time", "n_missed_windows", "total_distance"}
    if not rows or not needed.issubset(rows[0].keys()):
        raise SystemExit(f"CSV missing component columns {sorted(needed)}")

    by_algo_seed: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_algo_seed[r["algorithm"]][int(float(r.get("seed", 0)))] = r

    if args.algo not in by_algo_seed or args.baseline not in by_algo_seed:
        raise SystemExit(f"Need {args.algo} and {args.baseline}; have {list(by_algo_seed)}")

    lines = [
        f"# WPS metric sensitivity ({args.case})",
        "",
        rf"Paired \(\Delta S_{{\mathrm{{WPS}}}}\) = {args.algo} $-$ {args.baseline} under alternate weights.",
        f"Source: `{os.path.basename(args.csv)}`, N={len(set(by_algo_seed[args.algo]) & set(by_algo_seed[args.baseline]))}.",
        "",
        "| Variant | $(b,p,c)$ | mean $\\Delta$ | 95\\% CI | excludes 0 |",
        "|---|---|---:|---|---|",
    ]

    seeds = sorted(set(by_algo_seed[args.algo]) & set(by_algo_seed[args.baseline]))
    rows_tex = []
    for name, b, p, c in variants:
        diffs = []
        for s in seeds:
            ra = by_algo_seed[args.algo][s]
            rl = by_algo_seed[args.baseline][s]
            max_c = float(ra.get("max_coord", rl.get("max_coord", 1000.0)))
            sa = s_wps(float(ra["n_on_time"]), float(ra["n_missed_windows"]), float(ra["total_distance"]), max_c, b, p, c)
            sl = s_wps(float(rl["n_on_time"]), float(rl["n_missed_windows"]), float(rl["total_distance"]), max_c, b, p, c)
            diffs.append(sa - sl)
        mean, lo, hi = bootstrap_ci(np.asarray(diffs, dtype=float))
        excl = "yes" if not (lo <= 0.0 <= hi) else "no"
        lines.append(f"| {name} | $({b:g},{p:g},{c:g})$ | ${mean:+.1f}$ | $[{lo:+.1f},{hi:+.1f}]$ | {excl} |")
        rows_tex.append((name, b, p, c, mean, lo, hi))

    lines += ["", f"n_episodes={len(seeds)}", ""]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

    # Also print a TeX-ready snippet path
    tex_path = args.out.replace(".md", "_tex_rows.txt")
    with open(tex_path, "w", encoding="utf-8") as f:
        for name, b, p, c, mean, lo, hi in rows_tex:
            f.write(f"{name} & $({b:g},{p:g},{c:g})$ & ${mean:+.1f}$ & $[{lo:+.1f},{hi:+.1f}]$ \\\\\n")
    print(f"Wrote {args.out}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
