"""
Recompute S_WPS under alternate weights for Att-RAH vs Local-Hungarian.

Preferred input: episode-level CSV with columns
  algorithm, seed, n_on_time, n_missed_windows, total_distance, max_coord
  (or S components if already logged).

Fallback: if only mean S_WPS is available, print a note that full sensitivity
requires episode logs from wps_eval.py.

Usage:
  python experiments/wps_metric_sensitivity.py \\
    --csv experiments/results/wps_final_eval_episodes.csv \\
    --out experiments/results/wps_metric_sensitivity.md
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

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
    parser.add_argument("--csv", default=os.path.join(ROOT, "experiments", "results", "wps_final_eval_episodes.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "wps_metric_sensitivity.md"))
    parser.add_argument("--case", default="WPS_hard")
    args = parser.parse_args()

    variants = [
        ("default", 12.0, 30.0, 0.01),
        ("miss_minus_20pct", 12.0, 24.0, 0.01),
        ("miss_plus_20pct", 12.0, 36.0, 0.01),
        ("no_distance", 12.0, 30.0, 0.0),
    ]

    if not os.path.isfile(args.csv):
        msg = (
            f"# WPS metric sensitivity\n\n"
            f"Episode CSV not found at `{args.csv}`.\n\n"
            f"Regenerate with `wps_eval.py` exporting per-episode "
            f"`n_on_time`, `n_missed_windows`, `total_distance`, `max_coord` "
            f"for Att-RAH and Local-Hungarian on {args.case}.\n"
        )
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(msg)
        print(msg)
        return

    rows = load_rows(args.csv)
    # Filter case if present
    if rows and "case" in rows[0]:
        rows = [r for r in rows if r.get("case", args.case) == args.case]

    needed = {"n_on_time", "n_missed_windows", "total_distance"}
    if not rows or not needed.issubset(rows[0].keys()):
        msg = (
            "# WPS metric sensitivity\n\n"
            "CSV found but missing component columns "
            f"{sorted(needed)}. Cannot reweight offline.\n"
        )
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(msg)
        print(msg)
        return

    by_algo_seed: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        algo = r["algorithm"]
        seed = int(float(r.get("seed", r.get("episode", 0))))
        by_algo_seed[algo][seed] = r

    att_name = next((a for a in by_algo_seed if "Att-RAH" in a or a == "Att-RAH"), None)
    loc_name = next((a for a in by_algo_seed if "Local-Hung" in a or a == "Local-Hungarian"), None)
    if not att_name or not loc_name:
        print("Could not find Att-RAH / Local-Hungarian in CSV algorithms:", list(by_algo_seed))
        return

    lines = [
        f"# WPS metric sensitivity ({args.case})",
        "",
        f"Algorithms: {att_name} vs {loc_name}",
        "",
        "| Variant | (b,p,c) | mean Δ | 95% CI | excludes 0 |",
        "|---------|---------|--------|--------|------------|",
    ]

    seeds = sorted(set(by_algo_seed[att_name]) & set(by_algo_seed[loc_name]))
    for name, b, p, c in variants:
        diffs = []
        for s in seeds:
            ra = by_algo_seed[att_name][s]
            rl = by_algo_seed[loc_name][s]
            max_c = float(ra.get("max_coord", rl.get("max_coord", 1000.0)))
            sa = s_wps(
                float(ra["n_on_time"]),
                float(ra["n_missed_windows"]),
                float(ra["total_distance"]),
                max_c,
                b,
                p,
                c,
            )
            sl = s_wps(
                float(rl["n_on_time"]),
                float(rl["n_missed_windows"]),
                float(rl["total_distance"]),
                max_c,
                b,
                p,
                c,
            )
            diffs.append(sa - sl)
        mean, lo, hi = bootstrap_ci(np.asarray(diffs, dtype=float))
        excl = not (lo <= 0.0 <= hi)
        lines.append(
            f"| {name} | ({b:g},{p:g},{c:g}) | {mean:.2f} | [{lo:.2f},{hi:.2f}] | {excl} |"
        )

    lines.append("")
    lines.append(f"n_seeds={len(seeds)}")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
