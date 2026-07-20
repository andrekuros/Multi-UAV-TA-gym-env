"""Aggregate paper_eval CSV: mean±std tables + Wilcoxon vs CBBA (where possible)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        boots.append(float(np.mean(rng.choice(x, size=len(x), replace=True))))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(RESULTS / "paper_eval.csv"))
    parser.add_argument("--out", default=str(RESULTS / "paper_stats.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=["exp", "case", "algorithm"], keep="last")

    summary = []
    for (case, algo), g in df.groupby(["case", "algorithm"]):
        row = g.iloc[-1]
        summary.append(
            {
                "case": case,
                "label": row.get("label", case),
                "algorithm": algo,
                "mean_F_Reward": float(row["mean_F_Reward"]),
                "std_F_Reward": float(row["std_F_Reward"]),
                "episodes": int(row["episodes"]),
                "mean_makespan": float(row.get("mean_makespan", 0)),
                "mean_reallocations": float(row.get("mean_reallocations", 0)),
                "mean_decision_ms": float(row.get("mean_decision_ms", 0)),
            }
        )

    comparisons = []
    by = {(r["case"], r["algorithm"]): r for r in summary}
    for (case, algo), r in by.items():
        base = by.get((case, "CBBA"))
        if base is None or algo == "CBBA":
            continue
        delta = r["mean_F_Reward"] - base["mean_F_Reward"]
        comparisons.append(
            {
                "case": case,
                "algorithm": algo,
                "delta_vs_CBBA": delta,
                "better_than_CBBA": bool(delta > 0),
            }
        )

    # Wilcoxon / bootstrap from per-episode logs when present
    ep_path = RESULTS / "paper_eval_episodes.csv"
    wilcoxon = []
    if ep_path.exists() and stats is not None:
        ep = pd.read_csv(ep_path)
        for case in ep["case"].unique():
            sub = ep[ep["case"] == case]
            if "CBBA" not in set(sub["algorithm"]):
                continue
            base = sub[sub["algorithm"] == "CBBA"]["F_Reward"].to_numpy()
            for algo in sorted(set(sub["algorithm"]) - {"CBBA"}):
                a = sub[sub["algorithm"] == algo]["F_Reward"].to_numpy()
                n = min(len(a), len(base))
                if n < 5:
                    continue
                diff = a[:n] - base[:n]
                try:
                    w = stats.wilcoxon(a[:n], base[:n], zero_method="wilcox", alternative="two-sided")
                    p = float(w.pvalue)
                except Exception:
                    p = float("nan")
                lo, hi = bootstrap_ci(diff)
                wilcoxon.append(
                    {
                        "case": case,
                        "algorithm": algo,
                        "n": int(n),
                        "mean_delta": float(np.mean(diff)),
                        "wilcoxon_p": p,
                        "bootstrap_delta_ci95": [lo, hi],
                    }
                )

    out = {"summary": summary, "vs_CBBA": comparisons, "wilcoxon_vs_CBBA": wilcoxon}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Markdown table
    md_lines = ["# Paper evaluation summary", ""]
    cases = sorted({r["case"] for r in summary})
    for case in cases:
        md_lines.append(f"## {case}")
        md_lines.append("")
        md_lines.append("| Algorithm | F_Reward | Makespan | Realloc | Decision ms |")
        md_lines.append("|---|---:|---:|---:|---:|")
        rows = [r for r in summary if r["case"] == case]
        rows.sort(key=lambda r: -r["mean_F_Reward"])
        for r in rows:
            md_lines.append(
                f"| {r['algorithm']} | {r['mean_F_Reward']:.1f}±{r['std_F_Reward']:.1f} | "
                f"{r['mean_makespan']:.1f} | {r['mean_reallocations']:.1f} | {r['mean_decision_ms']:.3f} |"
            )
        md_lines.append("")
    md_path = RESULTS / "PAPER_RESULTS.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {args.out} and {md_path}")


if __name__ == "__main__":
    main()
