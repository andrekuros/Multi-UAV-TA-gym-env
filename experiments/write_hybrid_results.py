"""Write HYBRID_RESULTS.md from hybrid_eval.csv."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "experiments" / "results" / "hybrid_eval.csv"
df = pd.read_csv(csv_path).drop_duplicates(["case", "algorithm"], keep="last")
lines = [
    "# Hybrid dynamic results (30 eps)",
    "",
    "RG-DQN = Replan-Gate (RL Hold/Replan + Hungarian).",
    "RA-DQN = Residual Assignment (Hungarian proposal + optional Cap-Greedy override).",
    "",
]
for case in sorted(df.case.unique()):
    lines.append(f"## {case}")
    lines.append("")
    lines.append("| Algorithm | F_Reward | Replans | Decision ms |")
    lines.append("|---|---:|---:|---:|")
    sub = df[df.case == case].sort_values("mean_F_Reward", ascending=False)
    for _, r in sub.iterrows():
        rp = r["mean_algo_replans"] if "mean_algo_replans" in r.index else float("nan")
        lines.append(
            f"| {r['algorithm']} | {r['mean_F_Reward']:.1f}+/-{r['std_F_Reward']:.1f} | "
            f"{rp:.1f} | {r['mean_decision_ms']:.3f} |"
        )
    lines.append("")
out = ROOT / "experiments" / "results" / "HYBRID_RESULTS.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}")
print(df[df.case == "D3_combined"][["algorithm", "mean_F_Reward", "mean_algo_replans"]].sort_values("mean_F_Reward", ascending=False).to_string(index=False))
