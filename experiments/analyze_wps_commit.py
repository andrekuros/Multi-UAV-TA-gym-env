"""Summarize WPS_commit eval and bootstrap Att-Commit vs MLP / Urgency."""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.Hybrid.AttentionCommit import AttentionCommit, UrgencyCommit
from experiments.wps_eval import run_wps_episode, _bootstrap_ci_diff

RESULTS = os.path.join(ROOT, "experiments", "results")


def main():
    csv_path = os.path.join(RESULTS, "wps_commit_eval.csv")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    print("=== Aggregate (from wps_commit_eval.csv) ===")
    for r in rows:
        print(
            f"{r['algorithm']:20s} S={float(r['mean_S_WPS']):8.1f}+/-{float(r['std_S_WPS']):5.1f} "
            f"ot={float(r['mean_on_time_rate']):.2f} miss={float(r['mean_missed_windows']):.1f} "
            f"dS_vs_Local CI=[{float(r['delta_S_WPS_ci_lo']):+.1f},{float(r['delta_S_WPS_ci_hi']):+.1f}]"
        )

    att_path = os.path.join(ROOT, "dqn_Custom", "policy_AttCommit_WPS_commit.pth")
    mlp_path = os.path.join(ROOT, "dqn_Custom", "policy_MLPCommit_WPS_commit.pth")
    att = AttentionCommit(use_attention=True)
    att.load(att_path)
    att.eps = 0.0
    mlp = AttentionCommit(use_attention=False)
    mlp.load(mlp_path)
    mlp.eps = 0.0
    urg = UrgencyCommit()

    n = 100
    print(f"\n=== Paired bootstrap ({n} eps) Att vs MLP / Urgency ===")
    scores = {k: [] for k in ("Att-Commit", "MLP-Commit", "Urgency-Commit", "Local-Hungarian")}
    for ep in range(n):
        for algo, kw in (
            ("Local-Hungarian", {}),
            ("Urgency-Commit", {"urg_commit": urg}),
            ("MLP-Commit", {"mlp_commit": mlp}),
            ("Att-Commit", {"att_commit": att}),
        ):
            scores[algo].append(run_wps_episode(algo, "WPS_commit", ep, **kw)["S_WPS"])
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}/{n}", flush=True)

    att_s = np.array(scores["Att-Commit"])
    mlp_s = np.array(scores["MLP-Commit"])
    urg_s = np.array(scores["Urgency-Commit"])
    loc_s = np.array(scores["Local-Hungarian"])

    pairs = [
        ("Att-Commit vs Local-Hungarian", att_s, loc_s),
        ("Att-Commit vs MLP-Commit", att_s, mlp_s),
        ("Att-Commit vs Urgency-Commit", att_s, urg_s),
        ("MLP-Commit vs Local-Hungarian", mlp_s, loc_s),
        ("Urgency-Commit vs Local-Hungarian", urg_s, loc_s),
    ]
    lines = []
    for name, a, b in pairs:
        d, lo, hi = _bootstrap_ci_diff(a, b)
        sig = "YES" if (lo > 0 or hi < 0) else "no"
        line = f"{name}: dS={d:+.1f} CI=[{lo:+.1f},{hi:+.1f}] exclude0={sig}"
        print(line, flush=True)
        lines.append(line)

    claim_mlp = _bootstrap_ci_diff(att_s, mlp_s)
    claim_urg = _bootstrap_ci_diff(att_s, urg_s)
    att_wins = claim_mlp[1] > 0 and claim_urg[1] > 0
    print(
        f"\nAttention claim (Att>MLP and Att>Urgency, CI>0): {'HOLD' if att_wins else 'FAIL'}",
        flush=True,
    )

    # Persist per-ep for reuse
    npz = os.path.join(RESULTS, "wps_commit_paired.npz")
    np.savez(npz, att=att_s, mlp=mlp_s, urg=urg_s, local=loc_s)

    out_md = os.path.join(RESULTS, "WPS_COMMIT_ANALYSIS.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# WPS_commit Attention via Commit/Hold Results\n\n")
        f.write(
            "**Suite:** dual-front + `commit_horizon=25`, `reassign_penalty=2.0`.\n"
            "**Eval:** 100 episodes. Primary metric S_WPS (includes rematch term).\n"
            "**CSV:** [`wps_commit_eval.csv`](wps_commit_eval.csv)\n\n"
        )
        f.write("## Aggregate means\n\n")
        f.write("| Method | S_WPS | on_time | miss | dS vs Local (95% CI) |\n")
        f.write("|--------|-------|---------|------|----------------------|\n")
        for r in rows:
            f.write(
                f"| {r['algorithm']} | {float(r['mean_S_WPS']):.1f} +/- {float(r['std_S_WPS']):.1f} | "
                f"{float(r['mean_on_time_rate']):.2f} | {float(r['mean_missed_windows']):.1f} | "
                f"[{float(r['delta_S_WPS_ci_lo']):+.1f}, {float(r['delta_S_WPS_ci_hi']):+.1f}] |\n"
            )
        f.write("\n## Attention claim (paired bootstrap)\n\n")
        for line in lines:
            f.write(f"- {line}\n")
        f.write(
            f"\n**Verdict:** Attention claim "
            f"{'**supported**' if att_wins else '**not supported**'}.\n"
        )
        f.write(
            "\nSuccess criteria: Att-Commit > MLP-Commit and Att-Commit > Urgency-Commit "
            "with CI excluding 0.\n"
        )
        if not att_wins:
            f.write(
                "\nInterpretation: the commit/rematch interface changes the problem, "
                "but set-attention does not separate from MLP / urgency commit rules under "
                "the same lock+Hungarian engine. Do not claim architecture wins.\n"
            )
        else:
            f.write(
                "\nInterpretation: set-attention improves multi-agent commitment selection "
                "under rematch cost beyond flat MLP and hand urgency locks.\n"
            )
        f.write(
            "\n## Reproduce\n\n```bash\n"
            "python experiments/train_att_commit.py --episodes 280 --case WPS_commit\n"
            "python experiments/train_att_commit.py --mlp --episodes 200 --case WPS_commit\n"
            "python experiments/wps_eval.py --suite WPS_commit --episodes 100 \\\n"
            "  --algorithms Local-Hungarian,Urgency-Commit,MLP-Commit,Att-Commit,Global-Hungarian \\\n"
            "  --out experiments/results/wps_commit_eval.csv --exp commit100\n"
            "python experiments/analyze_wps_commit.py\n```\n"
        )
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
