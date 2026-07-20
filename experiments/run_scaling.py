"""Agent / task scaling curves for Cap-Greedy, CBBA-Replan, Hungarian, TBTA."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.paper_eval import evaluate_case
from experiments.paper_scenarios import TBTA_E3_FLAGS, CASE_SPECS

RESULTS = os.path.join(ROOT, "experiments", "results")


def build_scaling_cases():
    """Register temporary scaling cases into CASE_SPECS."""
    cases = {}
    # Agent scaling, tasks fixed Att6 Rec12
    for n_f1, n_r1, tag in [(1, 2, "a3"), (2, 4, "a6"), (3, 6, "a9"), (4, 8, "a12")]:
        cid = f"scale_agents_{tag}"
        cases[cid] = {
            "label": f"Agents F1={n_f1} R1={n_r1}",
            "agents": {"F1": n_f1, "F2": 0, "R1": n_r1, "R2": 0},
            "tasks": {"Att": 6, "Rec": 12, "Hold": 0},
            "fail_rate": 0.0,
            "threats_list": [],
            "arrival_rate": 0.0,
        }
    # Task scaling, agents fixed F1=2 R1=4
    for n_att, n_rec, tag in [(3, 6, "t9"), (6, 12, "t18"), (9, 18, "t27"), (12, 24, "t36")]:
        cid = f"scale_tasks_{tag}"
        cases[cid] = {
            "label": f"Tasks Att={n_att} Rec={n_rec}",
            "agents": {"F1": 2, "F2": 0, "R1": 4, "R2": 0},
            "tasks": {"Att": n_att, "Rec": n_rec, "Hold": 0},
            "fail_rate": 0.0,
            "threats_list": [],
            "arrival_rate": 0.0,
        }
    CASE_SPECS.update(cases)
    return list(cases.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--out", default=os.path.join(RESULTS, "scaling_curves.csv"))
    args = parser.parse_args()

    if not args.policy:
        cand = os.path.join(ROOT, "dqn_Custom", "policy_TBTA_E3_260715-131109.pth")
        args.policy = cand if os.path.exists(cand) else None

    case_ids = build_scaling_cases()
    algos = ["Cap-Greedy", "CBBA", "CBBA-Replan", "Hungarian"]
    if args.policy:
        algos.append("TBTA")

    os.makedirs(RESULTS, exist_ok=True)
    all_rows = []
    for cid in case_ids:
        print("=" * 50, cid, flush=True)
        rows = evaluate_case(cid, algos, args.episodes, dict(TBTA_E3_FLAGS), args.policy, exp_id="scaling")
        all_rows.extend(rows)
        write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
    print(f"Done -> {args.out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
