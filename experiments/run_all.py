"""
Autonomous experiment runner: train E1..E5, evaluate vs baselines, write final report.

CPU-oriented defaults (this machine has no CUDA):
  --max-epoch 40 --train-env-num 4

Usage:
  python experiments/run_all.py
  python experiments/run_all.py --max-epoch 40 --episodes 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.train_tbta import EXPERIMENTS, train
from experiments.eval_tbta import CASES, append_csv, evaluate

RESULTS_DIR = os.path.join(ROOT, "experiments", "results")
LOG_CSV = os.path.join(RESULTS_DIR, "eval_log.csv")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "final_summary.json")
REPORT_MD = os.path.join(RESULTS_DIR, "FINAL_RESULTS.md")

# Ordered ablation path (CPU-feasible subset; E5b/E5d optional via --exps)
EXP_ORDER = ["E1", "E2", "E3", "E4c", "E5a", "E1_none"]

EVAL_CASES = ["train_mixed", "scal_None", "scal_Agents_mid"]


def latest_checkpoint(exp_id: str) -> str | None:
    pattern = os.path.join(ROOT, "dqn_Custom", f"policy_TBTA_{exp_id}_*.pth")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def env_flags_for_exp(exp_id: str) -> dict:
    exp = EXPERIMENTS[exp_id]
    return {
        "early_terminate": exp["early_terminate"],
        "capability_mask": exp["capability_mask"],
        "saturate_mask": exp["saturate_mask"],
        "reward_weights": exp["reward_weights"],
    }


def run_baselines(episodes: int) -> list:
    rows = []
    for case in EVAL_CASES:
        rows.extend(
            evaluate(
                case_name=case,
                episodes=episodes,
                policy_path=None,
                env_flags={"early_terminate": True, "capability_mask": False, "saturate_mask": False},
                algorithms=["Random", "Greedy", "Swarm-GAP", "CBBA"],
                exp_id="BASE",
            )
        )
    append_csv(LOG_CSV, rows)
    return rows


def run_tbta_eval(exp_id: str, policy_path: str, episodes: int) -> list:
    flags = env_flags_for_exp(exp_id)
    rows = []
    for case in EVAL_CASES:
        # Prefer cases matching training distribution
        if exp_id == "E1_none" and case != "scal_None":
            continue
        if exp_id.startswith("E") and exp_id != "E1_none" and case == "scal_None":
            # still eval scal_None for transfer signal
            pass
        rows.extend(
            evaluate(
                case_name=case,
                episodes=episodes,
                policy_path=policy_path,
                env_flags=flags,
                algorithms=["TBTA"],
                exp_id=exp_id,
            )
        )
    if rows:
        append_csv(LOG_CSV, rows)
    return rows


def pick_best(rows: list) -> dict:
    """Best TBTA on train_mixed by mean_F_Reward."""
    tbta = [r for r in rows if r["algorithm"] == "TBTA" and r["case"] == "train_mixed"]
    if not tbta:
        tbta = [r for r in rows if r["algorithm"] == "TBTA"]
    if not tbta:
        return {}
    return max(tbta, key=lambda r: r["mean_F_Reward"])


def write_report(all_rows: list, started: str, elapsed_h: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    swarm = [r for r in all_rows if r["algorithm"] == "Swarm-GAP"]
    cbba = [r for r in all_rows if r["algorithm"] == "CBBA"]
    tbta = [r for r in all_rows if r["algorithm"] == "TBTA"]

    lines = [
        "# Final RL Experiment Results",
        "",
        f"- Started: {started}",
        f"- Elapsed hours: {elapsed_h:.2f}",
        f"- Log: `{LOG_CSV}`",
        "",
        "## Baseline reference (early_terminate=True)",
        "",
        "| Case | Algorithm | mean F_Reward | std |",
        "|------|-----------|---------------|-----|",
    ]
    for r in all_rows:
        if r["exp"] == "BASE":
            lines.append(
                f"| {r['case']} | {r['algorithm']} | {r['mean_F_Reward']:.2f} | {r['std_F_Reward']:.2f} |"
            )

    lines.extend(["", "## TBTA checkpoints", "", "| Exp | Case | mean F_Reward | std | policy |", "|-----|------|---------------|-----|--------|"])
    for r in tbta:
        lines.append(
            f"| {r['exp']} | {r['case']} | {r['mean_F_Reward']:.2f} | {r['std_F_Reward']:.2f} | `{os.path.basename(r['policy'])}` |"
        )

    best = pick_best(all_rows)
    lines.extend(["", "## Verdict", ""])
    if best:
        # Compare to Swarm-GAP on same case
        gap = next((r for r in swarm if r["case"] == best["case"]), None)
        cbb = next((r for r in cbba if r["case"] == best["case"]), None)
        lines.append(f"- Best TBTA: **{best['exp']}** on `{best['case']}` -> F_Reward **{best['mean_F_Reward']:.2f}**")
        if gap:
            delta = best["mean_F_Reward"] - gap["mean_F_Reward"]
            lines.append(f"- vs Swarm-GAP ({gap['mean_F_Reward']:.2f}): **{delta:+.2f}**")
        if cbb:
            delta = best["mean_F_Reward"] - cbb["mean_F_Reward"]
            lines.append(f"- vs CBBA ({cbb['mean_F_Reward']:.2f}): **{delta:+.2f}**")
        lines.append(f"- Checkpoint: `{best['policy']}`")
    else:
        lines.append("- No TBTA rows collected.")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump({"best": best, "rows": all_rows, "elapsed_hours": elapsed_h}, f, indent=2)

    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoch", type=int, default=30, help="Epochs per experiment (CPU default 30)")
    parser.add_argument("--train-env-num", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5, help="Eval episodes per algo/case")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true", help="Only eval existing checkpoints")
    parser.add_argument("--exps", default=",".join(EXP_ORDER), help="Comma-separated experiment ids")
    parser.add_argument("--step-per-epoch", type=int, default=3000)
    parser.add_argument("--step-per-collect", type=int, default=1500)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Fresh log for this full run
    if os.path.exists(LOG_CSV):
        bak = LOG_CSV.replace(".csv", f"_{datetime.now().strftime('%y%m%d-%H%M%S')}.csv")
        os.replace(LOG_CSV, bak)
        print(f"Archived previous log -> {bak}", flush=True)

    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    all_rows = []
    exp_ids = [e.strip() for e in args.exps.split(",") if e.strip()]

    print("=" * 60, flush=True)
    print("Phase 0: Baseline evaluation", flush=True)
    print("=" * 60, flush=True)
    try:
        all_rows.extend(run_baselines(args.episodes))
    except Exception:
        traceback.print_exc()
        print("Baseline eval failed; continuing.", flush=True)

    for exp_id in exp_ids:
        print("=" * 60, flush=True)
        print(f"Experiment {exp_id}: {EXPERIMENTS[exp_id]['desc']}", flush=True)
        print("=" * 60, flush=True)
        policy_path = None
        try:
            if not args.skip_train:
                train(
                    exp_id,
                    max_epoch=args.max_epoch,
                    seed=args.seed,
                    train_env_num=args.train_env_num,
                    step_per_epoch=args.step_per_epoch,
                    step_per_collect=args.step_per_collect,
                )
            policy_path = latest_checkpoint(exp_id)
            if not policy_path:
                print(f"No checkpoint for {exp_id}; skipping eval.", flush=True)
                continue
            print(f"Evaluating {policy_path}", flush=True)
            rows = run_tbta_eval(exp_id, policy_path, args.episodes)
            all_rows.extend(rows)
            # Intermediate report after each exp
            write_report(all_rows, started, (time.time() - t0) / 3600.0)
        except Exception:
            traceback.print_exc()
            print(f"Experiment {exp_id} failed; continuing.", flush=True)
            # Try eval if a partial best was saved mid-run
            policy_path = latest_checkpoint(exp_id)
            if policy_path:
                try:
                    rows = run_tbta_eval(exp_id, policy_path, args.episodes)
                    all_rows.extend(rows)
                except Exception:
                    traceback.print_exc()

    write_report(all_rows, started, (time.time() - t0) / 3600.0)
    print(f"\nDone. Report: {REPORT_MD}", flush=True)


if __name__ == "__main__":
    main()
