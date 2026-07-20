"""
Run TBTA curriculum fine-tune: CurD1 -> CurD2 -> CurD3.

Each stage saves policy_TBTA_<id>_latest.pth which the next stage loads.
Uses F_Reward checkpointing when configured in train_tbta EXPERIMENTS.

Usage:
  python experiments/run_curriculum.py --max-epoch 10 --train-env-num 6
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.train_tbta import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--train-env-num", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stages = ["CurD1", "CurD2", "CurD3"]
    for stage in stages:
        print("=" * 60, flush=True)
        print(f"Curriculum stage {stage}", flush=True)
        train(stage, max_epoch=args.max_epoch, seed=args.seed, train_env_num=args.train_env_num)
        latest = os.path.join(ROOT, "dqn_Custom", f"policy_TBTA_{stage}_latest.pth")
        if not os.path.exists(latest):
            # Fall back to newest matching run checkpoint
            dqn_dir = os.path.join(ROOT, "dqn_Custom")
            cands = sorted(
                [f for f in os.listdir(dqn_dir) if f.startswith(f"policy_TBTA_{stage}_") and f.endswith(".pth")],
                reverse=True,
            )
            if cands:
                src = os.path.join(dqn_dir, cands[0])
                shutil.copy2(src, latest)
                print(f"Copied {src} -> {latest}", flush=True)
    print("Curriculum complete.", flush=True)


if __name__ == "__main__":
    main()
