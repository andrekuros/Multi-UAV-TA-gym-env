"""
Small focused grid search for Att-Coalition v2 on WPS_escort.

Grid (~8 runs): d_model ∈ {64,128}, n_layers ∈ {2,3}, lr ∈ {1e-3, 3e-4}
Short train: 120 eps + 8-ep eval. Promote top-2 Att configs to full 400-ep
(+ matching MLP with same tokens/loss hyperparams).

Usage:
  python experiments/search_att_escort.py
  python experiments/search_att_escort.py --skip-full   # search only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from itertools import product

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "dqn_Custom")
RESULTS = os.path.join(ROOT, "experiments", "results")


def run_train(args_list: list) -> float:
    """Run train_escort.py with streamed stdout; parse best_score."""
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "train_escort.py")] + args_list
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    out_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        out_lines.append(line)
    rc = proc.wait()
    out = "".join(out_lines)
    if rc != 0:
        raise RuntimeError(f"train failed rc={rc}")
    best = None
    for line in out.splitlines():
        if "best_score=" in line:
            try:
                best = float(line.split("best_score=")[-1].strip())
            except ValueError:
                pass
    if best is None:
        raise RuntimeError("Could not parse best_score from training output")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="WPS_escort")
    parser.add_argument("--short-episodes", type=int, default=120)
    parser.add_argument("--full-episodes", type=int, default=400)
    parser.add_argument("--eval-eps-short", type=int, default=8)
    parser.add_argument("--eval-every-short", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    grid = list(
        product(
            [64, 128],  # d_model
            [2, 3],  # n_layers
            [1e-3, 3e-4],  # lr
        )
    )
    rows = []
    for d_model, n_layers, lr in grid:
        tag = f"d{d_model}_L{n_layers}_lr{lr:g}"
        out = os.path.join(OUT_DIR, f"policy_AttCoalition_{args.case}_search_{tag}.pth")
        score = run_train(
            [
                "--case",
                args.case,
                "--episodes",
                str(args.short_episodes),
                "--eval-every",
                str(args.eval_every_short),
                "--eval-eps",
                str(args.eval_eps_short),
                "--seed",
                str(args.seed),
                "--d-model",
                str(d_model),
                "--n-layers",
                str(n_layers),
                "--lr",
                str(lr),
                "--out",
                out,
            ]
        )
        rows.append(
            {
                "tag": tag,
                "d_model": d_model,
                "n_layers": n_layers,
                "lr": lr,
                "short_score": score,
                "ckpt": out,
            }
        )
        print(f"SEARCH {tag} -> {score:.1f}", flush=True)

    rows.sort(key=lambda r: r["short_score"], reverse=True)
    csv_path = os.path.join(RESULTS, f"att_escort_search_{args.case}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    top = rows[: args.top_k]
    summary = {
        "grid_size": len(rows),
        "ranking": rows,
        "promoted": top,
        "csv": csv_path,
    }
    json_path = os.path.join(RESULTS, f"att_escort_search_{args.case}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Search done. ranking -> {csv_path}", flush=True)
    for i, r in enumerate(top):
        print(f"  top{i+1}: {r['tag']} S_ESC={r['short_score']:.1f}", flush=True)

    if args.skip_full:
        return

    # Full retrain of best Att + matching MLP
    best = top[0]
    att_out = os.path.join(OUT_DIR, f"policy_AttCoalition_{args.case}_v2.pth")
    mlp_out = os.path.join(OUT_DIR, f"policy_MLPCoalition_{args.case}_v2.pth")
    print("Full retrain Att with best config...", flush=True)
    att_score = run_train(
        [
            "--case",
            args.case,
            "--episodes",
            str(args.full_episodes),
            "--eval-every",
            "40",
            "--eval-eps",
            "12",
            "--seed",
            str(args.seed),
            "--d-model",
            str(best["d_model"]),
            "--n-layers",
            str(best["n_layers"]),
            "--lr",
            str(best["lr"]),
            "--out",
            att_out,
        ]
    )
    print("Full retrain MLP (matched tokens/loss)...", flush=True)
    mlp_score = run_train(
        [
            "--case",
            args.case,
            "--episodes",
            str(args.full_episodes),
            "--eval-every",
            "40",
            "--eval-eps",
            "12",
            "--seed",
            str(args.seed),
            "--mlp",
            "--d-model",
            str(best["d_model"]),
            "--n-layers",
            str(best["n_layers"]),
            "--lr",
            str(best["lr"]),
            "--out",
            mlp_out,
        ]
    )
    summary["full"] = {
        "best_config": best,
        "att_ckpt": att_out,
        "att_score": att_score,
        "mlp_ckpt": mlp_out,
        "mlp_score": mlp_score,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Full done. Att={att_score:.1f} MLP={mlp_score:.1f}\n"
        f"  att={att_out}\n  mlp={mlp_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
