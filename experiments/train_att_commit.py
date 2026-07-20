"""
Train Att-Commit or MLP-Commit on WPS_commit (dual-front + rematch penalty).

Usage:
  python experiments/train_att_commit.py --episodes 280 --case WPS_commit
  python experiments/train_att_commit.py --mlp --episodes 200 --case WPS_commit
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.Hybrid.AttentionCommit import AttentionCommit
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _events
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv

OUT_DIR = os.path.join(ROOT, "dqn_Custom")


def run_episode(env, policy: AttentionCommit, hung: HungarianAllocator, explore: bool):
    observation, info = env.reset()
    done = {a: False for a in env.agents}
    trunc = {a: False for a in env.agents}
    s_prev = 0.0
    while not all(done.values()) and not all(trunc.values()):
        events = _events(info)
        should = (
            env.time_steps == 0
            or env.time_steps % 12 == 0
            or any(
                (ev[0] if isinstance(ev, (list, tuple)) else ev)
                in ("Reset_Allocation", "New_Threat", "Agent_Fail")
                for ev in events
            )
        )
        actions = {}
        tok = policy.build_tokens(env)
        pri = np.zeros(policy.max_tasks, dtype=np.float32)
        com = np.zeros(policy.max_agents, dtype=np.float32)
        if should:
            pri, com = policy.act(tok, explore=explore)
            result, _, _, _ = policy._plan_from_scores(
                env, hung, tok, pri, com, events=events, force=True
            )
            for agent_name, task in result:
                if env.last_tasks_info and task in env.last_tasks_info:
                    actions[agent_name] = env.last_tasks_info.index(task)

        observation, reward, done, trunc, info = env.step(actions)
        s_now = float(env.compute_s_wps())
        step_r = (s_now - s_prev) / 20.0
        s_prev = s_now
        next_tok = policy.build_tokens(env)
        ep_done = all(done.values()) or all(trunc.values())
        if should:
            policy.push(tok, pri, com, step_r, next_tok, ep_done)
            policy.update(batch_size=64)
    final = info.get("metrics", {}) if isinstance(info, dict) else {}
    return (
        float(final.get("S_WPS", env.compute_s_wps())),
        float(final.get("on_time_rate", 0.0)),
        int(final.get("n_missed_windows", 0)),
        int(final.get("n_task_switches", getattr(env, "n_task_switches", 0))),
        float(final.get("F_Reward", env.F_Reward)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="WPS_commit")
    parser.add_argument("--episodes", type=int, default=280)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--eval-eps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp", action="store_true", help="Train MLP-Commit instead of Att-Commit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)

    tag = "MLPCommit" if args.mlp else "AttCommit"
    out = args.out or os.path.join(OUT_DIR, f"policy_{tag}_{args.case}.pth")
    os.makedirs(OUT_DIR, exist_ok=True)

    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(CASE_SPECS[args.case], flags)
    cfg.multiple_tasks_per_agent = True

    policy = AttentionCommit(use_attention=not args.mlp)
    best_score = -1e9

    for ep in range(1, args.episodes + 1):
        policy.eps = max(0.05, 0.45 - 0.4 * ep / args.episodes)
        env = MultiUAVEnv(cfg)
        hung = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
        s_wps, ot, miss, switches, f_legacy = run_episode(env, policy, hung, explore=True)
        if ep % 20 == 0:
            print(
                f"[{tag}] ep={ep}/{args.episodes} S_WPS={s_wps:.1f} on_time={ot:.2f} "
                f"miss={miss} switches={switches} eps={policy.eps:.2f}",
                flush=True,
            )
        if ep % args.eval_every == 0 or ep == args.episodes:
            policy.eps = 0.0
            evals_s, evals_ot = [], []
            for _ in range(args.eval_eps):
                ev = MultiUAVEnv(cfg)
                hung_e = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
                ss, oot, _, _, _ = run_episode(ev, policy, hung_e, explore=False)
                evals_s.append(ss)
                evals_ot.append(oot)
            mean_s = float(np.mean(evals_s))
            mean_ot = float(np.mean(evals_ot))
            score = mean_s + 100.0 * mean_ot
            print(f"  EVAL S_WPS={mean_s:.1f} on_time={mean_ot:.2f} score={score:.1f}", flush=True)
            if score > best_score:
                best_score = score
                policy.save(out)
                print(f"  Best saved -> {out}", flush=True)
            policy.eps = max(0.05, 0.45 - 0.4 * ep / args.episodes)

    print(f"Done. checkpoint={out} best_score={best_score:.1f}", flush=True)


if __name__ == "__main__":
    main()
