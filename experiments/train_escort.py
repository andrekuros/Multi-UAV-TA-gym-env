"""
Train Att-Coalition or MLP-Coalition on WPS_escort (v2 actor-critic).

Usage:
  python experiments/train_escort.py --episodes 400 --case WPS_escort
  python experiments/train_escort.py --mlp --episodes 400 --case WPS_escort
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.Hybrid.AttentionEscort import AttentionEscort
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _events
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv

OUT_DIR = os.path.join(ROOT, "dqn_Custom")


def run_episode(env, policy: AttentionEscort, hung: HungarianAllocator, explore: bool):
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
                in (
                    "Reset_Allocation",
                    "New_Threat",
                    "Agent_Fail",
                    "Escort_Created",
                    "Escort_Retired",
                )
                for ev in events
            )
        )
        actions = {}
        if should:
            result, tok, scores, noise, logits, selected = policy.plan(
                env, hung, events=events, explore=explore, force=True
            )
            for agent_name, task in result:
                if env.last_tasks_info and task in env.last_tasks_info:
                    actions[agent_name] = env.last_tasks_info.index(task)
        else:
            tok = scores = noise = logits = selected = None

        observation, reward, done, trunc, info = env.step(actions)
        s_now = float(env.compute_s_esc())
        step_r = (s_now - s_prev) / 20.0
        s_prev = s_now
        next_tok = policy.build_tokens(env)
        ep_done = all(done.values()) or all(trunc.values())
        if should and tok is not None:
            policy.push(tok, scores, noise, logits, selected, step_r, next_tok, ep_done)
            # Throttle updates: transformer AC is expensive under frequent escort replans
            if len(policy.buffer) % 4 == 0:
                policy.update(batch_size=64)
    final = info.get("metrics", {}) if isinstance(info, dict) else {}
    return (
        float(final.get("S_ESC", env.compute_s_esc())),
        float(final.get("S_WPS", env.compute_s_wps())),
        float(final.get("escort_coverage_rate", 0.0)),
        int(final.get("recon_losses", 0)),
        int(final.get("protected_rec_completed", 0)),
    )


def make_policy(args) -> AttentionEscort:
    return AttentionEscort(
        use_attention=not args.mlp,
        max_tasks=args.max_tasks,
        max_agents=args.max_agents,
        d_model=args.d_model,
        nhead=args.nhead,
        n_layers=args.n_layers,
        lr=args.lr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="WPS_escort")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--eval-eps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp", action="store_true", help="Train MLP-Coalition instead of Att-Coalition")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-tasks", type=int, default=48)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)

    tag = "MLPCoalition" if args.mlp else "AttCoalition"
    out = args.out or os.path.join(OUT_DIR, f"policy_{tag}_{args.case}_v2.pth")
    os.makedirs(OUT_DIR, exist_ok=True)

    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(CASE_SPECS[args.case], flags)
    cfg.multiple_tasks_per_agent = True

    policy = make_policy(args)
    best_score = -1e9

    for ep in range(1, args.episodes + 1):
        policy.eps = max(0.05, 0.45 - 0.4 * ep / args.episodes)
        env = MultiUAVEnv(cfg)
        hung = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
        s_esc, s_wps, cov, losses, prot = run_episode(env, policy, hung, explore=True)
        if ep % 20 == 0:
            print(
                f"[{tag}] ep={ep}/{args.episodes} S_ESC={s_esc:.1f} S_WPS={s_wps:.1f} "
                f"cov={cov:.2f} recon_loss={losses} prot={prot} eps={policy.eps:.2f}",
                flush=True,
            )
        if ep % args.eval_every == 0 or ep == args.episodes:
            policy.eps = 0.0
            evals = []
            for _ in range(args.eval_eps):
                ev = MultiUAVEnv(cfg)
                hung_e = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
                ss, _, _, _, _ = run_episode(ev, policy, hung_e, explore=False)
                evals.append(ss)
            mean_s = float(np.mean(evals))
            print(f"  EVAL S_ESC={mean_s:.1f}", flush=True)
            if mean_s > best_score:
                best_score = mean_s
                policy.save(out)
                print(f"  Best saved -> {out}", flush=True)
            policy.eps = max(0.05, 0.45 - 0.4 * ep / args.episodes)

    print(f"Done. checkpoint={out} best_score={best_score:.1f}", flush=True)
    return out, best_score


if __name__ == "__main__":
    main()
