"""
Train WPS pair-cost hybrids: Phase A imitation of Global-Hungarian on visible edges,
then Phase B RL fine-tune on delta S_WPS.

Usage:
  python experiments/train_pair_cost.py --phase il --episodes 240 --case WPS_hard
  python experiments/train_pair_cost.py --phase il --mlp --episodes 240 --case WPS_hard
  python experiments/train_pair_cost.py --phase rl --episodes 200 --init dqn_Custom/policy_AttPair_WPS_hard_il.pth
  python experiments/train_pair_cost.py --phase rl --mlp --episodes 200 --init dqn_Custom/policy_MLPPair_WPS_hard_il.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.Hybrid.ContextPairHybrid import ContextPairHybrid
from TaskAllocation.Hybrid.PairCostHybrid import PairCostHybrid
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _events, _open_tasks
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv

OUT_DIR = os.path.join(ROOT, "dqn_Custom")


def _should_replan(env, events, interval=20):
    return (
        env.time_steps == 0
        or env.time_steps % interval == 0
        or any(
            (ev[0] if isinstance(ev, (list, tuple)) else ev)
            in ("Reset_Allocation", "New_Threat", "Agent_Fail")
            for ev in events
        )
    )


def _apply_assign(env, pairs):
    actions = {}
    for agent_name, task in pairs:
        if env.last_tasks_info and task in env.last_tasks_info:
            actions[agent_name] = env.last_tasks_info.index(task)
    return actions


def _expert_mask(tok: dict, expert_pairs) -> np.ndarray:
    mask = np.zeros((tok["agent_feats"].shape[0], tok["task_feats"].shape[0]), dtype=np.float32)
    name_to_i = {
        a.name: i
        for i, a in enumerate(tok["live"][: mask.shape[0]])
        if not tok["agent_mask"][i]
    }
    tid_to_j = {tid: j for j, tid in enumerate(tok["task_ids"])}
    for agent_name, task in expert_pairs:
        i = name_to_i.get(agent_name)
        j = tid_to_j.get(getattr(task, "id", None))
        if i is None or j is None:
            continue
        if tok["edge_valid"][i, j] < 0.5:
            continue  # never imitate through the visibility mask
        mask[i, j] = 1.0
    return mask


def eval_local_swps(cfg, policy: PairCostHybrid, n: int = 12) -> float:
    scores = []
    policy_eps = getattr(policy, "eps", 0.0)
    policy.eps = 0.0
    for seed in range(n):
        env = MultiUAVEnv(cfg)
        hung = HungarianAllocator(replan_interval=20, max_coord=env.max_coord)
        observation, info = env.reset(seed=seed)
        done = {a: False for a in env.agents}
        trunc = {a: False for a in env.agents}
        while not all(done.values()) and not all(trunc.values()):
            events = _events(info)
            actions = {}
            if _should_replan(env, events):
                result, *_ = policy.plan(env, hung, events=events, explore=False, force=True)
                actions = _apply_assign(env, result)
            observation, reward, done, trunc, info = env.step(actions)
        final = info.get("metrics", {}) if isinstance(info, dict) else {}
        scores.append(float(final.get("S_WPS", env.compute_s_wps())))
    policy.eps = policy_eps
    return float(np.mean(scores))


def run_il_episode(env, policy: PairCostHybrid, hung_local: HungarianAllocator, hung_global: HungarianAllocator):
    observation, info = env.reset()
    done = {a: False for a in env.agents}
    trunc = {a: False for a in env.agents}
    losses = []
    while not all(done.values()) and not all(trunc.values()):
        events = _events(info)
        actions = {}
        if _should_replan(env, events):
            # Expert: Global-Hungarian (no visibility mask)
            expert = hung_global.allocate_tasks(
                env.get_live_agents(),
                _open_tasks(env),
                time_step=env.time_steps,
                events=events,
                force=True,
            )
            tok = policy.build_tokens(env)
            expert_mask = _expert_mask(tok, expert)
            if expert_mask.sum() > 0 and tok["edge_valid"].sum() > 0:
                losses.append(policy.imitation_loss(tok, expert_mask))
            # Rollout uses Global expert actions so the episode stays on-policy for the teacher
            actions = _apply_assign(env, expert)
        observation, reward, done, trunc, info = env.step(actions)
    return float(np.mean(losses) if losses else 0.0)


def run_rl_episode(env, policy: PairCostHybrid, hung: HungarianAllocator, explore: bool):
    observation, info = env.reset()
    done = {a: False for a in env.agents}
    trunc = {a: False for a in env.agents}
    s_prev = float(env.compute_s_wps())
    while not all(done.values()) and not all(trunc.values()):
        events = _events(info)
        actions = {}
        tok = scores = noise = logits = selected = None
        if _should_replan(env, events):
            result, tok, scores, noise, logits, selected = policy.plan(
                env, hung, events=events, explore=explore, force=True
            )
            actions = _apply_assign(env, result)
        observation, reward, done, trunc, info = env.step(actions)
        s_now = float(env.compute_s_wps())
        step_r = (s_now - s_prev) / 20.0
        s_prev = s_now
        ep_done = all(done.values()) or all(trunc.values())
        if tok is not None:
            next_tok = policy.build_tokens(env)
            policy.push(tok, scores, noise, logits, selected, step_r, next_tok, ep_done)
            if len(policy.buffer) % 2 == 0:
                policy.update(batch_size=64)
    final = info.get("metrics", {}) if isinstance(info, dict) else {}
    return float(final.get("S_WPS", env.compute_s_wps()))


def make_policy(args):
    cls = ContextPairHybrid if args.context else PairCostHybrid
    return cls(
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
    parser.add_argument("--case", default="WPS_hard")
    parser.add_argument("--phase", choices=["il", "rl"], default="il")
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--eval-eps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument(
        "--context",
        action="store_true",
        help="Use Att/MLP ContextPair (context encoder + edge scores)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--init", default=None, help="Checkpoint to load (required for sensible RL phase)")
    parser.add_argument("--max-tasks", type=int, default=32)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)

    if args.context:
        tag = "MLPContextPair" if args.mlp else "AttContextPair"
        if args.case == "WPS_hard" and args.phase == "il":
            # plan default for context claim
            pass
    else:
        tag = "MLPPair" if args.mlp else "AttPair"
    phase_tag = args.phase
    out = args.out or os.path.join(OUT_DIR, f"policy_{tag}_{args.case}_{phase_tag}.pth")
    os.makedirs(OUT_DIR, exist_ok=True)

    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(CASE_SPECS[args.case], flags)
    cfg.multiple_tasks_per_agent = True

    policy = make_policy(args)
    if args.init:
        policy.load(args.init)
        print(f"Loaded init -> {args.init}", flush=True)
    elif args.phase == "rl":
        il_path = os.path.join(OUT_DIR, f"policy_{tag}_{args.case}_il.pth")
        if os.path.isfile(il_path):
            policy.load(il_path)
            print(f"Loaded IL init -> {il_path}", flush=True)

    best_score = -1e9

    for ep in range(1, args.episodes + 1):
        env = MultiUAVEnv(cfg)
        if args.phase == "il":
            hung_l = HungarianAllocator(replan_interval=20, max_coord=env.max_coord)
            hung_g = HungarianAllocator(replan_interval=20, max_coord=env.max_coord)
            loss = run_il_episode(env, policy, hung_l, hung_g)
            if ep % 20 == 0:
                print(f"[{tag}/IL] ep={ep}/{args.episodes} loss={loss:.4f}", flush=True)
        else:
            policy.explore_std = max(0.05, 0.2 - 0.15 * ep / args.episodes)
            hung = HungarianAllocator(replan_interval=20, max_coord=env.max_coord)
            s_wps = run_rl_episode(env, policy, hung, explore=True)
            if ep % 20 == 0:
                print(
                    f"[{tag}/RL] ep={ep}/{args.episodes} S_WPS={s_wps:.1f} std={policy.explore_std:.2f}",
                    flush=True,
                )

        if ep % args.eval_every == 0 or ep == args.episodes:
            mean_s = eval_local_swps(cfg, policy, n=args.eval_eps)
            print(f"  EVAL Local-policy S_WPS={mean_s:.1f}", flush=True)
            if mean_s > best_score:
                best_score = mean_s
                policy.save(out)
                canon = os.path.join(OUT_DIR, f"policy_{tag}_{args.case}.pth")
                policy.save(canon)
                print(f"  Best saved -> {out}", flush=True)

    print(f"Done. checkpoint={out} best_Local_S_WPS={best_score:.1f}", flush=True)
    return out, best_score


if __name__ == "__main__":
    main()
