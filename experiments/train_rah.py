"""
Train Reserve-Aware Hybrid on WPS_hard.

Usage:
  python experiments/train_rah.py --episodes 300 --case WPS_hard
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.Hybrid.ReserveAwareHybrid import ReserveAwareHybrid, build_rah_state
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _events
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv

OUT_DIR = os.path.join(ROOT, "dqn_Custom")


def run_episode(env, rah: ReserveAwareHybrid, hung: HungarianAllocator, explore: bool):
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
        state = build_rah_state(env, events, 0)
        rho, pri = 0.0, np.zeros(rah.max_tasks, dtype=np.float32)
        if should:
            if explore:
                rho, pri = rah.act(state, explore=True)
            result, rho_p, task_pri, state = rah.plan(env, hung, events=events, force=True)
            if not explore:
                rho = rho_p
            for agent_name, task in result:
                if env.last_tasks_info and task in env.last_tasks_info:
                    actions[agent_name] = env.last_tasks_info.index(task)

        observation, reward, done, trunc, info = env.step(actions)
        s_now = float(env.compute_s_wps())
        step_r = (s_now - s_prev) / 20.0
        s_prev = s_now
        next_state = build_rah_state(env, _events(info), 0)
        ep_done = all(done.values()) or all(trunc.values())
        if should:
            rah.push(state, rho, pri, step_r, next_state, ep_done)
            rah.update(batch_size=64)
    final = info.get("metrics", {}) if isinstance(info, dict) else {}
    return (
        float(final.get("S_WPS", env.compute_s_wps())),
        float(final.get("on_time_rate", 0.0)),
        int(final.get("n_missed_windows", 0)),
        float(final.get("F_Reward", env.F_Reward)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="WPS_hard")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-eps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out = args.out or os.path.join(OUT_DIR, f"policy_RAH_{args.case}.pth")
    os.makedirs(OUT_DIR, exist_ok=True)

    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(CASE_SPECS[args.case], flags)
    cfg.multiple_tasks_per_agent = True

    rah = ReserveAwareHybrid()
    best_score = -1e9

    for ep in range(1, args.episodes + 1):
        rah.eps = max(0.05, 0.45 - 0.4 * ep / args.episodes)
        env = MultiUAVEnv(cfg)
        hung = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
        s_wps, ot, miss, f_legacy = run_episode(env, rah, hung, explore=True)
        if ep % 20 == 0:
            print(
                f"[RAH] ep={ep}/{args.episodes} S_WPS={s_wps:.1f} on_time={ot:.2f} miss={miss} F={f_legacy:.1f} eps={rah.eps:.2f}",
                flush=True,
            )
        if ep % args.eval_every == 0 or ep == args.episodes:
            rah.eps = 0.0
            evals_s, evals_ot = [], []
            for i in range(args.eval_eps):
                ev = MultiUAVEnv(cfg)
                hung_e = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
                ss, oot, _, _ = run_episode(ev, rah, hung_e, explore=False)
                evals_s.append(ss)
                evals_ot.append(oot)
            mean_s = float(np.mean(evals_s))
            mean_ot = float(np.mean(evals_ot))
            score = mean_s + 100.0 * mean_ot
            print(f"  EVAL S_WPS={mean_s:.1f} on_time={mean_ot:.2f} score={score:.1f}", flush=True)
            if score > best_score:
                best_score = score
                rah.save(out)
                print(f"  Best saved -> {out}", flush=True)
            rah.eps = max(0.05, 0.45 - 0.4 * ep / args.episodes)

    print(f"Done. checkpoint={out} best_score={best_score:.1f}", flush=True)


if __name__ == "__main__":
    main()
