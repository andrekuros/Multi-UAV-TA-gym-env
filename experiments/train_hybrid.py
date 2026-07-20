"""
Train Replan-Gate DQN and Residual Assignment DQN on dynamic suites.

Usage:
  python experiments/train_hybrid.py --algo RG-DQN --episodes 400 --case D3_combined
  python experiments/train_hybrid.py --algo RA-DQN --episodes 400 --case D3_combined
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.BehaviourBased.CapabilityGreedy import CapabilityGreedy
from TaskAllocation.Hybrid.ReplanGate import (
    GateTransition,
    ReplanGateAgent,
    ResidualAssignmentAgent,
    build_gate_state,
)
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _open_tasks, _events
from experiments.paper_scenarios import CASE_SPECS, TBTA_E3_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv

RESULTS = os.path.join(ROOT, "dqn_Custom")
os.makedirs(RESULTS, exist_ok=True)


def _apply_hungarian(env, hung: HungarianAllocator, force: bool = False, events=None):
    un = _open_tasks(env)
    live = env.get_live_agents()
    if not un or not live:
        return {}, 0
    result = hung.allocate_tasks(live, un, time_step=env.time_steps, events=events, force=force)
    actions = {}
    for agent_name, task in result:
        if env.last_tasks_info and task in env.last_tasks_info:
            actions[agent_name] = env.last_tasks_info.index(task)
    return actions, 1 if result else 0


def _apply_cap_greedy_one(env, agent, cap_g: CapabilityGreedy):
    un = _open_tasks(env)
    if not un or not env.last_tasks_info:
        return {}
    # Temporarily restrict to this agent
    act = cap_g.allocate_tasks([agent], un)
    if act and act[0][1] in env.last_tasks_info:
        return {act[0][0]: env.last_tasks_info.index(act[0][1])}
    return {}


def run_rg_episode(env, agent: ReplanGateAgent, hung: HungarianAllocator, explore: bool, replan_cost: float = 0.05):
    observation, info = env.reset()
    done = {a: False for a in env.agents}
    trunc = {a: False for a in env.agents}
    steps_since = 999
    f_prev = 0.0
    n_replans = 0
    while not all(done.values()) and not all(trunc.values()):
        events = _events(info)
        state = build_gate_state(env, events, steps_since)
        gate_act = agent.act(state, explore=explore)
        actions = {}
        replan_this = False
        if gate_act == 1 or env.time_steps == 0:
            actions, did = _apply_hungarian(env, hung, force=True, events=events)
            if did or gate_act == 1:
                replan_this = True
                n_replans += 1
                steps_since = 0
                agent.n_replans += 1
        else:
            steps_since += 1

        observation, reward, done, trunc, info = env.step(actions)
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
        f_now = float(metrics.get("F_Reward", getattr(env, "F_Reward", 0.0)))
        # Dense shaping: quality of progress minus replan cost
        step_r = (f_now - f_prev) / 100.0 - (replan_cost if replan_this and env.time_steps > 0 else 0.0)
        # Bonus for reacting to reset events
        if any(
            (ev[0] if isinstance(ev, (list, tuple)) else ev) in ("Reset_Allocation", "New_Threat", "Agent_Fail")
            for ev in events
        ):
            if replan_this:
                step_r += 0.1
            else:
                step_r -= 0.05
        f_prev = f_now
        next_state = build_gate_state(env, _events(info), steps_since)
        ep_done = all(done.values()) or all(trunc.values())
        agent.push(GateTransition(state, gate_act, step_r, next_state, ep_done))
        agent.update(batch_size=64)
    final = info.get("metrics", {}) if isinstance(info, dict) else {}
    return float(final.get("F_Reward", env.F_Reward)), n_replans


def run_ra_episode(env, agent: ResidualAssignmentAgent, hung: HungarianAllocator, cap_g: CapabilityGreedy, explore: bool):
    observation, info = env.reset()
    done = {a: False for a in env.agents}
    trunc = {a: False for a in env.agents}
    steps_since = 999
    f_prev = 0.0
    n_overrides = 0
    while not all(done.values()) and not all(trunc.values()):
        events = _events(info)
        live = env.get_live_agents()
        state = agent.build_state(env, events, steps_since, live)
        # Periodic / event-driven decision
        should_decide = env.time_steps == 0 or steps_since >= 20 or any(
            (ev[0] if isinstance(ev, (list, tuple)) else ev) in ("Reset_Allocation", "New_Threat", "Agent_Fail")
            for ev in events
        )
        actions = {}
        ra_act = 0
        if should_decide:
            ra_act = agent.act(state, n_live=len(live), explore=explore)
            prop, _ = _apply_hungarian(env, hung, force=True, events=events)
            actions.update(prop)
            if ra_act > 0 and ra_act <= len(live):
                ag = live[ra_act - 1]
                override = _apply_cap_greedy_one(env, ag, cap_g)
                actions.update(override)
                n_overrides += 1
                agent.n_overrides += 1
            steps_since = 0
        else:
            steps_since += 1

        observation, reward, done, trunc, info = env.step(actions)
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
        f_now = float(metrics.get("F_Reward", getattr(env, "F_Reward", 0.0)))
        step_r = (f_now - f_prev) / 100.0 - (0.02 if ra_act > 0 else 0.0)
        f_prev = f_now
        next_state = agent.build_state(env, _events(info), steps_since, env.get_live_agents())
        ep_done = all(done.values()) or all(trunc.values())
        if should_decide:
            agent.push(GateTransition(state, ra_act, step_r, next_state, ep_done))
            agent.update(batch_size=64)
    final = info.get("metrics", {}) if isinstance(info, dict) else {}
    return float(final.get("F_Reward", env.F_Reward)), n_overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["RG-DQN", "RA-DQN"], default="RG-DQN")
    parser.add_argument("--case", default="D3_combined")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-eps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    flags = dict(TBTA_E3_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(CASE_SPECS[args.case], flags)
    cfg.multiple_tasks_per_agent = True

    out = args.out or os.path.join(RESULTS, f"policy_{args.algo}_{args.case}.pth")
    best_f = -1e9
    best_path = out

    if args.algo == "RG-DQN":
        agent = ReplanGateAgent()
        hung = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)  # only force-replans
        history = []
        for ep in range(1, args.episodes + 1):
            agent.eps = max(0.05, 0.4 - 0.35 * ep / args.episodes)
            env = MultiUAVEnv(cfg)
            f, n_r = run_rg_episode(env, agent, hung, explore=True)
            history.append(f)
            if ep % 20 == 0:
                print(f"[{args.algo}] ep={ep}/{args.episodes} F={f:.1f} replans={n_r} eps={agent.eps:.2f} buf={len(agent.buffer)}", flush=True)
            if ep % args.eval_every == 0 or ep == args.episodes:
                evals = []
                agent.eps = 0.0
                for i in range(args.eval_eps):
                    e = MultiUAVEnv(cfg)
                    # Fix seed via reset
                    e.reset(seed=i)
                    # Re-run with explore=False — rebuild hung
                    hung_e = HungarianAllocator(replan_interval=10**9, max_coord=e.max_coord)
                    # Manual seeded episode
                    ev = MultiUAVEnv(cfg)
                    ff, _ = run_rg_episode(ev, agent, hung_e, explore=False)
                    evals.append(ff)
                mean_f = float(np.mean(evals))
                print(f"  EVAL mean_F={mean_f:.1f} +/- {float(np.std(evals)):.1f}", flush=True)
                if mean_f > best_f:
                    best_f = mean_f
                    agent.save(out)
                    print(f"  Best saved F={best_f:.1f} -> {out}", flush=True)
                agent.eps = max(0.05, 0.4 - 0.35 * ep / args.episodes)
        agent.save(out)
    else:
        agent = ResidualAssignmentAgent()
        cap_g = CapabilityGreedy()
        for ep in range(1, args.episodes + 1):
            agent.eps = max(0.05, 0.4 - 0.35 * ep / args.episodes)
            env = MultiUAVEnv(cfg)
            hung = HungarianAllocator(replan_interval=10**9, max_coord=env.max_coord)
            f, n_o = run_ra_episode(env, agent, hung, cap_g, explore=True)
            if ep % 20 == 0:
                print(f"[{args.algo}] ep={ep}/{args.episodes} F={f:.1f} overrides={n_o} eps={agent.eps:.2f}", flush=True)
            if ep % args.eval_every == 0 or ep == args.episodes:
                evals = []
                agent.eps = 0.0
                for i in range(args.eval_eps):
                    ev = MultiUAVEnv(cfg)
                    hung_e = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
                    ff, _ = run_ra_episode(ev, agent, hung_e, cap_g, explore=False)
                    evals.append(ff)
                mean_f = float(np.mean(evals))
                print(f"  EVAL mean_F={mean_f:.1f} +/- {float(np.std(evals)):.1f}", flush=True)
                if mean_f > best_f:
                    best_f = mean_f
                    agent.save(out)
                    print(f"  Best saved F={best_f:.1f} -> {out}", flush=True)
                agent.eps = max(0.05, 0.4 - 0.35 * ep / args.episodes)
        agent.save(out)

    print(f"Done. checkpoint={out} best_F={best_f:.1f}", flush=True)


if __name__ == "__main__":
    main()
