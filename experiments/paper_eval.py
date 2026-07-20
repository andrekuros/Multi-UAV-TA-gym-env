"""
Publication evaluation harness: multi-algorithm, multi-metric, static + D1–D3.

Usage:
  python experiments/paper_eval.py --episodes 30 --suite all
  python experiments/paper_eval.py --suite dynamic --policy dqn_Custom/policy_....pth
  python experiments/paper_eval.py --suite static --ilp-oracle
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tianshou.data import Batch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.BehaviourBased.CapabilityGreedy import CapabilityGreedy
from TaskAllocation.BehaviourBased.Greedy import GreedyAgent
from TaskAllocation.BehaviourBased.swarm_gap import SwarmGap
from TaskAllocation.MarketBased.CBBA import CBBA
from TaskAllocation.MarketBased.CBBA_Replan import CBBAReplan
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from TaskAllocation.OptimizationBased.ilp_oracle import ILPOracle, solve_capacity_ilp
from TaskAllocation.RL_Policies.Tianshou_Policy import _get_model
from experiments.paper_scenarios import CASE_SPECS, DEFAULT_ENV_FLAGS, TBTA_E3_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv
from mUAV_TA.MultiDroneEnvUtils import agentEnvOptions

RESULTS_DIR = os.path.join(ROOT, "experiments", "results")


def make_config(spec: Dict[str, Any], env_flags: Dict[str, Any]) -> agentEnvOptions:
    return agentEnvOptions(
        render_speed=-1,
        simulation_frame_rate=0.01,
        max_time_steps=150,
        action_mode="TaskAssign",
        agents=dict(spec["agents"]),
        tasks=dict(spec["tasks"]),
        random_init_pos=False,
        num_obstacles=0,
        multiple_tasks_per_agent=False,
        multiple_agents_per_task=True,
        fail_rate=float(spec.get("fail_rate", 0.0)),
        threats_list=list(spec.get("threats_list") or []),
        fixed_seed=-1,
        early_terminate=bool(env_flags.get("early_terminate", True)),
        capability_mask=bool(env_flags.get("capability_mask", False)),
        saturate_mask=bool(env_flags.get("saturate_mask", False)),
        reward_weights=env_flags.get("reward_weights"),
        arrival_rate=float(spec.get("arrival_rate", 0.0)),
        include_time_windows=bool(env_flags.get("include_time_windows", False)),
        dynamic_idle_penalty=float(env_flags.get("dynamic_idle_penalty", 0.0)),
        sense_radius=float(spec.get("sense_radius", 0.0) or 0.0),
        threat_delay=int(spec.get("threat_delay", 0) or 0),
        hard_windows=bool(spec.get("hard_windows", False)),
        window_length=int(spec.get("window_length", 30) or 30),
        burst_mode=bool(spec.get("burst_mode", False)),
        burst_size=int(spec.get("burst_size", 3) or 3),
        miss_penalty=float(spec.get("miss_penalty", 25.0) or 0.0),
        on_time_bonus=float(spec.get("on_time_bonus", 10.0) or 0.0),
        dual_region_bursts=bool(spec.get("dual_region_bursts", False)),
        share_knowledge=bool(spec.get("share_knowledge", True)),
        commit_horizon=int(spec.get("commit_horizon", 0) or 0),
        reassign_penalty=float(spec.get("reassign_penalty", 0.0) or 0.0),
        escort_enabled=bool(spec.get("escort_enabled", False)),
        escort_radius=float(spec.get("escort_radius", 70.0) or 70.0),
        escort_requirement=float(spec.get("escort_requirement", 1.2) or 1.2),
        escort_intercept_radius=float(spec.get("escort_intercept_radius", 100.0) or 100.0),
        mutual_support_radius=float(spec.get("mutual_support_radius", 80.0) or 80.0),
        escort_agent_types=tuple(spec.get("escort_agent_types", ("F1", "F2")) or ("F1", "F2")),
    )


def _task_residual(task) -> float:
    if getattr(task, "kind", None) == "Escort" or float(getattr(task, "required_agents", 0) or 0) > 0:
        required = float(getattr(task, "required_agents", 1) or 1)
        allocated = len(getattr(task, "allocationDetails", {}) or {})
        return max(required - allocated, 0.0)
    return max(
        float(task.currentReqs[task.typeIdx] - task.allocatedReqs[task.typeIdx]),
        0.0,
    )


def _open_tasks(env):
    return [
        t
        for t in env.tasks
        if t.id != 0 and t.status != 2 and _task_residual(t) > 0
    ]


def _events(info) -> list:
    if isinstance(info, dict):
        return list(info.get("events") or [])
    return []


def run_episode(
    algorithm: str,
    case_id: str,
    seed: int,
    env_flags: Dict[str, Any],
    policy=None,
    replan_interval: int = 20,
    hybrid_agent=None,
) -> Dict[str, float]:
    spec = CASE_SPECS[case_id]
    cfg = make_config(spec, env_flags)
    # Classical TA stacks often allow multi-task planners; RL uses one-task policy
    cfg.multiple_tasks_per_agent = algorithm not in ("TBTA",)
    env = MultiUAVEnv(cfg)
    observation, info = env.reset(seed=seed)
    done = {a: False for a in env.agents}
    truncations = {a: False for a in env.agents}
    rnd = np.random.RandomState(seed)

    gap = cbba = cbba_r = hung = cap_g = greedy = ilp = None
    if algorithm == "Swarm-GAP":
        gap = SwarmGap(env.agents_obj, [], exchange_interval=replan_interval)
    elif algorithm == "CBBA":
        cbba = CBBA(env.agents_obj, env.tasks, env.max_coord, seed=seed)
    elif algorithm == "CBBA-Replan":
        cbba_r = CBBAReplan(env.agents_obj, env.tasks, env.max_coord, seed=seed, replan_interval=replan_interval)
    elif algorithm in ("Hungarian", "RG-DQN", "RA-DQN"):
        hung = HungarianAllocator(replan_interval=replan_interval if algorithm == "Hungarian" else 10**9, max_coord=env.max_coord)
    elif algorithm == "Cap-Greedy":
        cap_g = CapabilityGreedy()
    elif algorithm == "Greedy":
        greedy = GreedyAgent()
    elif algorithm == "ILP-Oracle":
        ilp = ILPOracle(max_coord=env.max_coord)

    if algorithm in ("RG-DQN", "RA-DQN") and cap_g is None:
        cap_g = CapabilityGreedy()

    episode_reward = 0.0
    decision_ms = []
    latest_metrics: Dict[str, Any] = {}
    n_algo_replans = 0
    steps_since_replan = 999

    while not all(done.values()) and not all(truncations.values()):
        actions = {}
        events = _events(info)
        t0 = time.perf_counter()

        if algorithm == "Random":
            un = [t for t in env.tasks if t.status != 2 and t.type != "Hold"]
            if un and env.last_tasks_info:
                agent = env.agent_by_name[env.current_agent]
                if agent.tasks == [env.task_idle] and agent.state != 99:
                    task = un[int(rnd.randint(0, len(un)))]
                    if task in env.last_tasks_info:
                        actions = {agent.name: env.last_tasks_info.index(task)}
        elif algorithm == "Greedy":
            un = _open_tasks(env)
            if un:
                act = greedy.allocate_tasks(env.agents_obj, un)
                if act:
                    actions[act[0][0]] = [env.last_tasks_info.index(act[0][1])]
        elif algorithm == "Cap-Greedy":
            un = _open_tasks(env)
            if un:
                act = cap_g.allocate_tasks(env.get_live_agents(), un)
                if act and act[0][1] in env.last_tasks_info:
                    actions[act[0][0]] = env.last_tasks_info.index(act[0][1])
        elif algorithm == "Swarm-GAP":
            if env.time_steps % gap.exchange_interval == 0:
                un = _open_tasks(env)
                if un:
                    result_actions = gap.process_token(env.agents_obj, un)
                    if result_actions is not None:
                        for action in result_actions:
                            actions[action[0]] = [env.last_tasks_info.index(act) for act in action[1]]
        elif algorithm == "CBBA":
            # One-shot style: allocate early; keep as slow replan every 40 steps for robustness
            if env.time_steps == 0 or env.time_steps % 40 == 0:
                un = _open_tasks(env)
                if un and env.get_live_agents():
                    result_actions = cbba.allocate_tasks(env.get_live_agents(), un)
                    if result_actions:
                        for action in result_actions:
                            actions[action[0]] = [env.last_tasks_info.index(act) for act in action[1]]
        elif algorithm == "CBBA-Replan":
            un = _open_tasks(env)
            if un and env.get_live_agents():
                result_actions = cbba_r.allocate_tasks(
                    env.get_live_agents(), un, time_step=env.time_steps, events=events
                )
                if result_actions:
                    n_algo_replans = cbba_r.n_replans
                    for action in result_actions:
                        actions[action[0]] = [env.last_tasks_info.index(act) for act in action[1]]
        elif algorithm == "Hungarian":
            un = _open_tasks(env)
            if un and env.get_live_agents():
                result = hung.allocate_tasks(
                    env.get_live_agents(), un, time_step=env.time_steps, events=events
                )
                n_algo_replans = hung.n_replans
                for agent_name, task in result:
                    if task in env.last_tasks_info:
                        actions[agent_name] = env.last_tasks_info.index(task)
        elif algorithm == "RG-DQN":
            from TaskAllocation.Hybrid.ReplanGate import build_gate_state

            state = build_gate_state(env, events, steps_since_replan)
            gate_act = 1 if env.time_steps == 0 else hybrid_agent.act(state, explore=False)
            if gate_act == 1 or env.time_steps == 0:
                result = hung.allocate_tasks(
                    env.get_live_agents(), _open_tasks(env), time_step=env.time_steps, events=events, force=True
                )
                n_algo_replans += 1
                steps_since_replan = 0
                for agent_name, task in result:
                    if env.last_tasks_info and task in env.last_tasks_info:
                        actions[agent_name] = env.last_tasks_info.index(task)
            else:
                steps_since_replan += 1
        elif algorithm == "RA-DQN":
            live = env.get_live_agents()
            should = env.time_steps == 0 or steps_since_replan >= 20 or any(
                (ev[0] if isinstance(ev, (list, tuple)) else ev) in ("Reset_Allocation", "New_Threat", "Agent_Fail")
                for ev in events
            )
            if should:
                state = hybrid_agent.build_state(env, events, steps_since_replan, live)
                ra_act = hybrid_agent.act(state, n_live=len(live), explore=False)
                result = hung.allocate_tasks(
                    live, _open_tasks(env), time_step=env.time_steps, events=events, force=True
                )
                n_algo_replans += 1
                for agent_name, task in result:
                    if env.last_tasks_info and task in env.last_tasks_info:
                        actions[agent_name] = env.last_tasks_info.index(task)
                if ra_act > 0 and ra_act <= len(live):
                    act = cap_g.allocate_tasks([live[ra_act - 1]], _open_tasks(env))
                    if act and env.last_tasks_info and act[0][1] in env.last_tasks_info:
                        actions[act[0][0]] = env.last_tasks_info.index(act[0][1])
                steps_since_replan = 0
            else:
                steps_since_replan += 1
        elif algorithm == "ILP-Oracle":
            result = ilp.allocate_tasks(env.get_live_agents(), env.tasks, time_step=env.time_steps)
            for agent_name, task in result:
                if env.last_tasks_info and task in env.last_tasks_info:
                    actions[agent_name] = env.last_tasks_info.index(task)
        elif algorithm == "TBTA":
            agent_id = env.agents_obj[env.agent_selector._current_agent].name
            obs_batch = Batch(obs=[observation[agent_id]], info=[{}])
            if env.agents_obj[env.agent_selector._current_agent].type != "AA":
                action = policy(obs_batch).act
                action = int(np.argmax(np.asarray(action)))
                actions[agent_id] = action

        decision_ms.append((time.perf_counter() - t0) * 1000.0)
        observation, reward, done, truncations, info = env.step(actions)
        episode_reward += sum(reward.values()) / max(env.n_agents, 1)
        if (all(done.values()) or all(truncations.values())) and isinstance(info, dict) and "metrics" in info:
            latest_metrics = info["metrics"]

    out = {
        "F_Reward": float(latest_metrics.get("F_Reward", env.F_Reward)),
        "S_Reward": float(episode_reward),
        "F_time": float(latest_metrics.get("F_time", 0)),
        "F_distance": float(latest_metrics.get("F_distance", 0)),
        "makespan": float(latest_metrics.get("makespan", env.conclusion_time)),
        "total_distance": float(latest_metrics.get("total_distance", env.total_distance)),
        "n_reallocations": float(latest_metrics.get("n_reallocations", env.n_reallocations)),
        "n_arrivals": float(latest_metrics.get("n_arrivals", env.n_arrivals)),
        "Losses": float(latest_metrics.get("Losses", 0)),
        "Kills": float(latest_metrics.get("Kills", 0)),
        "decision_ms_mean": float(np.mean(decision_ms) if decision_ms else 0.0),
        "algo_replans": float(n_algo_replans),
    }
    return out


def evaluate_case(
    case_id: str,
    algorithms: List[str],
    episodes: int,
    env_flags: Dict[str, Any],
    policy_path: Optional[str] = None,
    exp_id: str = "",
    hybrid_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    policy = None
    if "TBTA" in algorithms:
        if not policy_path or not os.path.exists(policy_path):
            algorithms = [a for a in algorithms if a != "TBTA"]
        else:
            cfg = make_config(CASE_SPECS[case_id], env_flags)
            tmp = MultiUAVEnv(cfg)
            tmp.reset(seed=0)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            policy = _get_model(model="CustomNetMultiHead", env=tmp, seed=0, algorithm="DQN")
            policy.load_state_dict(torch.load(policy_path, map_location=device))
            policy.eval()
            policy.set_eps(0.0)

    hybrid_agents = {}
    for name in ("RG-DQN", "RA-DQN"):
        if name not in algorithms:
            continue
        path = hybrid_path
        if not path:
            cand = os.path.join(ROOT, "dqn_Custom", f"policy_{name}_D3_combined.pth")
            path = cand if os.path.exists(cand) else None
        if not path or not os.path.exists(path):
            algorithms = [a for a in algorithms if a != name]
            print(f"No {name} checkpoint; skipping.", flush=True)
            continue
        if name == "RG-DQN":
            from TaskAllocation.Hybrid.ReplanGate import ReplanGateAgent

            ag = ReplanGateAgent()
            ag.load(path)
            ag.eps = 0.0
            hybrid_agents[name] = ag
        else:
            from TaskAllocation.Hybrid.ReplanGate import ResidualAssignmentAgent

            ag = ResidualAssignmentAgent()
            ag.load(path)
            ag.eps = 0.0
            hybrid_agents[name] = ag

    rows = []
    ep_path = os.path.join(RESULTS_DIR, "paper_eval_episodes.csv")
    for algo in algorithms:
        scores = []
        t0 = time.time()
        for ep in range(episodes):
            flags = dict(env_flags)
            if algo not in ("TBTA",):
                flags["capability_mask"] = False
                flags["saturate_mask"] = False
            scores.append(
                run_episode(
                    algo,
                    case_id,
                    ep,
                    flags,
                    policy=policy if algo == "TBTA" else None,
                    hybrid_agent=hybrid_agents.get(algo),
                )
            )
            append_csv(
                ep_path,
                [
                    {
                        "exp": exp_id or "paper",
                        "case": case_id,
                        "algorithm": algo,
                        "episode": ep,
                        "F_Reward": scores[-1]["F_Reward"],
                        "makespan": scores[-1]["makespan"],
                        "total_distance": scores[-1]["total_distance"],
                        "n_reallocations": scores[-1]["n_reallocations"],
                        "decision_ms_mean": scores[-1]["decision_ms_mean"],
                        "algo_replans": scores[-1]["algo_replans"],
                    }
                ],
            )
        elapsed = time.time() - t0
        row = {
            "exp": exp_id or "paper",
            "case": case_id,
            "label": CASE_SPECS[case_id].get("label", case_id),
            "algorithm": algo,
            "episodes": episodes,
            "mean_F_Reward": float(np.mean([s["F_Reward"] for s in scores])),
            "std_F_Reward": float(np.std([s["F_Reward"] for s in scores])),
            "mean_makespan": float(np.mean([s["makespan"] for s in scores])),
            "mean_total_distance": float(np.mean([s["total_distance"] for s in scores])),
            "mean_reallocations": float(np.mean([s["n_reallocations"] for s in scores])),
            "mean_arrivals": float(np.mean([s["n_arrivals"] for s in scores])),
            "mean_Losses": float(np.mean([s["Losses"] for s in scores])),
            "mean_decision_ms": float(np.mean([s["decision_ms_mean"] for s in scores])),
            "mean_algo_replans": float(np.mean([s["algo_replans"] for s in scores])),
            "seconds": round(elapsed, 2),
            "policy": policy_path or hybrid_path or "",
        }
        rows.append(row)
        print(
            f"[{row['exp']}] {case_id} {algo}: F={row['mean_F_Reward']:.1f}+/-{row['std_F_Reward']:.1f} "
            f"T={row['mean_makespan']:.1f} realloc={row['mean_reallocations']:.1f} "
            f"replans={row['mean_algo_replans']:.1f} ({elapsed:.1f}s)",
            flush=True,
        )
    return rows


def append_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)


def run_ilp_gap(case_id: str = "static_strike", seeds: int = 10) -> List[Dict[str, Any]]:
    """Static optimality-gap reference vs ILP objective / Cap-Greedy F_Reward."""
    rows = []
    for seed in range(seeds):
        spec = CASE_SPECS[case_id]
        cfg = make_config(spec, {**DEFAULT_ENV_FLAGS, "capability_mask": False, "saturate_mask": False})
        env = MultiUAVEnv(cfg)
        env.reset(seed=seed)
        ilp = solve_capacity_ilp(env.get_live_agents(), env.tasks, max_coord=env.max_coord)
        # Execute Cap-Greedy and ILP assignment for F_Reward comparison
        for name, actions in [
            ("ILP-Oracle", [(a, t) for a, t in ilp["actions"]]),
        ]:
            env2 = MultiUAVEnv(cfg)
            obs, info = env2.reset(seed=seed)
            # Apply ILP assignment once as indices if possible
            done = {a: False for a in env2.agents}
            trunc = {a: False for a in env2.agents}
            assigned = False
            while not all(done.values()) and not all(trunc.values()):
                actions_dict = {}
                if not assigned and env2.last_tasks_info:
                    for agent_name, task in actions:
                        if task in env2.last_tasks_info:
                            actions_dict[agent_name] = env2.last_tasks_info.index(task)
                    assigned = True
                obs, rew, done, trunc, info = env2.step(actions_dict)
            metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
            rows.append(
                {
                    "case": case_id,
                    "seed": seed,
                    "algorithm": name,
                    "ilp_status": ilp["status"],
                    "ilp_objective": ilp["objective"],
                    "F_Reward": float(metrics.get("F_Reward", env2.F_Reward)),
                    "n_assign": len(actions),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all", choices=["all", "static", "dynamic", "ilp"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--policy", default=None, help="TBTA .pth checkpoint")
    parser.add_argument("--hybrid-policy", default=None, help="RG-DQN / RA-DQN .pth checkpoint")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "paper_eval.csv"))
    parser.add_argument("--exp", default="paper")
    parser.add_argument("--algorithms", default="Random,Greedy,Cap-Greedy,Swarm-GAP,CBBA,CBBA-Replan,Hungarian,TBTA")
    parser.add_argument("--tbta-name", default="TBTA", help="Label for RL rows (e.g. TBTA-D3)")
    parser.add_argument("--env-flags", default="e3", choices=["e3", "default", "d3"], help="Env flag preset")
    parser.add_argument("--ilp-oracle", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    if "TBTA" in algos and not args.policy:
        # Prefer E3 then E4c
        for cand in [
            os.path.join(ROOT, "dqn_Custom", "policy_TBTA_E3_260715-131109.pth"),
            os.path.join(ROOT, "dqn_Custom", "policy_TBTA_E4c_260715-142340.pth"),
        ]:
            if os.path.exists(cand):
                args.policy = cand
                break
        if not args.policy:
            algos = [a for a in algos if a != "TBTA"]
            print("No TBTA policy found; skipping TBTA.", flush=True)

    static_cases = ["static_strike", "recon_strike_mix", "agent_scaling_mid"]
    dynamic_cases = ["D1_attrition", "D2_popup_threats", "D3_combined"]
    if args.suite == "static":
        cases = static_cases
    elif args.suite == "dynamic":
        cases = dynamic_cases
    elif args.suite == "ilp":
        cases = []
    else:
        cases = static_cases + dynamic_cases

    if args.env_flags == "default":
        flags = dict(DEFAULT_ENV_FLAGS)
    elif args.env_flags == "d3":
        flags = {
            **DEFAULT_ENV_FLAGS,
            "include_time_windows": True,
            "dynamic_idle_penalty": 0.05,
            "reward_weights": {
                "action": 0.0,
                "distance": 1.0,
                "quality": 1.0,
                "s_quality": 1.0,
                "time": 0.1,
                "alloc": 0.1,
                "time_penaulty": 0.25,
                "step": 0.0,
            },
        }
    else:
        flags = dict(TBTA_E3_FLAGS)
    all_rows: List[Dict[str, Any]] = []
    for case_id in cases:
        print("=" * 60, flush=True)
        print(f"Case {case_id}: {CASE_SPECS[case_id]['label']}", flush=True)
        rows = evaluate_case(
            case_id, algos, args.episodes, flags, args.policy, args.exp, hybrid_path=args.hybrid_policy
        )
        if args.tbta_name != "TBTA":
            for r in rows:
                if r["algorithm"] == "TBTA":
                    r["algorithm"] = args.tbta_name
        all_rows.extend(rows)
        append_csv(args.out, rows)

    if args.suite in ("all", "ilp") or args.ilp_oracle:
        print("=" * 60, flush=True)
        print("ILP oracle gap (static_strike)", flush=True)
        gap_rows = run_ilp_gap("static_strike", seeds=min(10, args.episodes))
        gap_path = os.path.join(RESULTS_DIR, "ilp_gap.csv")
        append_csv(gap_path, gap_rows)
        print(f"Wrote {len(gap_rows)} ILP rows -> {gap_path}", flush=True)

    summary_path = os.path.join(RESULTS_DIR, "paper_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "policy": args.policy, "episodes": args.episodes}, f, indent=2)
    print(f"Done. CSV={args.out} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
