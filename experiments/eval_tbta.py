"""
Evaluate TBTA checkpoint (+ optional baselines) on UCF-style cases.
"""
from __future__ import annotations

import argparse
import csv
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

from TaskAllocation.BehaviourBased.Greedy import GreedyAgent
from TaskAllocation.BehaviourBased.swarm_gap import SwarmGap
from TaskAllocation.MarketBased.CBBA import CBBA
from TaskAllocation.RL_Policies.Tianshou_Policy import _get_model
from mUAV_TA.DroneEnv import MultiUAVEnv
from mUAV_TA.MultiDroneEnvUtils import agentEnvOptions

CASES = {
    "scal_None": {"F1": 0, "F2": 2, "R1": 0, "R2": 0, "Att": 15, "Rec": 0, "Hold": 0},
    "scal_Agents_mid": {"F1": 3, "F2": 0, "R1": 6, "R2": 0, "Att": 6, "Rec": 24, "Hold": 0},
    "scal_Tasks_mid": {"F1": 2, "F2": 2, "R1": 3, "R2": 3, "Att": 4, "Rec": 10, "Hold": 0},
    "train_mixed": {"F1": 2, "F2": 0, "R1": 4, "R2": 0, "Att": 6, "Rec": 12, "Hold": 0},
}


def make_config(case: Dict[str, int], env_flags: Dict[str, Any]) -> agentEnvOptions:
    return agentEnvOptions(
        render_speed=-1,
        simulation_frame_rate=0.01,
        max_time_steps=150,
        action_mode="TaskAssign",
        agents={"F1": case.get("F1", 0), "F2": case.get("F2", 0), "R1": case.get("R1", 0), "R2": case.get("R2", 0)},
        tasks={"Att": case.get("Att", 0), "Rec": case.get("Rec", 0), "Hold": case.get("Hold", 0)},
        random_init_pos=False,
        num_obstacles=0,
        multiple_tasks_per_agent=False,
        multiple_agents_per_task=True,
        fail_rate=0.0,
        threats_list=[],
        fixed_seed=-1,
        early_terminate=bool(env_flags.get("early_terminate", False)),
        capability_mask=bool(env_flags.get("capability_mask", False)),
        saturate_mask=bool(env_flags.get("saturate_mask", False)),
        reward_weights=env_flags.get("reward_weights"),
    )


def _open_tasks(env):
    return [
        t
        for t in env.tasks
        if t.id != 0
        and t.allocatedReqs[t.typeIdx] < t.currentReqs[t.typeIdx]
        and t.status != 2
    ]


def run_episode(algorithm: str, case: Dict[str, int], seed: int, env_flags: Dict[str, Any], policy=None) -> Dict[str, float]:
    cfg = make_config(case, env_flags)
    cfg.multiple_tasks_per_agent = algorithm != "TBTA"
    cfg.multiple_agents_per_task = True
    env = MultiUAVEnv(cfg)
    observation, info = env.reset(seed=seed)
    done = {a: False for a in env.agents}
    truncations = {a: False for a in env.agents}
    rnd = np.random.RandomState(seed)

    gap = cbba = greedy = None
    if algorithm == "Swarm-GAP":
        gap = SwarmGap(env.agents_obj, [], exchange_interval=20)
    elif algorithm == "CBBA":
        cbba = CBBA(env.agents_obj, env.tasks, env.max_coord, seed=seed)
    elif algorithm == "Greedy":
        greedy = GreedyAgent()

    episode_reward = 0.0
    latest_metrics = {"F_Reward": 0.0, "F_time": 0.0, "F_distance": 0.0}

    while not all(done.values()) and not all(truncations.values()):
        actions = {}
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
        elif algorithm == "Swarm-GAP":
            if env.time_steps % gap.exchange_interval == 0:
                un = _open_tasks(env)
                if un:
                    result_actions = gap.process_token(env.agents_obj, un)
                    if result_actions is not None:
                        for action in result_actions:
                            actions[action[0]] = [env.last_tasks_info.index(act) for act in action[1]]
        elif algorithm == "CBBA":
            un = _open_tasks(env)
            if un and len(env.get_live_agents()) > 0:
                result_actions = cbba.allocate_tasks(env.get_live_agents(), un)
                if result_actions is not None:
                    for action in result_actions:
                        actions[action[0]] = [env.last_tasks_info.index(act) for act in action[1]]
        elif algorithm == "TBTA":
            agent_id = env.agents_obj[env.agent_selector._current_agent].name
            obs_batch = Batch(obs=[observation[agent_id]], info=[{}])
            if env.agents_obj[env.agent_selector._current_agent].type != "AA":
                action = policy(obs_batch).act
                action = np.argmax(np.asarray(action))
                actions[agent_id] = action

        observation, reward, done, truncations, info = env.step(actions)
        episode_reward += sum(reward.values()) / max(env.n_agents, 1)
        if (all(done.values()) or all(truncations.values())) and isinstance(info, dict) and "metrics" in info:
            latest_metrics = info["metrics"]

    return {
        "F_Reward": float(latest_metrics.get("F_Reward", env.F_Reward)),
        "S_Reward": float(episode_reward),
        "F_time": float(latest_metrics.get("F_time", 0)),
        "F_distance": float(latest_metrics.get("F_distance", 0)),
    }


def evaluate(
    case_name: str,
    episodes: int,
    policy_path: Optional[str],
    env_flags: Dict[str, Any],
    algorithms: List[str],
    exp_id: str = "",
) -> List[Dict[str, Any]]:
    case = CASES[case_name]
    rows = []
    policy = None
    if "TBTA" in algorithms:
        if not policy_path or not os.path.exists(policy_path):
            raise FileNotFoundError(f"Missing policy: {policy_path}")
        cfg = make_config(case, env_flags)
        cfg.multiple_tasks_per_agent = False
        tmp = MultiUAVEnv(cfg)
        tmp.reset(seed=0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = _get_model(model="CustomNetMultiHead", env=tmp, seed=0, algorithm="DQN")
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        policy.eval()
        policy.set_eps(0.0)

    for algo in algorithms:
        scores = []
        t0 = time.time()
        for ep in range(episodes):
            flags = dict(env_flags)
            if algo != "TBTA":
                flags["capability_mask"] = False
                flags["saturate_mask"] = False
                flags["reward_weights"] = None
            scores.append(run_episode(algo, case, ep, flags, policy=policy if algo == "TBTA" else None))
        elapsed = time.time() - t0
        row = {
            "exp": exp_id,
            "case": case_name,
            "algorithm": algo,
            "episodes": episodes,
            "mean_F_Reward": float(np.mean([s["F_Reward"] for s in scores])),
            "std_F_Reward": float(np.std([s["F_Reward"] for s in scores])),
            "mean_S_Reward": float(np.mean([s["S_Reward"] for s in scores])),
            "mean_F_time": float(np.mean([s["F_time"] for s in scores])),
            "seconds": round(elapsed, 2),
            "policy": policy_path or "",
        }
        rows.append(row)
        print(
            f"[{exp_id}] {case_name} {algo}: F_Reward={row['mean_F_Reward']:.2f} +/- {row['std_F_Reward']:.2f} ({elapsed:.1f}s)",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="train_mixed", choices=list(CASES.keys()))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--exp", default="")
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "eval_log.csv"))
    parser.add_argument("--early-terminate", action="store_true")
    parser.add_argument("--capability-mask", action="store_true")
    parser.add_argument("--saturate-mask", action="store_true")
    parser.add_argument("--algorithms", default="Random,Greedy,Swarm-GAP,CBBA,TBTA")
    args = parser.parse_args()

    env_flags = {
        "early_terminate": args.early_terminate,
        "capability_mask": args.capability_mask,
        "saturate_mask": args.saturate_mask,
        "reward_weights": None,
    }
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    if "TBTA" in algos and not args.policy:
        raise SystemExit("TBTA requires --policy")

    rows = evaluate(args.case, args.episodes, args.policy, env_flags, algos, args.exp)
    append_csv(args.out, rows)
    print(f"Wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
