"""
WPS evaluation: Local vs Global baselines + MLP-RAH + Att-RAH ablations.

Usage:
  python experiments/wps_eval.py --suite WPS_hard --episodes 100
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.BehaviourBased.CapabilityGreedy import CapabilityGreedy
from TaskAllocation.Hybrid.AttentionCommit import AttentionCommit, UrgencyCommit
from TaskAllocation.Hybrid.AttentionRAH import AttentionRAH
from TaskAllocation.Hybrid.ReserveAwareHybrid import ReserveAwareHybrid, build_rah_state
from TaskAllocation.MarketBased.CBBA_Replan import CBBAReplan
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _open_tasks, _events
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv

RESULTS = os.path.join(ROOT, "experiments", "results")


def _apply_assign(env, pairs):
    actions = {}
    for agent_name, task in pairs:
        if env.last_tasks_info and task in env.last_tasks_info:
            actions[agent_name] = env.last_tasks_info.index(task)
    return actions


def _should_replan(env, events, interval=15):
    return (
        env.time_steps == 0
        or env.time_steps % interval == 0
        or any(
            (ev[0] if isinstance(ev, (list, tuple)) else ev)
            in ("Reset_Allocation", "New_Threat", "Agent_Fail")
            for ev in events
        )
    )


def run_wps_episode(
    algorithm: str,
    case_id: str,
    seed: int,
    rah: Optional[ReserveAwareHybrid] = None,
    att_rah: Optional[AttentionRAH] = None,
    att_commit: Optional[AttentionCommit] = None,
    mlp_commit: Optional[AttentionCommit] = None,
    urg_commit: Optional[UrgencyCommit] = None,
) -> Dict[str, float]:
    spec = CASE_SPECS[case_id]
    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(spec, flags)
    cfg.multiple_tasks_per_agent = True
    env = MultiUAVEnv(cfg)
    observation, info = env.reset(seed=seed)
    done = {a: False for a in env.agents}
    trunc = {a: False for a in env.agents}

    hung = HungarianAllocator(replan_interval=20, max_coord=env.max_coord)
    hung_oracle = HungarianAllocator(replan_interval=20, max_coord=env.max_coord)
    cbba_r = CBBAReplan(env.agents_obj, env.tasks, env.max_coord, seed=seed, replan_interval=20)
    cap_g = CapabilityGreedy()
    n_replans = 0
    decision_ms = []
    latest = {}

    while not all(done.values()) and not all(trunc.values()):
        events = _events(info)
        t0 = time.perf_counter()
        actions = {}

        if algorithm == "Global-Hungarian":
            result = hung_oracle.allocate_tasks(
                env.get_live_agents(), _open_tasks(env), time_step=env.time_steps, events=events
            )
            n_replans = hung_oracle.n_replans
            actions = _apply_assign(env, result)
        elif algorithm == "Local-Hungarian":
            open_k = _open_tasks(env)
            result = hung.allocate_tasks(
                env.get_live_agents(),
                open_k,
                time_step=env.time_steps,
                events=events,
                agent_known_ids=env.agent_visibility_map(),
            )
            n_replans = hung.n_replans
            actions = _apply_assign(env, result)
        elif algorithm == "Local-CBBA-Replan":
            known = env.known_tasks_for(None)
            open_k = [
                t
                for t in known
                if t.id != 0 and t.status != 2 and t.allocatedReqs[t.typeIdx] < t.currentReqs[t.typeIdx]
            ]
            vis = env.agent_visibility_map() or {}
            result = cbba_r.allocate_tasks(
                env.get_live_agents(), open_k, time_step=env.time_steps, events=events
            )
            if result:
                n_replans = cbba_r.n_replans
                for action in result:
                    agent_name = action[0]
                    known_ids = vis.get(agent_name) if vis else None
                    idxs = []
                    for act in action[1]:
                        if act not in env.last_tasks_info:
                            continue
                        if known_ids is not None and act.id not in known_ids:
                            continue
                        idxs.append(env.last_tasks_info.index(act))
                    if idxs:
                        actions[agent_name] = idxs
        elif algorithm == "Local-Cap-Greedy":
            open_k = _open_tasks(env)
            vis = env.agent_visibility_map()
            act = cap_g.allocate_tasks(env.get_live_agents(), open_k)
            if act and env.last_tasks_info and act[0][1] in env.last_tasks_info:
                agent_name, task = act[0][0], act[0][1]
                if vis is None or task.id in vis.get(agent_name, set()):
                    actions[agent_name] = env.last_tasks_info.index(task)
        elif algorithm in ("RAH", "MLP-RAH"):
            if _should_replan(env, events) and rah is not None:
                result, _, _, _ = rah.plan(env, hung, events=events, force=True)
                n_replans = rah.n_replans
                actions = _apply_assign(env, result)
        elif algorithm == "RAH-no-reserve":
            if _should_replan(env, events) and rah is not None:
                state = build_rah_state(env, events, 0)
                _, pri_vec = rah.act(state, explore=False)
                open_known = _open_tasks(env)
                vis = env.agent_visibility_map()
                task_pri = {}
                for i, t in enumerate(open_known[: rah.max_tasks]):
                    urgency = 0.0
                    dl = getattr(t, "hard_deadline", None)
                    if dl is not None:
                        remaining = max(dl - env.time_steps, 0)
                        urgency = 1.0 - min(remaining / 40.0, 1.0)
                    scarcity = 0.0
                    if vis is not None:
                        n_know = sum(1 for s in vis.values() if t.id in s)
                        scarcity = 1.0 - min(n_know / max(len(vis), 1), 1.0)
                    task_pri[t.id] = 0.4 * urgency + 0.35 * float(pri_vec[i]) + 0.25 * scarcity
                result = hung.allocate_tasks(
                    env.get_live_agents(),
                    open_known,
                    time_step=env.time_steps,
                    events=events,
                    force=True,
                    task_priorities=task_pri,
                    reserved_agent_names=[],
                    agent_known_ids=vis,
                )
                n_replans += 1 if result else 0
                actions = _apply_assign(env, result)
        elif algorithm == "Att-RAH":
            if _should_replan(env, events) and att_rah is not None:
                result, _, _, _ = att_rah.plan(env, hung, events=events, force=True)
                n_replans = att_rah.n_replans
                actions = _apply_assign(env, result)
        elif algorithm == "Att-RAH-no-reserve":
            if _should_replan(env, events) and att_rah is not None:
                result, _, _, _ = att_rah.plan(
                    env, hung, events=events, force=True, force_no_reserve=True
                )
                n_replans = att_rah.n_replans
                actions = _apply_assign(env, result)
        elif algorithm == "Att-RAH-no-priority":
            if _should_replan(env, events) and att_rah is not None:
                result, _, _, _ = att_rah.plan(
                    env, hung, events=events, force=True, force_no_learned_pri=True
                )
                n_replans = att_rah.n_replans
                actions = _apply_assign(env, result)
        elif algorithm == "Att-Commit":
            if _should_replan(env, events) and att_commit is not None:
                result, _, _, _ = att_commit.plan(env, hung, events=events, force=True)
                n_replans = att_commit.n_replans
                actions = _apply_assign(env, result)
        elif algorithm == "MLP-Commit":
            if _should_replan(env, events) and mlp_commit is not None:
                result, _, _, _ = mlp_commit.plan(env, hung, events=events, force=True)
                n_replans = mlp_commit.n_replans
                actions = _apply_assign(env, result)
        elif algorithm == "Urgency-Commit":
            if urg_commit is None:
                urg_commit = UrgencyCommit()
            if _should_replan(env, events):
                result, _, _, _ = urg_commit.plan(env, hung, events=events, force=True)
                n_replans = urg_commit.n_replans
                actions = _apply_assign(env, result)

        decision_ms.append((time.perf_counter() - t0) * 1000.0)
        observation, reward, done, trunc, info = env.step(actions)
        if (all(done.values()) or all(trunc.values())) and isinstance(info, dict) and "metrics" in info:
            latest = info["metrics"]

    return {
        "F_Reward": float(latest.get("F_Reward", env.F_Reward)),
        "S_WPS": float(latest.get("S_WPS", env.compute_s_wps())),
        "on_time_rate": float(latest.get("on_time_rate", 0.0)),
        "n_missed_windows": float(latest.get("n_missed_windows", env.n_missed_windows)),
        "n_on_time": float(latest.get("n_on_time", env.n_on_time)),
        "n_windowed_tasks": float(latest.get("n_windowed_tasks", env.n_windowed_tasks)),
        "reserve_idle_fraction": float(latest.get("reserve_idle_fraction", 0.0)),
        "makespan": float(latest.get("makespan", env.conclusion_time)),
        "total_distance": float(latest.get("total_distance", env.total_distance)),
        "decision_ms_mean": float(np.mean(decision_ms) if decision_ms else 0.0),
        "algo_replans": float(n_replans),
        "n_task_switches": float(latest.get("n_task_switches", getattr(env, "n_task_switches", 0))),
        "max_coord": float(getattr(env, "max_coord", 1000.0)),
    }


def _bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    """Paired bootstrap 95% CI for mean(a-b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(0)
    means = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        means.append(float(d[idx].mean()))
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(d.mean()), lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="WPS",
        choices=["WPS", "WPS_hard", "WPS_attn", "WPS_commit", "all_wps"],
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--rah", default=os.path.join(ROOT, "dqn_Custom", "policy_RAH_WPS_hard.pth"))
    parser.add_argument("--att-rah", default=os.path.join(ROOT, "dqn_Custom", "policy_AttRAH_WPS_hard.pth"))
    parser.add_argument(
        "--att-commit", default=os.path.join(ROOT, "dqn_Custom", "policy_AttCommit_WPS_commit.pth")
    )
    parser.add_argument(
        "--mlp-commit", default=os.path.join(ROOT, "dqn_Custom", "policy_MLPCommit_WPS_commit.pth")
    )
    parser.add_argument("--out", default=os.path.join(RESULTS, "wps_eval.csv"))
    parser.add_argument(
        "--episodes-out",
        default=None,
        help="Optional path for per-episode CSV (algorithm,seed,components).",
    )
    parser.add_argument("--exp", default="wps30")
    parser.add_argument(
        "--algorithms",
        default=(
            "Local-Cap-Greedy,Local-Hungarian,Local-CBBA-Replan,Global-Hungarian,"
            "MLP-RAH,Att-RAH,Att-RAH-no-reserve,Att-RAH-no-priority"
        ),
    )
    args = parser.parse_args()

    if args.suite == "WPS_hard":
        cases = ["WPS_hard"]
    elif args.suite == "WPS_attn":
        cases = ["WPS_attn"]
    elif args.suite == "WPS_commit":
        cases = ["WPS_commit"]
    elif args.suite == "all_wps":
        cases = ["WPS_easy", "WPS_hard", "WPS_burst", "WPS_attn", "WPS_commit"]
    else:
        cases = ["WPS_easy", "WPS_hard"]

    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    rah = None
    att_rah = None
    att_commit = None
    mlp_commit = None
    urg_commit = UrgencyCommit()
    mlp_names = {"RAH", "MLP-RAH", "RAH-no-reserve"}
    att_names = {"Att-RAH", "Att-RAH-no-reserve", "Att-RAH-no-priority"}
    if any(a in mlp_names for a in algos):
        if os.path.exists(args.rah):
            rah = ReserveAwareHybrid()
            rah.load(args.rah)
            rah.eps = 0.0
        else:
            print(f"No MLP-RAH checkpoint at {args.rah}; skipping MLP-RAH variants.", flush=True)
            algos = [a for a in algos if a not in mlp_names]
    if any(a in att_names for a in algos):
        if os.path.exists(args.att_rah):
            att_rah = AttentionRAH()
            att_rah.load(args.att_rah)
            att_rah.eps = 0.0
        else:
            print(f"No Att-RAH checkpoint at {args.att_rah}; skipping Att-RAH variants.", flush=True)
            algos = [a for a in algos if a not in att_names]
    if "Att-Commit" in algos:
        if os.path.exists(args.att_commit):
            att_commit = AttentionCommit(use_attention=True)
            att_commit.load(args.att_commit)
            att_commit.eps = 0.0
        else:
            print(f"No Att-Commit checkpoint at {args.att_commit}; skipping.", flush=True)
            algos = [a for a in algos if a != "Att-Commit"]
    if "MLP-Commit" in algos:
        if os.path.exists(args.mlp_commit):
            mlp_commit = AttentionCommit(use_attention=False)
            mlp_commit.load(args.mlp_commit)
            mlp_commit.eps = 0.0
        else:
            print(f"No MLP-Commit checkpoint at {args.mlp_commit}; skipping.", flush=True)
            algos = [a for a in algos if a != "MLP-Commit"]

    os.makedirs(RESULTS, exist_ok=True)
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    rows = []
    per_ep = {}  # (case, algo) -> list of episode dicts

    for case in cases:
        print("=" * 60, case, flush=True)
        for algo in algos:
            scores = []
            t0 = time.time()
            for ep in range(args.episodes):
                scores.append(
                    run_wps_episode(
                        algo,
                        case,
                        ep,
                        rah=rah,
                        att_rah=att_rah,
                        att_commit=att_commit,
                        mlp_commit=mlp_commit,
                        urg_commit=urg_commit,
                    )
                )
            elapsed = time.time() - t0
            per_ep[(case, algo)] = scores
            row = {
                "exp": args.exp,
                "case": case,
                "label": CASE_SPECS[case]["label"],
                "algorithm": algo,
                "episodes": args.episodes,
                "mean_S_WPS": float(np.mean([s["S_WPS"] for s in scores])),
                "std_S_WPS": float(np.std([s["S_WPS"] for s in scores])),
                "mean_on_time_rate": float(np.mean([s["on_time_rate"] for s in scores])),
                "std_on_time_rate": float(np.std([s["on_time_rate"] for s in scores])),
                "mean_missed_windows": float(np.mean([s["n_missed_windows"] for s in scores])),
                "mean_on_time": float(np.mean([s["n_on_time"] for s in scores])),
                "mean_F_Reward": float(np.mean([s["F_Reward"] for s in scores])),
                "std_F_Reward": float(np.std([s["F_Reward"] for s in scores])),
                "mean_total_distance": float(np.mean([s["total_distance"] for s in scores])),
                "mean_makespan": float(np.mean([s["makespan"] for s in scores])),
                "mean_reserve_idle": float(np.mean([s["reserve_idle_fraction"] for s in scores])),
                "mean_decision_ms": float(np.mean([s["decision_ms_mean"] for s in scores])),
                "mean_algo_replans": float(np.mean([s["algo_replans"] for s in scores])),
                "seconds": round(elapsed, 2),
            }
            # Paired bootstrap vs Local-Hungarian when available
            local_key = (case, "Local-Hungarian")
            if local_key in per_ep and algo != "Local-Hungarian":
                local_s = np.array([x["S_WPS"] for x in per_ep[local_key]])
                algo_s = np.array([x["S_WPS"] for x in scores])
                local_ot = np.array([x["on_time_rate"] for x in per_ep[local_key]])
                algo_ot = np.array([x["on_time_rate"] for x in scores])
                d_s, lo_s, hi_s = _bootstrap_ci_diff(algo_s, local_s)
                d_ot, lo_ot, hi_ot = _bootstrap_ci_diff(algo_ot, local_ot)
                row["delta_S_WPS_vs_LocalH"] = d_s
                row["delta_S_WPS_ci_lo"] = lo_s
                row["delta_S_WPS_ci_hi"] = hi_s
                row["delta_on_time_vs_LocalH"] = d_ot
                row["delta_on_time_ci_lo"] = lo_ot
                row["delta_on_time_ci_hi"] = hi_ot
            else:
                row["delta_S_WPS_vs_LocalH"] = 0.0
                row["delta_S_WPS_ci_lo"] = 0.0
                row["delta_S_WPS_ci_hi"] = 0.0
                row["delta_on_time_vs_LocalH"] = 0.0
                row["delta_on_time_ci_lo"] = 0.0
                row["delta_on_time_ci_hi"] = 0.0

            rows.append(row)
            print(
                f"[{args.exp}] {case} {algo}: S_WPS={row['mean_S_WPS']:.1f}+/-{row['std_S_WPS']:.1f} "
                f"on_time={row['mean_on_time_rate']:.2f} miss={row['mean_missed_windows']:.1f} "
                f"F={row['mean_F_Reward']:.1f} ({elapsed:.1f}s)",
                flush=True,
            )

    # Second pass: recompute bootstrap for algos that ran before Local-Hungarian in the loop
    # (if Local was first, already fine). Recompute all deltas now that all per_ep filled.
    for row in rows:
        case, algo = row["case"], row["algorithm"]
        local_key = (case, "Local-Hungarian")
        if local_key not in per_ep or algo == "Local-Hungarian":
            continue
        local_s = np.array([x["S_WPS"] for x in per_ep[local_key]])
        algo_s = np.array([x["S_WPS"] for x in per_ep[(case, algo)]])
        local_ot = np.array([x["on_time_rate"] for x in per_ep[local_key]])
        algo_ot = np.array([x["on_time_rate"] for x in per_ep[(case, algo)]])
        d_s, lo_s, hi_s = _bootstrap_ci_diff(algo_s, local_s)
        d_ot, lo_ot, hi_ot = _bootstrap_ci_diff(algo_ot, local_ot)
        row["delta_S_WPS_vs_LocalH"] = d_s
        row["delta_S_WPS_ci_lo"] = lo_s
        row["delta_S_WPS_ci_hi"] = hi_s
        row["delta_on_time_vs_LocalH"] = d_ot
        row["delta_on_time_ci_lo"] = lo_ot
        row["delta_on_time_ci_hi"] = hi_ot

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if args.episodes_out:
        ep_rows = []
        for (case, algo), scores in per_ep.items():
            for seed, s in enumerate(scores):
                ep_rows.append(
                    {
                        "exp": args.exp,
                        "case": case,
                        "algorithm": algo,
                        "seed": seed,
                        "S_WPS": s["S_WPS"],
                        "n_on_time": s["n_on_time"],
                        "n_missed_windows": s["n_missed_windows"],
                        "total_distance": s["total_distance"],
                        "max_coord": s.get("max_coord", 1000.0),
                        "on_time_rate": s["on_time_rate"],
                    }
                )
        with open(args.episodes_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()))
            w.writeheader()
            w.writerows(ep_rows)
        print(f"Episodes -> {args.episodes_out}", flush=True)
    summary = os.path.join(RESULTS, "wps_final_eval_summary.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"Done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
