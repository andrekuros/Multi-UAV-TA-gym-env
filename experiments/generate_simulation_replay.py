"""Generate one deterministic, reviewable WPS_commit replay for the web UI."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TaskAllocation.Hybrid.AttentionCommit import UrgencyCommit
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import _events, make_config
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv


def _should_replan(env, events, interval: int = 15) -> bool:
    return (
        env.time_steps == 0
        or env.time_steps % interval == 0
        or any(
            (event[0] if isinstance(event, (list, tuple)) and event else event)
            in (
                "Reset_Allocation",
                "New_Threat",
                "Agent_Fail",
                "Escort_Created",
                "Escort_Retired",
            )
            for event in events
        )
    )


def _actions(env, pairs) -> dict:
    actions = {}
    for agent_name, task in pairs:
        if env.last_tasks_info and task in env.last_tasks_info:
            actions[agent_name] = env.last_tasks_info.index(task)
    return actions


def _event_record(event, time_step: int) -> dict:
    if isinstance(event, (list, tuple)):
        kind = str(event[0]) if event else "Unknown"
        detail = [str(value) for value in event[1:]]
    else:
        kind, detail = str(event), []
    return {"time": time_step, "type": kind, "detail": detail}


def _task_key(task: dict) -> tuple:
    # Attack/detect tasks can share numeric ids; type+id is unique in the log.
    return (task["type"], int(task["id"]))


def _infer_events(previous: dict, current: dict) -> list[dict]:
    """Create reviewer-friendly events for state changes the env does not emit."""
    time_step = current["time"]
    inferred = []
    prev_agents = {agent["name"]: agent for agent in previous["agents"]}
    prev_tasks = {_task_key(task): task for task in previous["tasks"]}
    prev_threats = {threat["id"] for threat in previous["threats"]}

    for agent in current["agents"]:
        old = prev_agents.get(agent["name"])
        if old and old["state"] != -1 and agent["state"] == -1:
            inferred.append(
                {"time": time_step, "type": "Agent_Fail", "detail": [agent["name"]]}
            )

    for task in current["tasks"]:
        key = _task_key(task)
        old = prev_tasks.get(key)
        label = f"{task['type']}{task['id']}"
        if old is None:
            region = "left" if task["position"][0] < 600 else "right"
            inferred.append(
                {
                    "time": time_step,
                    "type": "Task_Arrival",
                    "detail": [label, region],
                }
            )
        elif old["status"] != 2 and task["status"] == 2:
            missed = task["deadline"] is not None and time_step > task["deadline"]
            inferred.append(
                {
                    "time": time_step,
                    "type": "Window_Missed" if missed else "Task_Completed",
                    "detail": [label],
                }
            )
        if old and old["known_by"] == 0 and task["known_by"] > 0:
            inferred.append(
                {
                    "time": time_step,
                    "type": "Task_Discovered",
                    "detail": [label, f"by {task['known_by']} UAV(s)"],
                }
            )

    for threat in current["threats"]:
        if threat["id"] not in prev_threats:
            inferred.append(
                {"time": time_step, "type": "Threat_Spawn", "detail": [str(threat["id"])]}
            )

    for name in current["decision"]["new_commits"]:
        inferred.append({"time": time_step, "type": "Agent_Commit", "detail": [name]})
    if current["decision"]["replanned"]:
        inferred.append({"time": time_step, "type": "Replan", "detail": []})
    return inferred


def _frame(env, events: list, replanned: bool, committed: list[str]) -> dict:
    visibility = env.agent_visibility_map()
    visibility = visibility or {
        agent.name: {task.id for task in env.tasks if task.id != 0}
        for agent in env.get_live_agents()
    }
    known_count = {}
    for known in visibility.values():
        for task_id in known:
            known_count[task_id] = known_count.get(task_id, 0) + 1

    agents = []
    for agent in env.agents_obj:
        task = agent.tasks[0] if agent.tasks else env.task_idle
        agents.append(
            {
                "id": int(agent.id),
                "name": agent.name,
                "type": agent.type,
                "position": [float(agent.position[0]), float(agent.position[1])],
                "state": int(agent.state),
                "task_id": int(task.id),
                "commit_until": int(getattr(agent, "commit_until", 0) or 0),
                "known_tasks": len(visibility.get(agent.name, set())),
            }
        )

    tasks = []
    for task in env.tasks:
        if task.id == 0:
            continue
        required = float(task.currentReqs[task.typeIdx])
        allocated = float(task.allocatedReqs[task.typeIdx])
        deadline = getattr(task, "hard_deadline", None)
        kind = getattr(task, "kind", None)
        prot = getattr(task, "protected_agent", None)
        n_assigned = len(getattr(task, "allocationDetails", {}) or {})
        tasks.append(
            {
                "id": int(task.id),
                "type": task.type,
                "kind": kind,
                "position": [float(task.position[0]), float(task.position[1])],
                "status": int(task.status),
                "created_at": int(getattr(task, "created_at", 0) or 0),
                "deadline": None if deadline is None else int(deadline),
                "required": required,
                "allocated": allocated,
                "known_by": int(known_count.get(task.id, 0)),
                "is_dynamic": deadline is not None,
                "is_escort": kind == "Escort",
                "required_agents": int(getattr(task, "required_agents", 0) or 0),
                "assigned_agents": int(n_assigned),
                "protected_agent": None if prot is None else str(prot.name),
                "protected_position": None
                if prot is None
                else [float(prot.position[0]), float(prot.position[1])],
            }
        )

    threats = [
        {
            "id": int(threat.id),
            "position": [float(threat.position[0]), float(threat.position[1])],
            "status": int(threat.status),
            "group": int(threat.threat_group),
            "threat_type": getattr(threat, "threat_type", None),
            "mission_target": None
            if getattr(threat, "mission_target_agent", None) is None
            else str(threat.mission_target_agent.name),
            "intercepting": None
            if getattr(threat, "intercepting_agent", None) is None
            else str(threat.intercepting_agent.name),
        }
        for threat in env.threats
    ]

    escort_cov = float(
        getattr(env, "escort_covered_steps", 0)
        / max(getattr(env, "escort_required_steps", 0), 1)
    )
    return {
        "time": int(env.time_steps),
        "agents": agents,
        "tasks": tasks,
        "threats": threats,
        "events": [_event_record(event, env.time_steps) for event in events],
        "decision": {"replanned": replanned, "new_commits": committed},
        "metrics": {
            "s_wps": float(env.compute_s_wps()),
            "s_esc": float(env.compute_s_esc()) if hasattr(env, "compute_s_esc") else float(env.compute_s_wps()),
            "on_time": int(env.n_on_time),
            "missed": int(env.n_missed_windows),
            "switches": int(env.n_task_switches),
            "distance": float(env.total_distance),
            "active_agents": sum(1 for agent in env.agents_obj if agent.state != -1),
            "open_tasks": sum(1 for task in env.tasks if task.id != 0 and task.status != 2),
            "escort_coverage": escort_cov,
            "recon_losses": int(getattr(env, "recon_losses", 0)),
            "protected_rec": int(getattr(env, "protected_rec_completed", 0)),
            "mutual_support": int(getattr(env, "mutual_support_engagements", 0)),
        },
    }


def generate(seed: int, output: Path, scenario: str = "WPS_commit") -> dict:
    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    config = make_config(CASE_SPECS[scenario], flags)
    config.multiple_tasks_per_agent = True

    env = MultiUAVEnv(config)
    _, info = env.reset(seed=seed)

    if scenario == "WPS_escort":
        from TaskAllocation.Hybrid.AttentionEscort import UrgencyCoalition

        planner = UrgencyCoalition()
        algorithm = "Urgency-Coalition + Coalition-Hungarian"
        title = "WPS_escort: protect recon with fighter coalitions"
    else:
        planner = UrgencyCommit()
        algorithm = "Urgency-Commit + Local-Hungarian"
        title = "WPS_commit: dual-front dynamic mission"

    hungarian = HungarianAllocator(replan_interval=10**9, max_coord=env.max_coord)
    done = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    frames = [_frame(env, [], False, [])]
    event_log = []
    while not all(done.values()) and not all(truncated.values()):
        previous_events = _events(info)
        actions = {}
        replanned = False
        committed = []
        if _should_replan(env, previous_events):
            if scenario == "WPS_escort":
                pairs = planner.plan(env, hungarian, events=previous_events, force=True)
                committed = []
            else:
                pairs, _, committed, _ = planner.plan(
                    env, hungarian, events=previous_events, force=True
                )
            actions = _actions(env, pairs)
            replanned = True

        _, _, done, truncated, info = env.step(actions)
        events = _events(info)
        current = _frame(env, events, replanned, committed)
        records = [_event_record(event, env.time_steps) for event in events]
        inferred = _infer_events(frames[-1], current)
        current["events"].extend(inferred)
        records.extend(inferred)
        event_log.extend(records)
        frames.append(current)

    replay = {
        "metadata": {
            "title": title,
            "scenario": scenario,
            "algorithm": algorithm,
            "seed": seed,
            "max_time_steps": int(config.max_time_steps),
            "area": [float(env.area_width), float(env.area_height)],
            "dynamics": {
                "arrival_rate": float(config.arrival_rate),
                "fail_rate": float(config.fail_rate),
                "sense_radius": float(config.sense_radius),
                "threat_delay": int(config.threat_delay),
                "hard_windows": bool(config.hard_windows),
                "window_length": int(config.window_length),
                "burst_mode": bool(config.burst_mode),
                "burst_size": int(config.burst_size),
                "dual_region_bursts": bool(config.dual_region_bursts),
                "share_knowledge": bool(config.share_knowledge),
                "commit_horizon": int(config.commit_horizon),
                "reassign_penalty": float(config.reassign_penalty),
                "escort_enabled": bool(getattr(config, "escort_enabled", False)),
                "escort_radius": float(getattr(config, "escort_radius", 0.0) or 0.0),
            },
        },
        "events": event_log,
        "frames": frames,
        "final_metrics": frames[-1]["metrics"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    return replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="WPS_commit", choices=list(CASE_SPECS.keys()))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    out = args.out or (
        ROOT
        / "experiments"
        / "results"
        / (f"{args.scenario.lower()}_replay.json")
    )
    replay = generate(args.seed, out, scenario=args.scenario)
    event_types = sorted({event["type"] for event in replay["events"]})
    print(
        f"Wrote {out} ({len(replay['frames'])} frames, "
        f"{len(replay['events'])} events: {', '.join(event_types) or 'none'})"
    )


if __name__ == "__main__":
    main()
