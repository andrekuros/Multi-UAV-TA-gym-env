"""Consensus-Based Bundle Algorithm with residual coalition slots."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def is_coalition_task(task) -> bool:
    return (
        getattr(task, "kind", None) == "Escort"
        or float(getattr(task, "required_agents", 0) or 0) > 0
    )


def residual_demand(task) -> float:
    if is_coalition_task(task):
        required = float(getattr(task, "required_agents", 1) or 1)
        allocated = len(getattr(task, "allocationDetails", {}) or {})
        return max(required - allocated, 0.0)
    return max(
        float(task.currentReqs[task.typeIdx] - task.allocatedReqs[task.typeIdx]),
        0.0,
    )


def agent_eligible(agent, task, known_ids=None) -> bool:
    if getattr(agent, "state", 0) == -1:
        return False
    if known_ids is not None and task.id not in known_ids:
        return False
    eligible = getattr(task, "eligible_agent_types", None)
    if eligible is not None:
        if isinstance(eligible, str):
            eligible = {eligible}
        if getattr(agent, "type", None) not in eligible:
            return False
    # Incumbent already counting toward residual demand
    if agent.id in (getattr(task, "allocationDetails", {}) or {}):
        return False
    if is_coalition_task(task):
        return True
    return float(agent.currentCap2Task[task.typeIdx]) > 0


def expand_slot_keys(tasks) -> List[Tuple[str, object]]:
    """Expand residual demand into distinct virtual auction keys."""
    slots: List[Tuple[str, object]] = []
    for task in tasks:
        if task.id == 0 or task.status == 2:
            continue
        rem = residual_demand(task)
        if rem <= 0:
            continue
        if is_coalition_task(task):
            n_slots = int(np.ceil(rem))
            for k in range(n_slots):
                slots.append((f"{task.id}#c{k}", task))
        else:
            # One auction slot per residual capability unit (capped).
            n_slots = max(1, int(np.ceil(min(rem, 4.0))))
            for k in range(n_slots):
                slots.append((f"{task.id}#r{k}", task))
    return slots


class CBBA:
    def __init__(self, drones, tasks, max_dist, seed=0):
        self.max_dist = max_dist
        self.seed = seed
        self.rndGen = random.Random(seed)

    def allocate_tasks(
        self,
        agents,
        tasks,
        Qs=None,
        agent_known_ids=None,
        reserved_agent_names=None,
        time_step: int = 0,
        max_tasks_per_agent: int = 1,
    ):
        """
        Returns list of (agent_name, [task, ...]) for backward compatibility.

        Coalition tasks auction residual headcount slots so multiple fighters
        can win the same escort/T1 task.
        """
        reserved = set(reserved_agent_names or [])
        live = [
            a
            for a in agents
            if getattr(a, "state", 0) != -1 and a.name not in reserved
        ]
        if not live or not tasks:
            return []

        self.task_dict = {task.id: task for task in tasks if task.id != 0}
        slots = expand_slot_keys(tasks)
        if not slots:
            return []

        slot_keys = [k for k, _ in slots]
        slot_task = {k: t for k, t in slots}
        remaining: Set[str] = set(slot_keys)
        agents_dict = {agent.id: agent for agent in live}
        bundles: Dict[int, List[str]] = {agent.id: [] for agent in live}
        paths: Dict[int, List[str]] = {agent.id: [] for agent in live}
        self.current_makespan = 0
        self.bids = {k: {"agent_id": None, "bid": -np.inf} for k in slot_keys}
        # Track real task ownership so one agent cannot win two slots of same task
        owned_tasks: Dict[int, Set[int]] = {agent.id: set() for agent in live}

        known_map = agent_known_ids
        max_iters = max(8, len(slot_keys) * 2)
        for _ in range(max_iters):
            if not remaining:
                break
            allocation_changed = False
            ordered_keys = list(remaining)
            self.rndGen.shuffle(ordered_keys)
            agent_order = list(live)
            self.rndGen.shuffle(agent_order)

            for slot_key in ordered_keys:
                task = slot_task[slot_key]
                for agent in agent_order:
                    known = (
                        None
                        if known_map is None
                        else known_map.get(agent.name, set())
                    )
                    if not agent_eligible(agent, task, known):
                        continue
                    if task.id in owned_tasks[agent.id]:
                        continue
                    if len(bundles[agent.id]) >= max_tasks_per_agent and slot_key not in bundles[agent.id]:
                        continue
                    if slot_key in bundles[agent.id]:
                        continue

                    bid = self.calculate_bid(agent, task, paths[agent.id], Qs=Qs)
                    if bid <= self.bids[slot_key]["bid"]:
                        continue

                    allocation_changed = True
                    prev_id = self.bids[slot_key]["agent_id"]
                    if prev_id is not None:
                        if slot_key in paths[prev_id]:
                            paths[prev_id].remove(slot_key)
                        if slot_key in bundles[prev_id]:
                            bundles[prev_id].remove(slot_key)
                        owned_tasks[prev_id].discard(task.id)

                    self.bids[slot_key] = {"agent_id": agent.id, "bid": bid}
                    insertion = self.determine_insertion_point(agent, task, paths[agent.id])
                    paths[agent.id].insert(insertion, slot_key)

            if not allocation_changed:
                break

            # Consensus: commit winning bids into bundles
            to_remove = []
            for slot_key, bid_info in self.bids.items():
                winner = bid_info["agent_id"]
                if winner is None:
                    continue
                task = slot_task[slot_key]
                for agent in live:
                    if slot_key in bundles[agent.id] and agent.id != winner:
                        bundles[agent.id].remove(slot_key)
                        if slot_key in paths[agent.id]:
                            paths[agent.id].remove(slot_key)
                        owned_tasks[agent.id].discard(task.id)
                if slot_key not in bundles[winner]:
                    if task.id in owned_tasks[winner]:
                        # Another slot of same task already owned — drop this bid
                        self.bids[slot_key] = {"agent_id": None, "bid": -np.inf}
                        if slot_key in paths[winner]:
                            paths[winner].remove(slot_key)
                        continue
                    if len(bundles[winner]) >= max_tasks_per_agent:
                        self.bids[slot_key] = {"agent_id": None, "bid": -np.inf}
                        if slot_key in paths[winner]:
                            paths[winner].remove(slot_key)
                        continue
                    bundles[winner].append(slot_key)
                    owned_tasks[winner].add(task.id)
                    to_remove.append(slot_key)

            for slot_key in to_remove:
                remaining.discard(slot_key)

            self.current_makespan = max(
                (self.calculate_total_time(agent, paths[agent.id]) for agent in live),
                default=0.0,
            )

        actions = []
        for agent_id, bundle in bundles.items():
            if not bundle:
                continue
            # Deduplicate real tasks while preserving order
            seen = set()
            task_list = []
            for slot_key in bundle:
                task = slot_task[slot_key]
                if task.id in seen:
                    continue
                seen.add(task.id)
                task_list.append(task)
            if task_list:
                actions.append((agents_dict[agent_id].name, task_list))
        return actions

    def calculate_bid(self, agent, task, path, Qs=None):
        if Qs is None:
            max_score = -np.inf
            for i in range(len(path) + 1):
                new_path = path[:i] + [f"tmp:{task.id}"] + path[i:]
                score = self._score_mixed_path(agent, new_path, task)
                if score > max_score:
                    max_score = score
            return max_score - self._score_mixed_path(agent, path, None)

        return Qs[agent.name][task.id]

    def determine_insertion_point(self, agent, task, path):
        max_score = -np.inf
        insertion_point = 0
        for i in range(len(path) + 1):
            new_path = path[:i] + [f"tmp:{task.id}"] + path[i:]
            score = self._score_mixed_path(agent, new_path, task)
            if score > max_score:
                max_score = score
                insertion_point = i
        return insertion_point

    def _resolve_path_tasks(self, path, tmp_task=None):
        resolved = []
        for key in path:
            if isinstance(key, str) and key.startswith("tmp:"):
                if tmp_task is not None:
                    resolved.append(tmp_task)
                continue
            # slot key -> task via id prefix
            if isinstance(key, str) and "#" in key:
                tid = int(key.split("#", 1)[0])
                task = self.task_dict.get(tid)
            elif isinstance(key, int):
                task = self.task_dict.get(key)
            else:
                task = self.task_dict.get(key)
            if task is not None:
                resolved.append(task)
        return resolved

    def _score_mixed_path(self, agent, path, tmp_task):
        score = 0.0
        temp_position = np.asarray(agent.position, dtype=float)
        temp_time = float(getattr(agent, "next_free_time", 0) or 0)
        for task in self._resolve_path_tasks(path, tmp_task):
            score += self.calculate_task_score(agent, task, temp_position, temp_time)
            dist = float(np.linalg.norm(temp_position - task.position))
            speed = max(float(getattr(agent, "max_speed", 1.0) or 1.0), 1e-6)
            temp_position = np.asarray(task.position, dtype=float)
            temp_time += dist / speed + float(getattr(task, "task_duration", 0) or 0)
        return score

    def calculate_score(self, agent, path):
        # Legacy path-of-task-ids API
        score = 0.0
        temp_position = np.asarray(agent.position, dtype=float)
        temp_time = float(getattr(agent, "next_free_time", 0) or 0)
        for task_id in path:
            task = self.task_dict[task_id]
            score += self.calculate_task_score(agent, task, temp_position, temp_time)
            dist = float(np.linalg.norm(temp_position - task.position))
            speed = max(float(getattr(agent, "max_speed", 1.0) or 1.0), 1e-6)
            temp_position = np.asarray(task.position, dtype=float)
            temp_time += dist / speed
        return score

    def calculate_task_score(self, agent, task, temp_position, temp_time):
        dist = float(np.linalg.norm(temp_position - task.position))
        quality = float(agent.currentCap2Task[task.typeIdx])
        if is_coalition_task(task):
            quality = max(quality, 1.0)
        speed = max(float(getattr(agent, "max_speed", 1.0) or 1.0), 1e-6)
        time = float(temp_time) + dist / speed

        deadline = getattr(task, "hard_deadline", None)
        if deadline is not None and time > float(deadline):
            return -50.0

        if time < self.current_makespan:
            return (
                -2.5 * dist / max(self.max_dist, 1.0)
                + 160.0 * quality
                + 2.0 * (self.current_makespan - time)
            )
        return (
            -2.5 * dist / max(self.max_dist, 1.0)
            + 160.0 * quality
            - 2.0 * (time - self.current_makespan)
        )

    def calculate_total_time(self, agent, path):
        temp_position = np.asarray(agent.position, dtype=float)
        temp_time = float(getattr(agent, "next_free_time", 0) or 0)
        speed = max(float(getattr(agent, "max_speed", 1.0) or 1.0), 1e-6)
        for key in path:
            if isinstance(key, str) and "#" in key:
                tid = int(key.split("#", 1)[0])
                task = self.task_dict.get(tid)
            else:
                task = self.task_dict.get(key)
            if task is None:
                continue
            dist = float(np.linalg.norm(temp_position - task.position))
            temp_position = np.asarray(task.position, dtype=float)
            temp_time += dist / speed + float(getattr(task, "task_duration", 0) or 0)
        return temp_time
