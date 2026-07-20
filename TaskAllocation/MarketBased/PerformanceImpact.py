"""
Performance Impact (PI) decentralized task allocation.

Literature-based in-process emulation of Zhao/Whitbrook PI:
  1) Task-inclusion phase — insert by minimum inclusion performance impact (IPI)
  2) Consensus / removal phase — resolve conflicts by removal performance impact (RPI)

Coalition residual demand is auctioned as virtual slots (same as coalition CBBA).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .CBBA import agent_eligible, expand_slot_keys, is_coalition_task

REPLAN_EVENTS = (
    "Reset_Allocation",
    "New_Threat",
    "Agent_Fail",
    "Escort_Created",
    "Escort_Retired",
)


class PerformanceImpact:
    """
    Deterministic PI allocator with coalition residual slots and local visibility.
    Returns (agent_name, [task, ...]) like CBBA for shared consumers.
    """

    def __init__(
        self,
        max_coord: float = 1000.0,
        seed: int = 0,
        replan_interval: int = 12,
        max_iters: int = 40,
    ):
        self.max_coord = float(max_coord)
        self.seed = int(seed)
        self.replan_interval = max(1, int(replan_interval))
        self.max_iters = max(4, int(max_iters))
        self.last_plan_step = -10**9
        self.n_replans = 0
        self.n_calls = 0
        self.task_dict: Dict[int, object] = {}

    def should_replan(self, time_step: int, events=None) -> bool:
        if time_step - self.last_plan_step >= self.replan_interval:
            return True
        if events:
            for ev in events:
                tag = ev[0] if isinstance(ev, (list, tuple)) and ev else ev
                if tag in REPLAN_EVENTS:
                    return True
        return False

    def allocate_tasks(
        self,
        agents,
        tasks,
        time_step: int = 0,
        events=None,
        force: bool = False,
        agent_known_ids=None,
        reserved_agent_names=None,
        max_tasks_per_agent: int = 1,
    ):
        self.n_calls += 1
        if not force and not self.should_replan(time_step, events):
            return []

        reserved = set(reserved_agent_names or [])
        live = [
            a
            for a in agents
            if getattr(a, "state", 0) != -1 and a.name not in reserved
        ]
        if not live or not tasks:
            self.last_plan_step = time_step
            self.n_replans += 1
            return []

        self.task_dict = {t.id: t for t in tasks if t.id != 0}
        slots = expand_slot_keys(tasks)
        if not slots:
            self.last_plan_step = time_step
            self.n_replans += 1
            return []

        slot_task = {k: t for k, t in slots}
        slot_keys = [k for k, _ in slots]
        known_map = agent_known_ids

        paths: Dict[int, List[str]] = {a.id: [] for a in live}
        # slot -> (agent_id, rpi)
        winners: Dict[str, Tuple[Optional[int], float]] = {
            k: (None, -np.inf) for k in slot_keys
        }
        remove_counts: Dict[Tuple[int, str], int] = {}
        assigned_agents: Set[int] = set()

        # ---- Inclusion: repeatedly assign the globally best (agent, slot) by min IPI ----
        for _ in range(len(slot_keys) * max(len(live), 1)):
            best = None  # (ipi, agent_id, slot_key, insert_at)
            for agent in live:
                if agent.id in assigned_agents and max_tasks_per_agent <= 1:
                    continue
                if len(paths[agent.id]) >= max_tasks_per_agent:
                    continue
                known = None if known_map is None else known_map.get(agent.name, set())
                owned_task_ids = {slot_task[k].id for k in paths[agent.id]}
                for slot_key in slot_keys:
                    task = slot_task[slot_key]
                    cur_winner, cur_rpi = winners[slot_key]
                    if cur_winner is not None and cur_winner == agent.id:
                        continue
                    if slot_key in paths[agent.id]:
                        continue
                    if task.id in owned_task_ids:
                        continue
                    if not agent_eligible(agent, task, known):
                        continue
                    ipi, insert_at = self._best_inclusion_impact(
                        agent, paths[agent.id], task, time_step
                    )
                    if not np.isfinite(ipi):
                        continue
                    # Steal only if our keep-value after insert can beat incumbent RPI
                    provisional_rpi = self._provisional_rpi(
                        agent, paths[agent.id], task, insert_at, time_step
                    )
                    if cur_winner is not None:
                        if provisional_rpi < cur_rpi - 1e-9:
                            continue
                        if (
                            abs(provisional_rpi - cur_rpi) <= 1e-9
                            and agent.id >= cur_winner
                        ):
                            continue
                    cand = (ipi, agent.id, slot_key, insert_at)
                    if best is None or cand < best:
                        best = cand

            if best is None:
                break

            _ipi, aid, slot_key, insert_at = best
            agent = next(a for a in live if a.id == aid)
            prev, _ = winners[slot_key]
            if prev is not None and prev != aid:
                if slot_key in paths[prev]:
                    paths[prev].remove(slot_key)
                    remove_counts[(prev, slot_key)] = (
                        remove_counts.get((prev, slot_key), 0) + 1
                    )
                if max_tasks_per_agent <= 1:
                    assigned_agents.discard(prev)
            paths[aid].insert(insert_at, slot_key)
            winners[slot_key] = (
                aid,
                self._removal_impact(agent, paths[aid], slot_key, time_step),
            )
            if max_tasks_per_agent <= 1:
                assigned_agents.add(aid)

        # ---- Consensus cleanup: unique winner per slot by max RPI ----
        for _ in range(self.max_iters):
            changed = False
            claimed: Dict[str, List[Tuple[int, float]]] = {k: [] for k in slot_keys}
            for agent in live:
                for slot_key in list(paths[agent.id]):
                    rpi = self._removal_impact(agent, paths[agent.id], slot_key, time_step)
                    claimed[slot_key].append((agent.id, rpi))

            for slot_key, claimants in claimed.items():
                if len(claimants) <= 1:
                    if claimants:
                        winners[slot_key] = claimants[0]
                    continue
                claimants.sort(key=lambda x: (-x[1], x[0]))
                keep_id = claimants[0][0]
                winners[slot_key] = claimants[0]
                for aid, _rpi in claimants[1:]:
                    if slot_key in paths[aid]:
                        paths[aid].remove(slot_key)
                        remove_counts[(aid, slot_key)] = (
                            remove_counts.get((aid, slot_key), 0) + 1
                        )
                        changed = True

            for agent in live:
                feasible = self._filter_feasible(agent, paths[agent.id], time_step)
                if feasible != paths[agent.id]:
                    dropped = set(paths[agent.id]) - set(feasible)
                    paths[agent.id] = feasible
                    for slot_key in dropped:
                        if winners[slot_key][0] == agent.id:
                            winners[slot_key] = (None, -np.inf)
                    changed = True

            if not changed:
                break

        actions = []
        for agent in live:
            if not paths[agent.id]:
                continue
            seen = set()
            task_list = []
            for slot_key in paths[agent.id]:
                task = slot_task[slot_key]
                if task.id in seen:
                    continue
                seen.add(task.id)
                task_list.append(task)
            if task_list:
                actions.append((agent.name, task_list))

        self.last_plan_step = time_step
        self.n_replans += 1
        return actions

    def _travel_time(self, agent, p0, p1) -> float:
        speed = max(float(getattr(agent, "max_speed", 1.0) or 1.0), 1e-6)
        return float(np.linalg.norm(np.asarray(p0) - np.asarray(p1))) / speed

    def _schedule(self, agent, path_slots: List[str], time_step: int):
        pos = np.asarray(agent.position, dtype=float)
        t = max(float(getattr(agent, "next_free_time", 0) or 0), float(time_step))
        out = []
        for key in path_slots:
            tid = int(str(key).split("#", 1)[0])
            task = self.task_dict[tid]
            travel = self._travel_time(agent, pos, task.position)
            start = t + travel
            dur = float(getattr(task, "task_duration", 0) or 0)
            finish = start + dur
            out.append((key, start, finish))
            pos = np.asarray(task.position, dtype=float)
            t = finish
        return out

    def _path_cost(self, agent, path_slots: List[str], time_step: int) -> float:
        """Lower is better: sum of start times with capability bonus and miss penalty."""
        sched = self._schedule(agent, path_slots, time_step)
        if not sched:
            return 0.0
        cost = 0.0
        for key, start, _finish in sched:
            tid = int(str(key).split("#", 1)[0])
            task = self.task_dict[tid]
            cost += start
            deadline = getattr(task, "hard_deadline", None)
            if deadline is not None and start > float(deadline):
                cost += 200.0 + (start - float(deadline))
            if is_coalition_task(task):
                cost -= 5.0 * max(float(agent.currentCap2Task[task.typeIdx]), 0.5)
            else:
                cost -= 5.0 * float(agent.currentCap2Task[task.typeIdx])
        return cost

    def _best_inclusion_impact(self, agent, path_slots, task, time_step: int):
        """IPI = cost(with) - cost(without). Lower is better."""
        self.task_dict[task.id] = task
        base = self._path_cost(agent, path_slots, time_step)
        best_ipi = np.inf
        best_at = 0
        for i in range(len(path_slots) + 1):
            mapped = path_slots[:i] + [f"{task.id}#ins"] + path_slots[i:]
            sched = self._schedule(agent, mapped, time_step)
            bad = False
            for key, start, _finish in sched:
                tid = int(str(key).split("#", 1)[0])
                t = self.task_dict[tid]
                deadline = getattr(t, "hard_deadline", None)
                if deadline is not None and start > float(deadline) + 1e-6:
                    bad = True
                    break
            if bad:
                continue
            ipi = self._path_cost(agent, mapped, time_step) - base
            if ipi < best_ipi - 1e-9:
                best_ipi = ipi
                best_at = i
        return best_ipi, best_at

    def _provisional_rpi(self, agent, path_slots, task, insert_at, time_step: int) -> float:
        mapped = path_slots[:insert_at] + [f"{task.id}#ins"] + path_slots[insert_at:]
        return self._removal_impact(agent, mapped, f"{task.id}#ins", time_step)

    def _removal_impact(self, agent, path_slots, slot_key: str, time_step: int) -> float:
        """RPI = cost(with) - cost(without). Higher means keep."""
        if slot_key not in path_slots:
            return -np.inf
        with_cost = self._path_cost(agent, path_slots, time_step)
        without = [k for k in path_slots if k != slot_key]
        return with_cost - self._path_cost(agent, without, time_step)

    def _filter_feasible(self, agent, path_slots: List[str], time_step: int) -> List[str]:
        sched = self._schedule(agent, path_slots, time_step)
        keep = []
        for key, start, _finish in sched:
            tid = int(str(key).split("#", 1)[0])
            task = self.task_dict[tid]
            deadline = getattr(task, "hard_deadline", None)
            if deadline is not None and start > float(deadline) + 1e-6:
                break
            keep.append(key)
        return keep
