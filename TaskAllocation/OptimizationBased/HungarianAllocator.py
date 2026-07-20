"""Periodic Hungarian (linear assignment) allocator for residual task demand."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


class HungarianAllocator:
    """
    Every `replan_interval` steps, solve a bipartite min-cost assignment between
    live agents and underfilled tasks (capability-feasible only).
    """

    def __init__(self, replan_interval: int = 20, max_coord: float = 1000.0):
        self.replan_interval = max(1, int(replan_interval))
        self.max_coord = max_coord
        self.last_plan_step = -10**9
        self.n_replans = 0
        self.n_calls = 0

    def should_replan(self, time_step: int, events=None) -> bool:
        if time_step - self.last_plan_step >= self.replan_interval:
            return True
        if events:
            for ev in events:
                tag = ev[0] if isinstance(ev, (list, tuple)) and ev else ev
                if tag in (
                    "Reset_Allocation",
                    "New_Threat",
                    "Agent_Fail",
                    "Escort_Created",
                    "Escort_Retired",
                ):
                    return True
        return False

    def _cost(
        self,
        agent,
        task,
        priority: float = 0.0,
        urgency: float = 0.0,
        missing: Optional[float] = None,
        capability: Optional[float] = None,
    ) -> float:
        cap = (
            float(agent.currentCap2Task[task.typeIdx])
            if capability is None
            else float(capability)
        )
        if cap <= 0:
            return 1e6
        dist = float(np.linalg.norm(agent.position - task.position))
        if missing is None:
            missing = float(
                task.currentReqs[task.typeIdx] - task.allocatedReqs[task.typeIdx]
            )
        missing = max(float(missing), 1e-6)
        return (
            dist / max(self.max_coord, 1.0)
            - 0.5 * min(cap, missing)
            - 0.4 * float(priority)
            - 0.6 * float(urgency)
        )

    def allocate_tasks(
        self,
        agents,
        tasks,
        time_step: int = 0,
        events=None,
        force: bool = False,
        task_priorities=None,
        reserved_agent_names=None,
        agent_known_ids=None,
        edge_scores=None,
    ):
        self.n_calls += 1
        if linear_sum_assignment is None:
            raise RuntimeError("scipy is required for HungarianAllocator")

        if not force and not self.should_replan(time_step, events):
            return []

        reserved = set(reserved_agent_names or [])
        live = [a for a in agents if getattr(a, "state", 0) != -1 and a.name not in reserved]

        def is_escort(task) -> bool:
            return (
                getattr(task, "kind", None) == "Escort"
                or float(getattr(task, "required_agents", 0) or 0) > 0
            )

        def residual_demand(task) -> float:
            if is_escort(task):
                required = float(getattr(task, "required_agents", 1) or 1)
                allocated = len(getattr(task, "allocationDetails", {}) or {})
                return max(required - allocated, 0.0)
            return max(
                float(
                    task.currentReqs[task.typeIdx]
                    - task.allocatedReqs[task.typeIdx]
                ),
                0.0,
            )

        open_tasks = [
            t
            for t in tasks
            if t.id != 0
            and t.status != 2
            and residual_demand(t) > 0
        ]
        if not live or not open_tasks:
            return []

        pri = task_priorities or {}
        scores = edge_scores or {}
        known_map = agent_known_ids  # agent_name -> set of task ids (None = all visible)
        residuals = {id(task): residual_demand(task) for task in open_tasks}
        free_agents = list(live)
        actions: List[Tuple[str, object]] = []

        while free_agents:
            round_tasks = [
                task for task in open_tasks if residuals[id(task)] > 1e-9
            ]
            if not round_tasks:
                break

            cost = np.full(
                (len(free_agents), len(round_tasks)), 1e6, dtype=np.float64
            )
            for i, agent in enumerate(free_agents):
                known = (
                    None
                    if known_map is None
                    else known_map.get(agent.name, set())
                )
                for j, task in enumerate(round_tasks):
                    if known is not None and task.id not in known:
                        continue

                    eligible_types = getattr(task, "eligible_agent_types", None)
                    if eligible_types is not None:
                        if isinstance(eligible_types, str):
                            eligible_types = {eligible_types}
                        if getattr(agent, "type", None) not in eligible_types:
                            continue

                    urgency = 0.0
                    dl = getattr(task, "hard_deadline", None)
                    if dl is not None:
                        remaining = max(dl - time_step, 0)
                        urgency = 1.0 - min(remaining / 40.0, 1.0)

                    delivered = (
                        1.0
                        if is_escort(task)
                        else float(agent.currentCap2Task[task.typeIdx])
                    )
                    base_cost = self._cost(
                        agent,
                        task,
                        priority=float(pri.get(task.id, 0.0)),
                        urgency=urgency,
                        missing=residuals[id(task)],
                        capability=delivered,
                    )
                    if base_cost < 1e5 / 2:
                        cost[i, j] = base_cost - float(
                            scores.get((agent.name, task.id), 0.0)
                        )

            row_ind, col_ind = linear_sum_assignment(cost)
            accepted = []
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e5 / 2:
                    continue
                agent = free_agents[r]
                task = round_tasks[c]
                delivered = (
                    1.0
                    if is_escort(task)
                    else float(agent.currentCap2Task[task.typeIdx])
                )
                actions.append((agent.name, task))
                residuals[id(task)] = max(
                    residuals[id(task)] - delivered, 0.0
                )
                accepted.append(agent)

            if not accepted:
                break
            accepted_ids = {id(agent) for agent in accepted}
            free_agents = [
                agent for agent in free_agents if id(agent) not in accepted_ids
            ]

        self.last_plan_step = time_step
        self.n_replans += 1
        return actions
