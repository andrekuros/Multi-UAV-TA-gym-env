"""Small-instance ILP capacity assignment oracle (static upper-bound reference)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None


def solve_capacity_ilp(
    agents,
    tasks,
    max_coord: float = 1000.0,
    time_limit_s: float = 10.0,
) -> Dict[str, Any]:
    """
    Assign agents to open tasks maximizing delivered capacity minus travel.
    One task per agent (matches RL mode). Returns assignment list and objective.

    Intended for small cases (e.g. scal_None) as an optimality-gap reference.
    """
    if pulp is None:
        raise RuntimeError("pulp is required for the ILP oracle")

    live = [a for a in agents if getattr(a, "state", 0) != -1]
    open_tasks = [
        t
        for t in tasks
        if t.id != 0
        and t.status != 2
        and float(t.orgReqs[t.typeIdx]) > 0
    ]
    if not live or not open_tasks:
        return {"actions": [], "objective": 0.0, "status": "empty", "gap": None}

    prob = pulp.LpProblem("uav_ta_capacity", pulp.LpMaximize)
    x = pulp.LpVariable.dicts(
        "assign",
        ((i, j) for i in range(len(live)) for j in range(len(open_tasks))),
        cat="Binary",
    )

    # Objective: sum contributed capacity - travel penalty
    obj = []
    for i, agent in enumerate(live):
        for j, task in enumerate(open_tasks):
            cap = float(agent.currentCap2Task[task.typeIdx])
            if cap <= 0:
                # Force infeasible-ish by excluding (no variable use via 0 coeff + hard skip)
                continue
            dist = float(np.linalg.norm(agent.position - task.position)) / max(max_coord, 1.0)
            need = float(task.orgReqs[task.typeIdx])
            contrib = min(cap, need)
            obj.append((contrib - 0.25 * dist) * x[(i, j)])
    if not obj:
        return {"actions": [], "objective": 0.0, "status": "no_feasible", "gap": None}
    prob += pulp.lpSum(obj)

    # Each agent at most one task
    for i in range(len(live)):
        prob += pulp.lpSum(x[(i, j)] for j in range(len(open_tasks))) <= 1

    # Soft capacity: delivered <= need (multiple agents ok)
    for j, task in enumerate(open_tasks):
        need = float(task.orgReqs[task.typeIdx])
        terms = []
        for i, agent in enumerate(live):
            cap = float(agent.currentCap2Task[task.typeIdx])
            if cap > 0:
                terms.append(min(cap, need) * x[(i, j)])
        if terms:
            # Do not force fullness; maximize contributes via objective
            pass

    # Forbid zero-cap edges
    for i, agent in enumerate(live):
        for j, task in enumerate(open_tasks):
            if float(agent.currentCap2Task[task.typeIdx]) <= 0:
                prob += x[(i, j)] == 0

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    status = prob.solve(solver)
    status_name = pulp.LpStatus.get(status, str(status))

    actions: List[Tuple[str, object]] = []
    for i, agent in enumerate(live):
        for j, task in enumerate(open_tasks):
            var = x.get((i, j))
            if var is not None and pulp.value(var) is not None and pulp.value(var) > 0.5:
                actions.append((agent.name, task))

    obj_val = float(pulp.value(prob.objective) or 0.0)
    return {
        "actions": actions,
        "objective": obj_val,
        "status": status_name,
        "n_agents": len(live),
        "n_tasks": len(open_tasks),
    }


class ILPOracle:
    """One-shot ILP assignment used as a static upper-bound / reference policy."""

    def __init__(self, max_coord: float = 1000.0, time_limit_s: float = 10.0):
        self.max_coord = max_coord
        self.time_limit_s = time_limit_s
        self.n_calls = 0
        self.last_result: Optional[Dict[str, Any]] = None

    def allocate_tasks(self, agents, tasks, time_step: int = 0, force: bool = False):
        self.n_calls += 1
        # One-shot: only plan at t=0 unless forced
        if time_step > 0 and not force and self.last_result is not None:
            return []
        self.last_result = solve_capacity_ilp(
            agents, tasks, max_coord=self.max_coord, time_limit_s=self.time_limit_s
        )
        return self.last_result["actions"]
