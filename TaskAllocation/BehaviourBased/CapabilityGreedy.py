"""Capability-aware nearest greedy for heterogeneous UAV task allocation."""
from __future__ import annotations

import numpy as np


class CapabilityGreedy:
    """Prefer tasks the agent can contribute to; break ties by distance."""

    def __init__(self, min_cap: float = 1e-6):
        self.min_cap = min_cap
        self.n_calls = 0

    def allocate_tasks(self, agents, tasks):
        """Return list of (agent_name, task) for one best (agent, task) pair."""
        self.n_calls += 1
        best = None
        best_score = float("-inf")

        live = [a for a in agents if getattr(a, "state", 0) != -1]
        open_tasks = [
            t
            for t in tasks
            if t.id != 0
            and t.status != 2
            and t.allocatedReqs[t.typeIdx] < t.currentReqs[t.typeIdx]
        ]

        for agent in live:
            for task in open_tasks:
                cap = float(agent.currentCap2Task[task.typeIdx])
                if cap <= self.min_cap:
                    continue
                missing = max(
                    float(task.currentReqs[task.typeIdx] - task.allocatedReqs[task.typeIdx]),
                    0.0,
                )
                if missing <= 0:
                    continue
                dist = float(np.linalg.norm(agent.position - task.position))
                # Higher is better: capability contribution minus travel penalty
                score = min(cap, missing) * 10.0 - dist / 1000.0
                if score > best_score:
                    best_score = score
                    best = (agent.name, task)

        return [best] if best is not None else []
