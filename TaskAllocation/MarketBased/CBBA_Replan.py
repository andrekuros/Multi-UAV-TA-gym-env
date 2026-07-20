"""CBBA that replans periodically or when dynamic events force reallocation."""
from __future__ import annotations

from .CBBA import CBBA

REPLAN_EVENTS = (
    "Reset_Allocation",
    "New_Threat",
    "Agent_Fail",
    "Escort_Created",
    "Escort_Retired",
)


class CBBAReplan:
    """
    Consensus-Based Bundle Algorithm with periodic / event-triggered replan.

    Fair dynamic competitor: re-solves open tasks on a schedule instead of
    allocating once at the beginning of the episode.
    """

    def __init__(self, agents, tasks, max_coord, seed: int = 0, replan_interval: int = 20):
        self.max_coord = max_coord
        self.seed = seed
        self.replan_interval = max(1, int(replan_interval))
        self._cbba = CBBA(agents, tasks, max_coord, seed=seed)
        self.last_plan_step = -10**9
        self.n_replans = 0
        self.n_calls = 0

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
        self.last_plan_step = time_step
        self.n_replans += 1
        # Fresh CBBA instance keeps bids coherent with current open task set
        self._cbba = CBBA(agents, tasks, self.max_coord, seed=self.seed + self.n_replans)
        return self._cbba.allocate_tasks(
            agents,
            tasks,
            agent_known_ids=agent_known_ids,
            reserved_agent_names=reserved_agent_names,
            time_step=time_step,
            max_tasks_per_agent=max_tasks_per_agent,
        )
