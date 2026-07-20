"""
Replan-Gate DQN (RG-DQN): RL decides Hold vs Hungarian replan; classical solver assigns.

Action space: 0 = Hold, 1 = Hungarian replan.
State: compact global dynamics features (events, unmet demand, time, idle fraction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_gate_state(env, events=None, steps_since_replan: int = 0) -> np.ndarray:
    """Compact global feature vector for the replan gate."""
    events = events or []
    fail = 0.0
    threat = 0.0
    reset = 0.0
    for ev in events:
        tag = ev[0] if isinstance(ev, (list, tuple)) and ev else ev
        if tag == "Agent_Fail":
            fail = 1.0
        elif tag == "New_Threat":
            threat = 1.0
        elif tag == "Reset_Allocation":
            reset = 1.0

    live = [a for a in env.agents_obj if getattr(a, "state", 0) != -1]
    open_tasks = [
        t
        for t in env.tasks
        if t.id != 0 and t.status != 2 and t.allocatedReqs[t.typeIdx] < t.currentReqs[t.typeIdx]
    ]
    unmet = []
    for t in open_tasks:
        need = float(t.currentReqs[t.typeIdx] - t.allocatedReqs[t.typeIdx])
        unmet.append(max(need, 0.0))
    mean_unmet = float(np.mean(unmet)) if unmet else 0.0
    idle = sum(1 for a in live if len(a.tasks) == 0 or (a.tasks and a.tasks[0].id == 0))
    n_live = max(len(live), 1)
    max_t = max(getattr(env, "max_time_steps", 150), 1)
    max_tasks = max(getattr(env, "max_tasks", 31), 1)

    return np.asarray(
        [
            fail,
            threat,
            reset,
            len(open_tasks) / max_tasks,
            min(mean_unmet / 10.0, 1.0),
            env.time_steps / max_t,
            steps_since_replan / max_t,
            idle / n_live,
            len(live) / max(getattr(env, "n_agents", n_live), 1),
            float(getattr(env, "n_arrivals", 0)) / max_tasks,
        ],
        dtype=np.float32,
    )


class GateNet(nn.Module):
    def __init__(self, state_dim: int = 10, n_actions: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class GateTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplanGateAgent:
    """Epsilon-greedy DQN over Hold / Hungarian-replan."""

    def __init__(
        self,
        state_dim: int = 10,
        n_actions: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.net = GateNet(state_dim, n_actions).to(self.device)
        self.target = GateNet(state_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.eps = 0.2
        self.buffer: List[GateTransition] = []
        self.max_buffer = 50_000
        self.n_updates = 0
        self.n_replans = 0

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.rand() < self.eps:
            return int(np.random.randint(0, self.n_actions))
        with torch.no_grad():
            q = self.net(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def push(self, tr: GateTransition):
        self.buffer.append(tr)
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer :]

    def update(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device)
        r = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        d = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(ns).max(dim=1)[0]
            target = r + self.gamma * (1.0 - d) * q_next
        loss = F.mse_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.n_updates += 1
        if self.n_updates % 50 == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "n_actions": self.n_actions}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.net.load_state_dict(data["net"])
        self.target.load_state_dict(data["net"])
        self.net.eval()


class ResidualAssignmentAgent:
    """
    Residual Assignment DQN: Hungarian proposes edges; RL accepts or redirects one idle/live agent.

    Actions: 0 = accept Hungarian proposal; 1..K = pick agent index to reassign to Cap-Greedy best task.
    """

    def __init__(
        self,
        max_agents: int = 12,
        state_dim: int = 10,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: Optional[str] = None,
    ):
        self.max_agents = max_agents
        self.n_actions = 1 + max_agents  # accept + override agent i
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.net = GateNet(state_dim + max_agents, self.n_actions).to(self.device)
        self.target = GateNet(state_dim + max_agents, self.n_actions).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.eps = 0.2
        self.buffer: List[GateTransition] = []
        self.max_buffer = 50_000
        self.n_updates = 0
        self.n_overrides = 0

    def build_state(self, env, events, steps_since_replan, live_agents) -> np.ndarray:
        base = build_gate_state(env, events, steps_since_replan)
        mask = np.zeros(self.max_agents, dtype=np.float32)
        for i, a in enumerate(live_agents[: self.max_agents]):
            mask[i] = 1.0 if getattr(a, "state", 0) != -1 else 0.0
        return np.concatenate([base, mask]).astype(np.float32)

    def act(self, state: np.ndarray, n_live: int, explore: bool = True) -> int:
        legal = list(range(min(n_live, self.max_agents) + 1))  # 0..n_live
        if explore and np.random.rand() < self.eps:
            return int(np.random.choice(legal))
        with torch.no_grad():
            q = self.net(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
            # Mask illegal
            vals = []
            for a in legal:
                vals.append((a, float(q[a].item())))
            return max(vals, key=lambda x: x[1])[0]

    def push(self, tr: GateTransition):
        self.buffer.append(tr)
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer :]

    def update(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device)
        r = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        d = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)
        q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(ns).max(dim=1)[0]
            target = r + self.gamma * (1.0 - d) * q_next
        loss = F.mse_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.n_updates += 1
        if self.n_updates % 50 == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "max_agents": self.max_agents}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.net.load_state_dict(data["net"])
        self.target.load_state_dict(data["net"])
        self.net.eval()
