"""
Reserve-Aware Hybrid (RAH): learned task priorities + reserve fraction reshape Local-Hungarian.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TaskAllocation.Hybrid.ReplanGate import GateTransition, build_gate_state


class RAHNet(nn.Module):
    """Outputs reserve scalar + up to max_tasks priority logits from global WPS state."""

    def __init__(self, state_dim: int = 14, max_tasks: int = 24, hidden: int = 128):
        super().__init__()
        self.max_tasks = max_tasks
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.reserve_head = nn.Linear(hidden, 1)
        self.priority_head = nn.Linear(hidden, max_tasks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        reserve = torch.sigmoid(self.reserve_head(h)).squeeze(-1)  # [B]
        priorities = torch.sigmoid(self.priority_head(h))  # [B, max_tasks]
        return reserve, priorities


def build_rah_state(env, events=None, steps_since_replan: int = 0) -> np.ndarray:
    base = build_gate_state(env, events, steps_since_replan)
    known = env.known_tasks_for(None)
    open_known = [
        t
        for t in known
        if t.id != 0 and t.status != 2 and t.allocatedReqs[t.typeIdx] < t.currentReqs[t.typeIdx]
    ]
    urgencies = []
    for t in open_known:
        dl = getattr(t, "hard_deadline", None)
        if dl is not None:
            urgencies.append(1.0 - min(max(dl - env.time_steps, 0) / 40.0, 1.0))
    mean_u = float(np.mean(urgencies)) if urgencies else 0.0
    n_known = len(open_known) / max(getattr(env, "max_tasks", 31), 1)
    miss = float(getattr(env, "n_missed_windows", 0)) / max(getattr(env, "n_windowed_tasks", 1), 1)
    return np.concatenate(
        [base, np.asarray([mean_u, n_known, miss, float(getattr(env, "burst_mode", 0))], dtype=np.float32)]
    ).astype(np.float32)


class ReserveAwareHybrid:
    """Epsilon-greedy DQN-style policy over discretized reserve + continuous priorities via actor net."""

    def __init__(
        self,
        state_dim: int = 14,
        max_tasks: int = 24,
        n_reserve_bins: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tasks = max_tasks
        self.n_reserve_bins = n_reserve_bins
        self.gamma = gamma
        self.net = RAHNet(state_dim, max_tasks).to(self.device)
        # Prefer low reserve at init (myopic Local is strong; reserve only when useful)
        nn.init.constant_(self.net.reserve_head.bias, -2.0)
        self.target = RAHNet(state_dim, max_tasks).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.eps = 0.2
        self.buffer: List[dict] = []
        self.max_buffer = 40_000
        self.n_updates = 0
        self.n_replans = 0

    def act(self, state: np.ndarray, explore: bool = True) -> Tuple[float, np.ndarray]:
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            reserve, priorities = self.net(x)
            rho = float(reserve.item())
            pri = priorities[0].cpu().numpy()
        if explore and np.random.rand() < self.eps:
            rho = float(np.random.rand() * 0.25)  # reserve up to 25%
            pri = np.clip(pri + np.random.randn(*pri.shape) * 0.2, 0.0, 1.0)
        return min(rho, 0.3), pri

    def push(self, state, rho, pri, reward, next_state, done):
        self.buffer.append(
            {
                "state": state,
                "rho": rho,
                "pri": pri[: self.max_tasks],
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
        )
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer :]

    def update(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.stack([b["next_state"] for b in batch]), dtype=torch.float32, device=self.device)
        r = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=self.device)
        d = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device)
        rho_t = torch.tensor([b["rho"] for b in batch], dtype=torch.float32, device=self.device)
        pri_t = torch.tensor(np.stack([b["pri"] for b in batch]), dtype=torch.float32, device=self.device)

        rho_pred, pri_pred = self.net(s)
        # Behavioral cloning toward taken actions + TD on scalar value proxy = -reserve waste + reward
        # Use reward regression through a value estimate = mean priority * (1-rho)
        value = (pri_pred.mean(dim=1) * (1.0 - rho_pred))
        with torch.no_grad():
            n_rho, n_pri = self.target(ns)
            n_value = n_pri.mean(dim=1) * (1.0 - n_rho)
            target = r + self.gamma * (1.0 - d) * n_value
        loss_v = F.mse_loss(value, target)
        loss_rho = F.mse_loss(rho_pred, rho_t)
        loss_pri = F.mse_loss(pri_pred, pri_t)
        loss = loss_v + 0.5 * loss_rho + 0.5 * loss_pri
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.n_updates += 1
        if self.n_updates % 40 == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "max_tasks": self.max_tasks}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.net.load_state_dict(data["net"])
        self.target.load_state_dict(data["net"])
        self.net.eval()

    def plan(self, env, hung, events=None, force: bool = True):
        """Apply RAH priorities/reserve then Local-Hungarian on known tasks."""
        state = build_rah_state(env, events, 0)
        rho, pri_vec = self.act(state, explore=False)
        open_known = [
            t
            for t in env.tasks
            if t.id != 0 and t.status != 2 and t.allocatedReqs[t.typeIdx] < t.currentReqs[t.typeIdx]
        ]
        # Blend learned priorities with deadline urgency (foresight residual on top of myopic urgency)
        task_pri = {}
        n_urgent = 0
        for i, t in enumerate(open_known[: self.max_tasks]):
            urgency = 0.0
            dl = getattr(t, "hard_deadline", None)
            if dl is not None:
                remaining = max(dl - env.time_steps, 0)
                urgency = 1.0 - min(remaining / 40.0, 1.0)
                if remaining <= 12:
                    n_urgent += 1
            # Extra boost for pop-ups known to few agents (local-info foresight)
            scarcity = 0.0
            vis = env.agent_visibility_map()
            if vis is not None:
                n_know = sum(1 for s in vis.values() if t.id in s)
                scarcity = 1.0 - min(n_know / max(len(vis), 1), 1.0)
            task_pri[t.id] = 0.4 * urgency + 0.35 * float(pri_vec[i]) + 0.25 * scarcity
        live = env.get_live_agents()
        # Soft reserve: only hold agents when several tight windows are open
        rho = min(float(rho), 0.25)
        if n_urgent >= 3:
            rho = max(rho, min(0.2, 0.05 * (n_urgent - 2)))
        elif n_urgent <= 1:
            rho = min(rho, 0.05)
        n_reserve = int(round(rho * len(live)))
        reserved = []
        vis = env.agent_visibility_map()
        if n_reserve > 0 and open_known:
            scores = []
            for a in live:
                known_ids = None if vis is None else vis.get(a.name, set())
                visible = [t for t in open_known if known_ids is None or t.id in known_ids]
                if not visible:
                    scores.append((1e9, a.name))
                    continue
                dmin = min(float(np.linalg.norm(a.position - t.position)) for t in visible)
                scores.append((dmin, a.name))
            scores.sort(reverse=True)  # farthest first → reserve
            reserved = [name for _, name in scores[:n_reserve]]
        result = hung.allocate_tasks(
            live,
            open_known,
            time_step=env.time_steps,
            events=events,
            force=force,
            task_priorities=task_pri,
            reserved_agent_names=reserved,
            agent_known_ids=vis,
        )
        self.n_replans += 1 if result else 0
        return result, rho, task_pri, state
