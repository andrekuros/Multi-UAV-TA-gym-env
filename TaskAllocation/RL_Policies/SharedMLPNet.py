"""Simple shared MLP DQN network (no attention) for architecture ablation."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net


class SharedMLPNet(Net):
    """Flattens position/caps/alloc + padded task vectors into an MLP Q-head."""

    def __init__(
        self,
        state_shape_agent: int,
        state_shape_task: int,
        action_shape: int,
        hidden_sizes: List[int],
        device: str,
        task_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(state_shape=0, action_shape=action_shape, hidden_sizes=hidden_sizes, device=device)
        self.device = device
        self.max_tasks = 31
        self.task_size = task_size or 10
        in_dim = 2 + 6 + 1 + self.max_tasks * self.task_size  # pos + caps + alloc + tasks
        layers: List[nn.Module] = []
        last = in_dim
        for h in (hidden_sizes or [256, 256]):
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, self.max_tasks))
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, obs, state=None, info=None):
        batch = len(obs["agent_position"])
        feats = []
        for b in range(batch):
            pos = np.asarray(obs["agent_position"][b], dtype=np.float32).reshape(-1)
            caps = np.asarray(obs["agent_caps"][b], dtype=np.float32).reshape(-1)
            alloc = np.asarray([obs["alloc_task"][b]], dtype=np.float32)
            tasks = obs["tasks_info"][b]
            flat_tasks = []
            for t in tasks[: self.max_tasks]:
                if t.get("status", -1) == -1:
                    flat_tasks.extend([-1.0] * self.task_size)
                else:
                    pos_t = np.asarray(t["position"], dtype=np.float32).reshape(-1)
                    req = np.asarray(t["current_reqs"], dtype=np.float32).reshape(-1)
                    allo = np.asarray(t["alloc_reqs"], dtype=np.float32).reshape(-1)
                    vec = np.concatenate([pos_t, req[:4], allo[:4]])[: self.task_size]
                    if len(vec) < self.task_size:
                        vec = np.pad(vec, (0, self.task_size - len(vec)))
                    flat_tasks.extend(vec.tolist())
            while len(flat_tasks) < self.max_tasks * self.task_size:
                flat_tasks.append(-1.0)
            row = np.concatenate([pos[:2], caps[:6], alloc, np.asarray(flat_tasks[: self.max_tasks * self.task_size])])
            feats.append(row)
        x = torch.tensor(np.asarray(feats), dtype=torch.float32, device=self.device)
        logits = self.net(x)
        return logits, state
