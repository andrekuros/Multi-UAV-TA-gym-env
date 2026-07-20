"""
Attention-Commit: set-attention → task priorities + per-agent commit locks → Local-Hungarian on free agents.

Also provides MLP-Commit (flat encoder) and Urgency-Commit (hand rule) controls.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TaskAllocation.Hybrid.AttentionRAH import (
    AGENT_FEAT_DIM,
    TASK_FEAT_DIM,
    _scarcity,
    _urgency,
    build_att_tokens,
)


def committed_names(env) -> List[str]:
    t = int(env.time_steps)
    return [
        a.name
        for a in env.get_live_agents()
        if int(getattr(a, "commit_until", 0) or 0) > t
    ]


def apply_agent_commits(env, names: List[str], horizon: int) -> None:
    if horizon <= 0:
        return
    until = int(env.time_steps) + int(horizon)
    by_name = {a.name: a for a in env.get_live_agents()}
    for name in names:
        a = by_name.get(name)
        if a is None:
            continue
        # Only lock agents that currently hold a real task
        if a.tasks and a.tasks[0].id != 0:
            a.commit_until = until


def enrich_commit_tokens(env, tok: dict) -> dict:
    """Append commit-remaining fraction to agent features (does not mutate Att-RAH dims in-place)."""
    af = tok["agent_feats"].copy()
    am = tok["agent_mask"]
    live = tok["live"]
    horizon = max(int(getattr(env, "commit_horizon", 25) or 25), 1)
    t = float(env.time_steps)
    extra = np.zeros((af.shape[0], 1), dtype=np.float32)
    for i, a in enumerate(live[: af.shape[0]]):
        if am[i]:
            continue
        rem = max(float(getattr(a, "commit_until", 0) or 0) - t, 0.0)
        extra[i, 0] = min(rem / horizon, 1.0)
    tok = dict(tok)
    tok["agent_feats"] = np.concatenate([af, extra], axis=1)
    return tok


AGENT_FEAT_DIM_C = AGENT_FEAT_DIM + 1


class AttCommitNet(nn.Module):
    def __init__(
        self,
        max_tasks: int = 32,
        max_agents: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.task_proj = nn.Linear(TASK_FEAT_DIM, d_model)
        self.agent_proj = nn.Linear(AGENT_FEAT_DIM_C, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.priority_head = nn.Linear(d_model, 1)
        self.commit_head = nn.Linear(d_model, 1)

    def forward(self, task_feats, task_mask, agent_feats, agent_mask):
        t_emb = self.task_proj(task_feats) + self.type_embed.weight[1]
        a_emb = self.agent_proj(agent_feats) + self.type_embed.weight[0]
        tokens = torch.cat([a_emb, t_emb], dim=1)
        pad_mask = torch.cat([agent_mask, task_mask], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=pad_mask)
        a_h = h[:, : self.max_agents, :]
        t_h = h[:, self.max_agents :, :]
        priorities = torch.sigmoid(self.priority_head(t_h).squeeze(-1)).masked_fill(task_mask, 0.0)
        commits = torch.sigmoid(self.commit_head(a_h).squeeze(-1)).masked_fill(agent_mask, 0.0)
        return priorities, commits


class MLPCommitNet(nn.Module):
    """Flat MLP over pooled agent/task features — architecture ablation."""

    def __init__(self, max_tasks: int = 32, max_agents: int = 16, hidden: int = 128):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        in_dim = max_tasks * TASK_FEAT_DIM + max_agents * AGENT_FEAT_DIM_C
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.priority_head = nn.Linear(hidden, max_tasks)
        self.commit_head = nn.Linear(hidden, max_agents)

    def forward(self, task_feats, task_mask, agent_feats, agent_mask):
        b = task_feats.size(0)
        flat = torch.cat(
            [task_feats.reshape(b, -1), agent_feats.reshape(b, -1)], dim=1
        )
        h = self.backbone(flat)
        priorities = torch.sigmoid(self.priority_head(h)).masked_fill(task_mask, 0.0)
        commits = torch.sigmoid(self.commit_head(h)).masked_fill(agent_mask, 0.0)
        return priorities, commits


class AttentionCommit:
    """Learned commit + priority policy wrapping Local-Hungarian on free agents."""

    def __init__(
        self,
        max_tasks: int = 32,
        max_agents: int = 16,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: Optional[str] = None,
        use_attention: bool = True,
        commit_threshold: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.gamma = gamma
        self.use_attention = use_attention
        self.commit_threshold = commit_threshold
        Net = AttCommitNet if use_attention else MLPCommitNet
        self.net = Net(max_tasks, max_agents).to(self.device)
        self.target = Net(max_tasks, max_agents).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.eps = 0.2
        self.buffer: List[dict] = []
        self.max_buffer = 40_000
        self.n_updates = 0
        self.n_replans = 0

    def _batch_from_tokens(self, tok: dict):
        tf = torch.tensor(tok["task_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
        tm = torch.tensor(tok["task_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        af = torch.tensor(tok["agent_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
        am = torch.tensor(tok["agent_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        return tf, tm, af, am

    def build_tokens(self, env) -> dict:
        return enrich_commit_tokens(env, build_att_tokens(env, self.max_tasks, self.max_agents))

    def act(self, tok: dict, explore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            pri_t, com_t = self.net(*self._batch_from_tokens(tok))
            pri = pri_t[0].cpu().numpy()
            com = com_t[0].cpu().numpy()
        if explore and np.random.rand() < self.eps:
            pri = np.clip(pri + np.random.randn(*pri.shape) * 0.2, 0.0, 1.0)
            com = np.clip(com + np.random.randn(*com.shape) * 0.2, 0.0, 1.0)
        return pri, com

    def push(self, tok, pri, com, reward, next_tok, done):
        keys = ("task_feats", "task_mask", "agent_feats", "agent_mask")
        self.buffer.append(
            {
                "tok": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tok.items() if k in keys},
                "pri": np.asarray(pri[: self.max_tasks], dtype=np.float32),
                "com": np.asarray(com[: self.max_agents], dtype=np.float32),
                "reward": reward,
                "next_tok": {
                    k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in next_tok.items() if k in keys
                },
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

        def stack_key(src, key, dtype=torch.float32):
            arr = np.stack([b[src][key] for b in batch])
            return torch.tensor(arr, dtype=dtype, device=self.device)

        tf, tm = stack_key("tok", "task_feats"), stack_key("tok", "task_mask", torch.bool)
        af, am = stack_key("tok", "agent_feats"), stack_key("tok", "agent_mask", torch.bool)
        ntf, ntm = stack_key("next_tok", "task_feats"), stack_key("next_tok", "task_mask", torch.bool)
        naf, nam = stack_key("next_tok", "agent_feats"), stack_key("next_tok", "agent_mask", torch.bool)

        r = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=self.device)
        d = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device)
        pri_t = torch.tensor(np.stack([b["pri"] for b in batch]), dtype=torch.float32, device=self.device)
        com_t = torch.tensor(np.stack([b["com"] for b in batch]), dtype=torch.float32, device=self.device)

        pri_pred, com_pred = self.net(tf, tm, af, am)
        n_tasks = (~tm).float().sum(dim=1).clamp(min=1.0)
        n_agents = (~am).float().sum(dim=1).clamp(min=1.0)
        value = pri_pred.sum(dim=1) / n_tasks + 0.5 * com_pred.sum(dim=1) / n_agents
        with torch.no_grad():
            n_pri, n_com = self.target(ntf, ntm, naf, nam)
            n_nt = (~ntm).float().sum(dim=1).clamp(min=1.0)
            n_na = (~nam).float().sum(dim=1).clamp(min=1.0)
            n_value = n_pri.sum(dim=1) / n_nt + 0.5 * n_com.sum(dim=1) / n_na
            target = r + self.gamma * (1.0 - d) * n_value
        loss_v = F.mse_loss(value, target)
        valid_t = (~tm).float()
        valid_a = (~am).float()
        loss_pri = ((pri_pred - pri_t) ** 2 * valid_t).sum() / valid_t.sum().clamp(min=1.0)
        loss_com = ((com_pred - com_t) ** 2 * valid_a).sum() / valid_a.sum().clamp(min=1.0)
        loss = loss_v + 0.5 * loss_pri + 0.5 * loss_com
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optim.step()
        self.n_updates += 1
        if self.n_updates % 40 == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {
                "net": self.net.state_dict(),
                "max_tasks": self.max_tasks,
                "max_agents": self.max_agents,
                "use_attention": self.use_attention,
                "kind": "AttentionCommit" if self.use_attention else "MLPCommit",
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.net.load_state_dict(data["net"])
        self.target.load_state_dict(data["net"])
        self.net.eval()

    def plan(self, env, hung, events=None, force: bool = True):
        tok = self.build_tokens(env)
        pri_vec, com_vec = self.act(tok, explore=False)
        return self._plan_from_scores(env, hung, tok, pri_vec, com_vec, events=events, force=force)

    def _plan_from_scores(self, env, hung, tok, pri_vec, com_vec, events=None, force: bool = True):
        open_known = tok["open_tasks"]
        vis = tok["vis"]
        live = tok["live"]
        reserved = committed_names(env)

        task_pri: Dict[int, float] = {}
        for i, t in enumerate(open_known[: self.max_tasks]):
            urg = _urgency(t, env.time_steps)
            scar = _scarcity(t, vis, max(len(live), 1))
            task_pri[t.id] = 0.35 * urg + 0.40 * float(pri_vec[i]) + 0.25 * scar

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
        assigned = {name for name, _ in result}
        horizon = int(getattr(env, "commit_horizon", 25) or 25)
        to_commit = []
        for i, a in enumerate(live[: self.max_agents]):
            if a.name in reserved:
                continue
            if a.name not in assigned:
                continue
            if float(com_vec[i]) >= self.commit_threshold:
                to_commit.append(a.name)
        apply_agent_commits(env, to_commit, horizon)
        self.n_replans += 1 if result else 0
        return result, task_pri, to_commit, tok


class UrgencyCommit:
    """Hand-rule commit: lock specialists / agents farthest from nearest urgent known task."""

    def __init__(self, commit_fraction: float = 0.35):
        self.commit_fraction = commit_fraction
        self.n_replans = 0

    def plan(self, env, hung, events=None, force: bool = True):
        tok = enrich_commit_tokens(env, build_att_tokens(env))
        open_known = tok["open_tasks"]
        vis = tok["vis"]
        live = tok["live"]
        reserved = committed_names(env)

        task_pri: Dict[int, float] = {}
        for t in open_known:
            urg = _urgency(t, env.time_steps)
            scar = _scarcity(t, vis, max(len(live), 1))
            task_pri[t.id] = 0.6 * urg + 0.4 * scar

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
        assigned = {name for name, _ in result}
        free_assigned = [a for a in live if a.name in assigned and a.name not in reserved]

        scores = []
        for a in free_assigned:
            known_ids = None if vis is None else vis.get(a.name, set())
            urgent = [
                t
                for t in open_known
                if (known_ids is None or t.id in known_ids)
                and getattr(t, "hard_deadline", None) is not None
                and _urgency(t, env.time_steps) >= (1.0 - 12.0 / 40.0)
            ]
            if urgent:
                dmin = min(float(np.linalg.norm(a.position - t.position)) for t in urgent)
            else:
                dmin = 0.0
            bonus = 500.0 if getattr(a, "type", "") == "F2" else 0.0
            scores.append((dmin + bonus, a.name))
        scores.sort(reverse=True)
        n_lock = max(1, int(round(self.commit_fraction * max(len(free_assigned), 1))))
        to_commit = [name for _, name in scores[:n_lock]]
        apply_agent_commits(env, to_commit, int(getattr(env, "commit_horizon", 25) or 25))
        self.n_replans += 1 if result else 0
        return result, task_pri, to_commit, tok
