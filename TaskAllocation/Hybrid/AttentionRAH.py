"""
Attention-RAH: set-attention over agent/task tokens → priorities + reserve → Local-Hungarian.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_FEAT_DIM = 13
AGENT_FEAT_DIM = 12

# Raw mode drops every feature that aggregates team/global state (task scarcity,
# known_by_count, nearest-specialist distance, region id, agent n_known_urgent) so the
# encoder must infer contention itself instead of reading it off an engineered input.
RAW_TASK_FEAT_DIM = 9
RAW_AGENT_FEAT_DIM = 11


def feat_dims(raw: bool = False):
    if raw:
        return RAW_TASK_FEAT_DIM, RAW_AGENT_FEAT_DIM
    return TASK_FEAT_DIM, AGENT_FEAT_DIM


def _urgency(task, time_step: int) -> float:
    dl = getattr(task, "hard_deadline", None)
    if dl is None:
        return 0.0
    remaining = max(dl - time_step, 0)
    return 1.0 - min(remaining / 40.0, 1.0)


def _scarcity(task, vis, n_agents: int) -> float:
    if vis is None or n_agents <= 0:
        return 0.0
    n_know = sum(1 for s in vis.values() if task.id in s)
    return 1.0 - min(n_know / max(n_agents, 1), 1.0)


def _known_by_count(task, vis) -> float:
    if vis is None:
        return 1.0
    return float(sum(1 for s in vis.values() if task.id in s))


def build_att_tokens(env, max_tasks: int = 32, max_agents: int = 16, raw: bool = False):
    """
    Build padded task/agent token matrices + masks for Attention-RAH.

    Enriched features for WPS_attn: known_by_count, nearest_specialist_dist, region_id,
    agent n_known_urgent, is_specialist (F2).

    With raw=True only per-entity attributes are emitted (no team aggregates), so that
    Att vs MLP measures whether the encoder discovers relational context on its own.
    """
    max_coord = float(getattr(env, "max_coord", 1000.0) or 1000.0)
    horizon = max(getattr(env, "max_time_steps", 150), 1)
    mid_x = float(getattr(env, "area_width", max_coord)) * 0.5
    vis = env.agent_visibility_map()
    live = env.get_live_agents()
    n_agents = max(len(live), 1)
    specialists = [a for a in live if getattr(a, "type", "") == "F2"]
    open_tasks = [
        t
        for t in env.tasks
        if t.id != 0 and t.status != 2 and t.allocatedReqs[t.typeIdx] < t.currentReqs[t.typeIdx]
    ]

    task_dim, agent_dim = feat_dims(raw)
    task_feats = np.zeros((max_tasks, task_dim), dtype=np.float32)
    task_mask = np.ones(max_tasks, dtype=bool)  # True = ignore (padding)
    task_ids: List[int] = []
    n_urgent = 0

    for i, t in enumerate(open_tasks[:max_tasks]):
        urg = _urgency(t, env.time_steps)
        scar = _scarcity(t, vis, n_agents)
        rem = max(float(t.currentReqs[t.typeIdx] - t.allocatedReqs[t.typeIdx]), 0.0)
        is_dynamic = 1.0 if getattr(t, "hard_deadline", None) is not None else 0.0
        if urg >= (1.0 - 12.0 / 40.0) and is_dynamic:
            n_urgent += 1
        ttype = getattr(t, "type", "")
        n_know = _known_by_count(t, vis)
        if specialists:
            d_spec = min(float(np.linalg.norm(a.position - t.position)) for a in specialists)
        else:
            d_spec = max_coord
        region = 0.0 if float(t.position[0]) < mid_x else 1.0
        if raw:
            dl = getattr(t, "hard_deadline", None)
            t_left = 1.0 if dl is None else min(max(dl - env.time_steps, 0) / horizon, 1.0)
            task_feats[i] = [
                float(t.position[0]) / max_coord,
                float(t.position[1]) / max_coord,
                float(getattr(t, "typeIdx", 0)) / 8.0,
                1.0 if ttype == "Att" else 0.0,
                1.0 if ttype == "Rec" else 0.0,
                1.0 if ttype == "Int" else 0.0,
                t_left,
                min(rem / 4.0, 1.0),
                is_dynamic,
            ]
        else:
            task_feats[i] = [
                float(t.position[0]) / max_coord,
                float(t.position[1]) / max_coord,
                float(getattr(t, "typeIdx", 0)) / 8.0,
                1.0 if ttype == "Att" else 0.0,
                1.0 if ttype == "Rec" else 0.0,
                1.0 if ttype == "Int" else 0.0,
                urg,
                scar,
                min(rem / 4.0, 1.0),
                is_dynamic,
                min(n_know / max(n_agents, 1), 1.0),
                min(d_spec / max_coord, 1.0),
                region,
            ]
        task_mask[i] = False
        task_ids.append(t.id)

    agent_feats = np.zeros((max_agents, agent_dim), dtype=np.float32)
    agent_mask = np.ones(max_agents, dtype=bool)
    for i, a in enumerate(live[:max_agents]):
        caps = getattr(a, "currentCap2Task", None)
        cap_att = float(caps[2]) if caps is not None and len(caps) > 2 else 0.0
        cap_def = float(caps[3]) if caps is not None and len(caps) > 3 else 0.0
        cap_rec = float(caps[1]) if caps is not None and len(caps) > 1 else 0.0
        idle = 1.0 if (not a.tasks) or a.tasks[0].id == 0 else 0.0
        atype = getattr(a, "type", "")
        known_ids = set() if vis is None else vis.get(a.name, set())
        n_known_urgent = 0
        for t in open_tasks:
            if t.id not in known_ids and vis is not None:
                continue
            if _urgency(t, env.time_steps) >= (1.0 - 12.0 / 40.0) and getattr(t, "hard_deadline", None) is not None:
                n_known_urgent += 1
        base = [
            float(a.position[0]) / max_coord,
            float(a.position[1]) / max_coord,
            1.0 if atype.startswith("F") else 0.0,
            1.0 if atype.startswith("R") else 0.0,
            idle,
            min(cap_att / 2.0, 1.0),
            min(cap_def / 2.0, 1.0),
            min(cap_rec / 2.0, 1.0),
            float(getattr(a, "state", 0)) / 5.0,
            float(env.time_steps) / horizon,
        ]
        if raw:
            agent_feats[i] = base + [1.0 if atype == "F2" else 0.0]
        else:
            agent_feats[i] = base + [
                min(n_known_urgent / 8.0, 1.0),
                1.0 if atype == "F2" else 0.0,
            ]
        agent_mask[i] = False

    return {
        "task_feats": task_feats,
        "task_mask": task_mask,
        "agent_feats": agent_feats,
        "agent_mask": agent_mask,
        "task_ids": task_ids,
        "open_tasks": open_tasks,
        "n_urgent": n_urgent,
        "vis": vis,
        "live": live,
    }


class AttRAHNet(nn.Module):
    """1–2 layer multi-head attention over concatenated agent+task tokens."""

    def __init__(
        self,
        max_tasks: int = 32,
        max_agents: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        task_feat_dim: int = TASK_FEAT_DIM,
        agent_feat_dim: int = AGENT_FEAT_DIM,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.d_model = d_model
        self.task_feat_dim = task_feat_dim
        self.agent_feat_dim = agent_feat_dim
        self.task_proj = nn.Linear(task_feat_dim, d_model)
        self.agent_proj = nn.Linear(agent_feat_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)  # 0=agent, 1=task
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.priority_head = nn.Linear(d_model, 1)
        self.reserve_head = nn.Linear(d_model, 1)

    def forward(
        self,
        task_feats: torch.Tensor,
        task_mask: torch.Tensor,
        agent_feats: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        task_feats: [B, T, F], task_mask True=pad
        agent_feats: [B, A, F], agent_mask True=pad
        returns: rho [B], priorities [B, T]
        """
        b = task_feats.size(0)
        t_emb = self.task_proj(task_feats) + self.type_embed.weight[1]
        a_emb = self.agent_proj(agent_feats) + self.type_embed.weight[0]
        tokens = torch.cat([a_emb, t_emb], dim=1)  # [B, A+T, D]
        pad_mask = torch.cat([agent_mask, task_mask], dim=1)  # True = pad
        # TransformerEncoder key_padding_mask: True = ignore
        h = self.encoder(tokens, src_key_padding_mask=pad_mask)
        a_h = h[:, : self.max_agents, :]
        t_h = h[:, self.max_agents :, :]

        pri_logits = self.priority_head(t_h).squeeze(-1)  # [B, T]
        priorities = torch.sigmoid(pri_logits)
        priorities = priorities.masked_fill(task_mask, 0.0)

        # Pool valid agent+task tokens for reserve
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        rho = torch.sigmoid(self.reserve_head(pooled)).squeeze(-1)
        return rho, priorities


class AttentionRAH:
    """Attention-based RAH policy wrapping Local-Hungarian."""

    def __init__(
        self,
        max_tasks: int = 32,
        max_agents: int = 16,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: Optional[str] = None,
        use_learned_priority: bool = True,
        use_reserve: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.gamma = gamma
        self.use_learned_priority = use_learned_priority
        self.use_reserve = use_reserve
        self.net = AttRAHNet(max_tasks, max_agents).to(self.device)
        nn.init.constant_(self.net.reserve_head.bias, -2.0)
        self.target = AttRAHNet(max_tasks, max_agents).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lr = lr
        self.eps = 0.2
        self.buffer: List[dict] = []
        self.max_buffer = 40_000
        self.n_updates = 0
        self.n_replans = 0

    def _batch_from_tokens(self, tok: dict) -> Tuple[torch.Tensor, ...]:
        tdim = self.net.task_feat_dim
        adim = self.net.agent_feat_dim
        tf_np = np.asarray(tok["task_feats"], dtype=np.float32)[..., :tdim]
        af_np = np.asarray(tok["agent_feats"], dtype=np.float32)[..., :adim]
        if tf_np.shape[-1] < tdim:
            pad = np.zeros(tf_np.shape[:-1] + (tdim - tf_np.shape[-1],), dtype=np.float32)
            tf_np = np.concatenate([tf_np, pad], axis=-1)
        if af_np.shape[-1] < adim:
            pad = np.zeros(af_np.shape[:-1] + (adim - af_np.shape[-1],), dtype=np.float32)
            af_np = np.concatenate([af_np, pad], axis=-1)
        tf = torch.tensor(tf_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        tm = torch.tensor(tok["task_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        af = torch.tensor(af_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        am = torch.tensor(tok["agent_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        return tf, tm, af, am

    def act(self, tok: dict, explore: bool = True) -> Tuple[float, np.ndarray]:
        with torch.no_grad():
            rho_t, pri_t = self.net(*self._batch_from_tokens(tok))
            rho = float(rho_t.item())
            pri = pri_t[0].cpu().numpy()
        if explore and np.random.rand() < self.eps:
            rho = float(np.random.rand() * 0.25)
            pri = np.clip(pri + np.random.randn(*pri.shape) * 0.2, 0.0, 1.0)
        return min(rho, 0.3), pri

    def push(self, tok, rho, pri, reward, next_tok, done):
        self.buffer.append(
            {
                "tok": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tok.items() if k in (
                    "task_feats", "task_mask", "agent_feats", "agent_mask"
                )},
                "rho": rho,
                "pri": np.asarray(pri[: self.max_tasks], dtype=np.float32),
                "reward": reward,
                "next_tok": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in next_tok.items() if k in (
                    "task_feats", "task_mask", "agent_feats", "agent_mask"
                )},
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

        def stack_key(key, dtype=torch.float32):
            arr = np.stack([b["tok"][key] for b in batch])
            return torch.tensor(arr, dtype=dtype, device=self.device)

        def stack_nkey(key, dtype=torch.float32):
            arr = np.stack([b["next_tok"][key] for b in batch])
            return torch.tensor(arr, dtype=dtype, device=self.device)

        tf, tm = stack_key("task_feats"), stack_key("task_mask", torch.bool)
        af, am = stack_key("agent_feats"), stack_key("agent_mask", torch.bool)
        ntf, ntm = stack_nkey("task_feats"), stack_nkey("task_mask", torch.bool)
        naf, nam = stack_nkey("agent_feats"), stack_nkey("agent_mask", torch.bool)

        r = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=self.device)
        d = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device)
        rho_t = torch.tensor([b["rho"] for b in batch], dtype=torch.float32, device=self.device)
        pri_t = torch.tensor(np.stack([b["pri"] for b in batch]), dtype=torch.float32, device=self.device)

        rho_pred, pri_pred = self.net(tf, tm, af, am)
        value = pri_pred.sum(dim=1) / (~tm).float().sum(dim=1).clamp(min=1.0) * (1.0 - rho_pred)
        with torch.no_grad():
            n_rho, n_pri = self.target(ntf, ntm, naf, nam)
            n_value = n_pri.sum(dim=1) / (~ntm).float().sum(dim=1).clamp(min=1.0) * (1.0 - n_rho)
            target = r + self.gamma * (1.0 - d) * n_value
        loss_v = F.mse_loss(value, target)
        loss_rho = F.mse_loss(rho_pred, rho_t)
        # only supervise valid task slots
        valid = (~tm).float()
        loss_pri = ((pri_pred - pri_t) ** 2 * valid).sum() / valid.sum().clamp(min=1.0)
        loss = loss_v + 0.5 * loss_rho + 0.5 * loss_pri
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
                "kind": "AttentionRAH",
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        state = data["net"]
        tw = state.get("task_proj.weight")
        aw = state.get("agent_proj.weight")
        if tw is not None and aw is not None:
            tdim, adim = int(tw.shape[1]), int(aw.shape[1])
            if tdim != self.net.task_feat_dim or adim != self.net.agent_feat_dim:
                self.net = AttRAHNet(
                    self.max_tasks,
                    self.max_agents,
                    task_feat_dim=tdim,
                    agent_feat_dim=adim,
                ).to(self.device)
                self.target = AttRAHNet(
                    self.max_tasks,
                    self.max_agents,
                    task_feat_dim=tdim,
                    agent_feat_dim=adim,
                ).to(self.device)
                self.optim = torch.optim.Adam(self.net.parameters(), lr=getattr(self, "lr", 1e-3))
        self.net.load_state_dict(state)
        self.target.load_state_dict(state)
        self.net.eval()

    def plan(self, env, hung, events=None, force: bool = True, force_no_reserve: bool = False, force_no_learned_pri: bool = False):
        tok = build_att_tokens(env, self.max_tasks, self.max_agents)
        rho, pri_vec = self.act(tok, explore=False)
        open_known = tok["open_tasks"]
        vis = tok["vis"]
        live = tok["live"]
        n_urgent = tok["n_urgent"]

        use_lp = self.use_learned_priority and not force_no_learned_pri
        use_res = self.use_reserve and not force_no_reserve

        task_pri: Dict[int, float] = {}
        for i, t in enumerate(open_known[: self.max_tasks]):
            urg = _urgency(t, env.time_steps)
            scar = _scarcity(t, vis, max(len(live), 1))
            learned = float(pri_vec[i]) if use_lp else 0.0
            if use_lp:
                task_pri[t.id] = 0.35 * urg + 0.40 * learned + 0.25 * scar
            else:
                task_pri[t.id] = 0.6 * urg + 0.4 * scar

        rho = min(float(rho), 0.25) if use_res else 0.0
        if use_res:
            if n_urgent >= 3:
                rho = max(rho, min(0.2, 0.05 * (n_urgent - 2)))
            elif n_urgent <= 1:
                rho = min(rho, 0.05)
        else:
            rho = 0.0

        n_reserve = int(round(rho * len(live))) if use_res else 0
        reserved = []
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
            scores.sort(reverse=True)
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
        return result, rho, task_pri, tok
