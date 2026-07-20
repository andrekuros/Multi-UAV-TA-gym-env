"""
Att-Coalition v2: local tokens → cross-attention pair logits → coalition Hungarian
with selected-edge actor-critic training.

Also provides MLP-Coalition (flat pair scorer) and Urgency-Coalition controls.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TaskAllocation.Hybrid.AttentionCommit import apply_agent_commits, committed_names
from TaskAllocation.Hybrid.AttentionRAH import _scarcity, _urgency


# Base features (legacy dims) + escort extras
# Task: pos(2), typeIdx, Att/Rec/Int, urg, scar, rem, dyn, n_know, d_spec, region,
#       is_escort, deficit, pressure, prot_xy(2), req_agents, threat_dist, prot_alive, fighter_pressure
TASK_FEAT_DIM_E = 22
# Agent: pos(2), F/R, idle, caps(3), state, t, n_urg, F2, escorting, dist_prot, commit, near_mix
AGENT_FEAT_DIM_E = 16

DEFAULT_MAX_TASKS = 48
DEFAULT_MAX_AGENTS = 16


def _open_tasks_residual(env):
    out = []
    for t in env.tasks:
        if t.id == 0 or t.status == 2:
            continue
        if getattr(t, "kind", None) == "Escort" or float(getattr(t, "required_agents", 0) or 0) > 0:
            required = float(getattr(t, "required_agents", 1) or 1)
            allocated = len(getattr(t, "allocationDetails", {}) or {})
            if required - allocated > 0:
                out.append(t)
        elif t.allocatedReqs[t.typeIdx] < t.currentReqs[t.typeIdx]:
            out.append(t)
    return out


def _threat_stats(env, task, max_coord: float) -> Tuple[float, float, float]:
    """Return (pressure, nearest_threat_dist_norm, fighter_pressure)."""
    threats = getattr(env, "threats", None) or []
    anchor = task.position
    prot = getattr(task, "protected_agent", None)
    if prot is not None:
        anchor = prot.position
    best = max_coord
    n_near = 0
    for th in threats:
        if getattr(th, "status", 2) == 2:
            continue
        d = float(np.linalg.norm(np.asarray(th.position) - np.asarray(anchor)))
        best = min(best, d)
        if d < 150.0:
            n_near += 1
    pressure = 1.0 - min(best / max_coord, 1.0)
    dist_n = min(best / max_coord, 1.0)
    fighter_pressure = min(n_near / 4.0, 1.0)
    return pressure, dist_n, fighter_pressure


def _task_priority_key(env, task, max_coord: float):
    urg = _urgency(task, env.time_steps)
    pressure, _, _ = _threat_stats(env, task, max_coord)
    is_escort = 1.0 if getattr(task, "kind", None) == "Escort" else 0.0
    is_int = 1.0 if getattr(task, "type", "") == "Int" else 0.0
    return -(1.5 * urg + 1.2 * pressure + 0.8 * is_escort + 0.5 * is_int)


def build_escort_tokens(env, max_tasks: int = DEFAULT_MAX_TASKS, max_agents: int = DEFAULT_MAX_AGENTS) -> dict:
    max_coord = float(getattr(env, "max_coord", 1000.0) or 1000.0)
    mid_x = float(getattr(env, "area_width", max_coord)) * 0.5
    vis = env.agent_visibility_map()
    live = env.get_live_agents()
    n_agents = max(len(live), 1)
    specialists = [a for a in live if getattr(a, "type", "") == "F2"]
    open_all = _open_tasks_residual(env)

    # Local task set: only tasks known by at least one live agent (if visibility on)
    if vis is None:
        open_tasks = list(open_all)
    else:
        known_union = set()
        for a in live:
            known_union |= set(vis.get(a.name, set()))
        open_tasks = [t for t in open_all if t.id in known_union]
        if not open_tasks:
            open_tasks = list(open_all)

    open_tasks.sort(key=lambda t: _task_priority_key(env, t, max_coord))
    horizon = max(int(getattr(env, "commit_horizon", 20) or 20), 1)
    t_now = float(env.time_steps)

    task_feats = np.zeros((max_tasks, TASK_FEAT_DIM_E), dtype=np.float32)
    task_mask = np.ones(max_tasks, dtype=bool)
    task_ids: List[int] = []
    kept_tasks = []

    for i, t in enumerate(open_tasks[:max_tasks]):
        urg = _urgency(t, env.time_steps)
        scar = _scarcity(t, vis, n_agents)
        if getattr(t, "kind", None) == "Escort" or float(getattr(t, "required_agents", 0) or 0) > 0:
            rem = max(
                float(getattr(t, "required_agents", 1) or 1)
                - len(getattr(t, "allocationDetails", {}) or {}),
                0.0,
            )
            req_agents = float(getattr(t, "required_agents", 1) or 1)
        else:
            rem = max(float(t.currentReqs[t.typeIdx] - t.allocatedReqs[t.typeIdx]), 0.0)
            req_agents = 1.0
        is_dynamic = 1.0 if getattr(t, "hard_deadline", None) is not None else 0.0
        ttype = getattr(t, "type", "")
        n_know = 0.0 if vis is None else float(sum(1 for s in vis.values() if t.id in s))
        if specialists:
            d_spec = min(float(np.linalg.norm(a.position - t.position)) for a in specialists)
        else:
            d_spec = max_coord
        region = 0.0 if float(t.position[0]) < mid_x else 1.0
        is_escort = 1.0 if getattr(t, "kind", None) == "Escort" else 0.0
        deficit = min(rem / 4.0, 1.0)
        pressure, threat_dist, fighter_pressure = _threat_stats(env, t, max_coord)
        prot = getattr(t, "protected_agent", None)
        prot_alive = 0.0
        if prot is not None:
            prot_x = float(prot.position[0]) / max_coord
            prot_y = float(prot.position[1]) / max_coord
            prot_alive = 0.0 if getattr(prot, "state", 0) == -1 else 1.0
        else:
            prot_x = float(t.position[0]) / max_coord
            prot_y = float(t.position[1]) / max_coord
        task_feats[i] = np.asarray(
            [
                float(t.position[0]) / max_coord,
                float(t.position[1]) / max_coord,
                float(getattr(t, "typeIdx", 0)) / 8.0,
                1.0 if ttype == "Att" else 0.0,
                1.0 if ttype == "Rec" else 0.0,
                1.0 if ttype == "Int" else 0.0,
                urg,
                scar,
                deficit,
                is_dynamic,
                min(n_know / max(n_agents, 1), 1.0),
                min(d_spec / max_coord, 1.0),
                region,
                is_escort,
                deficit,
                pressure,
                prot_x,
                prot_y,
                min(req_agents / 4.0, 1.0),
                threat_dist,
                prot_alive,
                fighter_pressure,
            ],
            dtype=np.float32,
        )
        task_mask[i] = False
        task_ids.append(t.id)
        kept_tasks.append(t)

    agent_feats = np.zeros((max_agents, AGENT_FEAT_DIM_E), dtype=np.float32)
    agent_mask = np.ones(max_agents, dtype=bool)
    edge_valid = np.zeros((max_agents, max_tasks), dtype=np.float32)

    for i, a in enumerate(live[:max_agents]):
        caps = getattr(a, "currentCap2Task", None)
        cap_att = float(caps[2]) if caps is not None and len(caps) > 2 else 0.0
        cap_def = float(caps[3]) if caps is not None and len(caps) > 3 else 0.0
        cap_rec = float(caps[1]) if caps is not None and len(caps) > 1 else 0.0
        idle = 1.0 if (not a.tasks) or a.tasks[0].id == 0 else 0.0
        atype = getattr(a, "type", "")
        known_ids = None if vis is None else vis.get(a.name, set())
        n_known_urgent = 0
        n_known_tasks = 0 if known_ids is None else len(known_ids)
        for t in open_all:
            if known_ids is not None and t.id not in known_ids:
                continue
            if _urgency(t, env.time_steps) >= (1.0 - 12.0 / 40.0) and getattr(t, "hard_deadline", None) is not None:
                n_known_urgent += 1
        is_escorting = 0.0
        dist_prot = 1.0
        near_escort = 0.0
        if a.tasks and a.tasks[0].id != 0 and getattr(a.tasks[0], "kind", None) == "Escort":
            is_escorting = 1.0
            prot = getattr(a.tasks[0], "protected_agent", None)
            if prot is not None:
                dist_prot = min(float(np.linalg.norm(a.position - prot.position)) / max_coord, 1.0)
                near_escort = 1.0 - dist_prot
        rem_commit = max(float(getattr(a, "commit_until", 0) or 0) - t_now, 0.0)
        agent_feats[i] = np.asarray(
            [
                float(a.position[0]) / max_coord,
                float(a.position[1]) / max_coord,
                1.0 if atype.startswith("F") else 0.0,
                1.0 if atype.startswith("R") else 0.0,
                idle,
                min(cap_att / 2.0, 1.0),
                min(cap_def / 2.0, 1.0),
                min(cap_rec / 2.0, 1.0),
                float(getattr(a, "state", 0)) / 5.0,
                float(env.time_steps) / max(getattr(env, "max_time_steps", 150), 1),
                min(n_known_urgent / 8.0, 1.0),
                1.0 if atype == "F2" else 0.0,
                is_escorting,
                dist_prot,
                min(rem_commit / horizon, 1.0),
                min(near_escort + n_known_tasks / 16.0, 1.0),
            ],
            dtype=np.float32,
        )
        agent_mask[i] = False

        for j, t in enumerate(kept_tasks):
            if known_ids is not None and t.id not in known_ids:
                continue
            eligible = getattr(t, "eligible_agent_types", None)
            if eligible is not None:
                elig = {eligible} if isinstance(eligible, str) else set(eligible)
                if atype not in elig:
                    continue
            edge_valid[i, j] = 1.0

    return {
        "task_feats": task_feats,
        "task_mask": task_mask,
        "agent_feats": agent_feats,
        "agent_mask": agent_mask,
        "edge_valid": edge_valid,
        "task_ids": task_ids,
        "open_tasks": kept_tasks,
        "vis": vis,
        "live": live,
    }


class AttCoalitionNet(nn.Module):
    """Cross-attention agent↔task encoder + pair logits + value head."""

    def __init__(
        self,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.d_model = d_model
        self.task_proj = nn.Linear(TASK_FEAT_DIM_E, d_model)
        self.agent_proj = nn.Linear(AGENT_FEAT_DIM_E, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.self_encoder = nn.TransformerEncoder(layer, num_layers=max(1, n_layers - 1))
        self.cross_a2t = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_t2a = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.pair_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, task_feats, task_mask, agent_feats, agent_mask):
        """
        Returns:
          logits [B,A,T], value [B]
        """
        t_emb = self.task_proj(task_feats) + self.type_embed.weight[1]
        a_emb = self.agent_proj(agent_feats) + self.type_embed.weight[0]
        tokens = torch.cat([a_emb, t_emb], dim=1)
        pad_mask = torch.cat([agent_mask, task_mask], dim=1)
        h = self.self_encoder(tokens, src_key_padding_mask=pad_mask)
        a_h = h[:, : self.max_agents, :]
        t_h = h[:, self.max_agents :, :]

        # Agent queries tasks; task queries agents
        a_ctx, _ = self.cross_a2t(
            a_h, t_h, t_h, key_padding_mask=task_mask, need_weights=False
        )
        t_ctx, _ = self.cross_t2a(
            t_h, a_h, a_h, key_padding_mask=agent_mask, need_weights=False
        )
        a_h = a_h + a_ctx
        t_h = t_h + t_ctx

        a_exp = a_h.unsqueeze(2).expand(-1, -1, self.max_tasks, -1)
        t_exp = t_h.unsqueeze(1).expand(-1, self.max_agents, -1, -1)
        pair = torch.cat([a_exp, t_exp, a_exp * t_exp], dim=-1)
        logits = self.pair_head(pair).squeeze(-1)
        logits = logits.masked_fill(agent_mask.unsqueeze(2), -1e9)
        logits = logits.masked_fill(task_mask.unsqueeze(1), -1e9)

        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value


class MLPCoalitionNet(nn.Module):
    """Flat MLP pair scorer + value head (no set attention)."""

    def __init__(
        self,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        hidden: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        in_dim = TASK_FEAT_DIM_E + AGENT_FEAT_DIM_E
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(max_tasks * TASK_FEAT_DIM_E + max_agents * AGENT_FEAT_DIM_E, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, task_feats, task_mask, agent_feats, agent_mask):
        b, a, _ = agent_feats.shape
        t = task_feats.size(1)
        a_exp = agent_feats.unsqueeze(2).expand(-1, -1, t, -1)
        t_exp = task_feats.unsqueeze(1).expand(-1, a, -1, -1)
        pair = torch.cat([a_exp, t_exp], dim=-1)
        logits = self.pair_mlp(pair).squeeze(-1)
        logits = logits.masked_fill(agent_mask.unsqueeze(2), -1e9)
        logits = logits.masked_fill(task_mask.unsqueeze(1), -1e9)
        flat = torch.cat([task_feats.reshape(b, -1), agent_feats.reshape(b, -1)], dim=1)
        value = self.value_mlp(flat).squeeze(-1)
        return logits, value


class AttentionEscort:
    """Learned agent–task edge scores + coalition Hungarian (actor-critic)."""

    def __init__(
        self,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        lr: float = 3e-4,
        gamma: float = 0.95,
        device: Optional[str] = None,
        use_attention: bool = True,
        commit_threshold: float = 0.5,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 3,
        explore_std: float = 0.35,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.gamma = gamma
        self.use_attention = use_attention
        self.commit_threshold = commit_threshold
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        self.explore_std = explore_std
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.lr = lr

        if use_attention:
            self.net = AttCoalitionNet(
                max_tasks, max_agents, d_model=d_model, nhead=nhead, n_layers=n_layers
            ).to(self.device)
            self.target = AttCoalitionNet(
                max_tasks, max_agents, d_model=d_model, nhead=nhead, n_layers=n_layers
            ).to(self.device)
        else:
            self.net = MLPCoalitionNet(max_tasks, max_agents, hidden=max(128, d_model * 2)).to(
                self.device
            )
            self.target = MLPCoalitionNet(
                max_tasks, max_agents, hidden=max(128, d_model * 2)
            ).to(self.device)
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
        return build_escort_tokens(env, self.max_tasks, self.max_agents)

    def _logits_to_scores(self, logits: torch.Tensor, edge_valid: Optional[np.ndarray] = None):
        scores = torch.sigmoid(logits)
        if edge_valid is not None:
            ev = torch.tensor(edge_valid, dtype=torch.float32, device=scores.device)
            if ev.dim() == 2:
                ev = ev.unsqueeze(0)
            scores = scores * ev
        return scores

    def act(
        self, tok: dict, explore: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns scores [A,T], noise [A,T], logits [A,T].
        """
        self.net.eval()
        with torch.no_grad():
            logits_t, _ = self.net(*self._batch_from_tokens(tok))
            logits = logits_t[0].cpu().numpy()
        noise = np.zeros_like(logits, dtype=np.float32)
        valid = tok.get("edge_valid")
        if explore and self.eps > 0:
            # Always add small Gaussian on valid edges; scale by eps
            std = self.explore_std * max(self.eps, 0.05)
            noise = np.random.randn(*logits.shape).astype(np.float32) * std
            if valid is not None:
                noise = noise * valid
            logits_noisy = logits + noise
        else:
            logits_noisy = logits
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits_noisy, -20, 20)))
        if valid is not None:
            scores = scores * valid
        # Zero padded
        scores = scores * (~tok["agent_mask"])[:, None] * (~tok["task_mask"])[None, :]
        return scores.astype(np.float32), noise, logits.astype(np.float32)

    def edge_score_dict(self, tok: dict, scores: np.ndarray) -> Dict[Tuple[str, int], float]:
        live = tok["live"]
        task_ids = tok["task_ids"]
        out: Dict[Tuple[str, int], float] = {}
        for i, agent in enumerate(live[: self.max_agents]):
            if tok["agent_mask"][i]:
                continue
            for j, tid in enumerate(task_ids):
                if tok["task_mask"][j]:
                    continue
                out[(agent.name, int(tid))] = float(scores[i, j])
        return out

    def _selected_mask(self, tok: dict, result) -> np.ndarray:
        mask = np.zeros((self.max_agents, self.max_tasks), dtype=np.float32)
        name_to_i = {
            a.name: i
            for i, a in enumerate(tok["live"][: self.max_agents])
            if not tok["agent_mask"][i]
        }
        tid_to_j = {tid: j for j, tid in enumerate(tok["task_ids"])}
        for agent_name, task in result:
            i = name_to_i.get(agent_name)
            j = tid_to_j.get(getattr(task, "id", None))
            if i is not None and j is not None:
                mask[i, j] = 1.0
        return mask

    def _plan_from_scores(self, env, hung, tok, scores, events=None, force=True):
        edge = self.edge_score_dict(tok, scores)
        reserved = committed_names(env)
        open_tasks = tok["open_tasks"]
        result = hung.allocate_tasks(
            env.get_live_agents(),
            open_tasks,
            time_step=env.time_steps,
            events=events,
            force=force,
            reserved_agent_names=reserved,
            agent_known_ids=env.agent_visibility_map(),
            edge_scores=edge,
        )
        self.n_replans = hung.n_replans
        assigned = [name for name, task in result if getattr(task, "id", 0) != 0]
        apply_agent_commits(env, assigned, int(getattr(env, "commit_horizon", 0) or 0))
        return result

    def plan(self, env, hung, events=None, explore: bool = False, force: bool = True):
        tok = self.build_tokens(env)
        scores, noise, logits = self.act(tok, explore=explore)
        result = self._plan_from_scores(env, hung, tok, scores, events=events, force=force)
        selected = self._selected_mask(tok, result)
        return result, tok, scores, noise, logits, selected

    def push(self, tok, scores, noise, logits, selected, reward, next_tok, done):
        self.buffer.append(
            {
                "tok": {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in tok.items()
                    if k not in ("open_tasks", "live", "vis")
                },
                "scores": np.asarray(scores, dtype=np.float32),
                "noise": np.asarray(noise, dtype=np.float32),
                "logits": np.asarray(logits, dtype=np.float32),
                "selected": np.asarray(selected, dtype=np.float32),
                "reward": float(reward),
                "next_tok": {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in next_tok.items()
                    if k not in ("open_tasks", "live", "vis")
                },
                "done": bool(done),
            }
        )
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer :]

    def update(self, batch_size: int = 64):
        if len(self.buffer) < min(batch_size, 16):
            return None
        bs = min(batch_size, len(self.buffer))
        self.net.train()
        idx = np.random.choice(len(self.buffer), size=bs, replace=False)
        batch = [self.buffer[i] for i in idx]

        def stack_tok(key, dtype=torch.float32):
            return torch.tensor(
                np.stack([b["tok"][key] for b in batch]), dtype=dtype, device=self.device
            )

        tf = stack_tok("task_feats")
        tm = stack_tok("task_mask", torch.bool).bool()
        af = stack_tok("agent_feats")
        am = stack_tok("agent_mask", torch.bool).bool()
        selected = torch.tensor(
            np.stack([b["selected"] for b in batch]), dtype=torch.float32, device=self.device
        )
        noise = torch.tensor(
            np.stack([b["noise"] for b in batch]), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device)

        ntf = torch.tensor(
            np.stack([b["next_tok"]["task_feats"] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        ntm = torch.tensor(
            np.stack([b["next_tok"]["task_mask"] for b in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        naf = torch.tensor(
            np.stack([b["next_tok"]["agent_feats"] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        nam = torch.tensor(
            np.stack([b["next_tok"]["agent_mask"] for b in batch]),
            dtype=torch.bool,
            device=self.device,
        )

        logits, values = self.net(tf, tm, af, am)
        with torch.no_grad():
            _, next_values = self.target(ntf, ntm, naf, nam)
            target_v = rewards + self.gamma * next_values * (1.0 - dones)
            advantage = (target_v - values).detach()
            advantage = advantage.clamp(-5.0, 5.0)

        # Gaussian log-prob of exploration noise around current logits, on selected edges
        std = max(self.explore_std * 0.5, 0.05)
        # log N(noise; 0, std) for selected edges — reinforce directions that were taken
        # Use: log π(a|s) ≈ -||noise_selected||^2 / (2σ^2) under reparameterized exploration
        sel_count = selected.sum(dim=(1, 2)).clamp(min=1.0)
        log_prob = -0.5 * ((noise / std) ** 2) * selected
        log_prob = log_prob.sum(dim=(1, 2)) / sel_count

        # Also nudge logits of selected edges upward when advantage > 0 via soft scores
        scores = torch.sigmoid(logits)
        selected_score = (scores * selected).sum(dim=(1, 2)) / sel_count
        policy_term = log_prob * advantage + 0.5 * selected_score * advantage

        entropy = -(
            scores.clamp(1e-6, 1 - 1e-6) * torch.log(scores.clamp(1e-6, 1 - 1e-6))
        )
        edge_valid = stack_tok("edge_valid") if "edge_valid" in batch[0]["tok"] else (~am).unsqueeze(2) * (~tm).unsqueeze(1)
        entropy = (entropy * edge_valid * (~am).unsqueeze(2) * (~tm).unsqueeze(1)).sum(
            dim=(1, 2)
        ) / edge_valid.sum(dim=(1, 2)).clamp(min=1.0)

        value_loss = F.mse_loss(values, target_v)
        loss = -policy_term.mean() + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()
        self.n_updates += 1
        if self.n_updates % 20 == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "use_attention": self.use_attention,
                "max_tasks": self.max_tasks,
                "max_agents": self.max_agents,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "n_layers": self.n_layers,
                "lr": self.lr,
                "version": 2,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Rebuild net if architecture differs
        max_tasks = int(ckpt.get("max_tasks", self.max_tasks))
        max_agents = int(ckpt.get("max_agents", self.max_agents))
        use_attention = bool(ckpt.get("use_attention", self.use_attention))
        d_model = int(ckpt.get("d_model", self.d_model))
        nhead = int(ckpt.get("nhead", self.nhead))
        n_layers = int(ckpt.get("n_layers", self.n_layers))
        version = int(ckpt.get("version", 1))

        if (
            max_tasks != self.max_tasks
            or max_agents != self.max_agents
            or use_attention != self.use_attention
            or d_model != self.d_model
            or n_layers != self.n_layers
            or version < 2
        ):
            self.max_tasks = max_tasks
            self.max_agents = max_agents
            self.use_attention = use_attention
            self.d_model = d_model
            self.nhead = nhead
            self.n_layers = n_layers
            if use_attention and version >= 2:
                self.net = AttCoalitionNet(
                    max_tasks, max_agents, d_model=d_model, nhead=nhead, n_layers=n_layers
                ).to(self.device)
                self.target = AttCoalitionNet(
                    max_tasks, max_agents, d_model=d_model, nhead=nhead, n_layers=n_layers
                ).to(self.device)
            elif use_attention:
                # Old v1 checkpoint incompatible with v2 heads — load best-effort into new net fails
                # Rebuild and try load; if shapes mismatch, raise clear error
                self.net = AttCoalitionNet(
                    max_tasks, max_agents, d_model=d_model, nhead=nhead, n_layers=n_layers
                ).to(self.device)
                self.target = AttCoalitionNet(
                    max_tasks, max_agents, d_model=d_model, nhead=nhead, n_layers=n_layers
                ).to(self.device)
            else:
                self.net = MLPCoalitionNet(
                    max_tasks, max_agents, hidden=max(128, d_model * 2)
                ).to(self.device)
                self.target = MLPCoalitionNet(
                    max_tasks, max_agents, hidden=max(128, d_model * 2)
                ).to(self.device)
            self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        try:
            self.net.load_state_dict(ckpt["state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {path} is incompatible with Att-Coalition v2 "
                f"(version={version}). Retrain with train_escort.py."
            ) from exc
        self.target.load_state_dict(self.net.state_dict())
        self.net.eval()


class UrgencyCoalition:
    """Hand-crafted pair scores: urgency × capability fit × escort pressure."""

    def __init__(self):
        self.n_replans = 0

    def plan(self, env, hung, events=None, force: bool = True):
        open_tasks = _open_tasks_residual(env)
        live = env.get_live_agents()
        max_coord = float(getattr(env, "max_coord", 1000.0) or 1000.0)
        edge: Dict[Tuple[str, int], float] = {}
        for agent in live:
            for task in open_tasks:
                eligible = getattr(task, "eligible_agent_types", None)
                if eligible is not None and agent.type not in set(
                    eligible if not isinstance(eligible, str) else {eligible}
                ):
                    continue
                urg = _urgency(task, env.time_steps)
                pressure, _, _ = _threat_stats(env, task, max_coord)
                is_escort = 1.0 if getattr(task, "kind", None) == "Escort" else 0.0
                cap = (
                    float(agent.currentCap2Task[task.typeIdx])
                    if agent.currentCap2Task[task.typeIdx] > 0
                    else 0.0
                )
                dist = float(np.linalg.norm(agent.position - task.position)) / max_coord
                score = (
                    0.45 * urg
                    + 0.35 * pressure * (0.5 + 0.5 * is_escort)
                    + 0.3 * min(cap, 1.0)
                    - 0.25 * dist
                )
                if agent.type.startswith("F") and (is_escort or task.type == "Int"):
                    score += 0.2
                if agent.type.startswith("R") and task.type == "Rec":
                    score += 0.2
                edge[(agent.name, task.id)] = float(np.clip(score, 0.0, 1.0))

        reserved = committed_names(env)
        result = hung.allocate_tasks(
            live,
            open_tasks,
            time_step=env.time_steps,
            events=events,
            force=force,
            reserved_agent_names=reserved,
            agent_known_ids=env.agent_visibility_map(),
            edge_scores=edge,
        )
        self.n_replans = hung.n_replans
        assigned = [name for name, task in result if getattr(task, "id", 0) != 0]
        apply_agent_commits(env, assigned, int(getattr(env, "commit_horizon", 0) or 0))
        return result
