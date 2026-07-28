"""
WPS pair-cost hybrid: visibility-masked agent–task edge scores → Local-Hungarian.

Att-Pair: self-attn + cross-attn pair head (escort-style).
MLP-Pair: flat per-edge MLP on concatenated features (matched control).
Urgency-Pair: engineered edge scores (no learning).
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

DEFAULT_MAX_TASKS = 32
DEFAULT_MAX_AGENTS = 16
SCORE_CLAMP = 0.35  # keep base dist/cap/urg dominant early


def build_pair_tokens(
    env, max_tasks: int = DEFAULT_MAX_TASKS, max_agents: int = DEFAULT_MAX_AGENTS
) -> dict:
    """WPS tokens (Att-RAH features) plus bipartite edge_valid visibility mask."""
    tok = build_att_tokens(env, max_tasks=max_tasks, max_agents=max_agents)
    live = tok["live"]
    kept = tok["open_tasks"][:max_tasks]
    vis = tok["vis"]
    edge_valid = np.zeros((max_agents, max_tasks), dtype=np.float32)

    for i, a in enumerate(live[:max_agents]):
        if tok["agent_mask"][i]:
            continue
        known_ids = None if vis is None else vis.get(a.name, set())
        atype = getattr(a, "type", "")
        caps = getattr(a, "currentCap2Task", None)
        for j, t in enumerate(kept):
            if tok["task_mask"][j]:
                continue
            if known_ids is not None and t.id not in known_ids:
                continue
            eligible = getattr(t, "eligible_agent_types", None)
            if eligible is not None:
                elig = {eligible} if isinstance(eligible, str) else set(eligible)
                if atype not in elig:
                    continue
            if caps is not None and len(caps) > getattr(t, "typeIdx", 0):
                if float(caps[t.typeIdx]) <= 0:
                    continue
            edge_valid[i, j] = 1.0

    tok["edge_valid"] = edge_valid
    tok["open_tasks"] = kept
    tok["task_ids"] = [t.id for t in kept]
    return tok


def urgency_edge_scores(env, tok: dict) -> np.ndarray:
    """Hand-crafted edge residual: urgency + scarcity − normalized distance."""
    max_coord = float(getattr(env, "max_coord", 1000.0) or 1000.0)
    live = tok["live"]
    tasks = tok["open_tasks"]
    vis = tok["vis"]
    n_agents = max(len(live), 1)
    scores = np.zeros((tok["agent_feats"].shape[0], tok["task_feats"].shape[0]), dtype=np.float32)
    for i, a in enumerate(live[: scores.shape[0]]):
        if tok["agent_mask"][i]:
            continue
        for j, t in enumerate(tasks):
            if tok["task_mask"][j] or tok["edge_valid"][i, j] < 0.5:
                continue
            urg = _urgency(t, env.time_steps)
            scar = _scarcity(t, vis, n_agents)
            dist = float(np.linalg.norm(a.position - t.position)) / max(max_coord, 1.0)
            scores[i, j] = float(np.clip(0.5 * urg + 0.3 * scar - 0.4 * dist, -SCORE_CLAMP, SCORE_CLAMP))
    return scores


class AttPairNet(nn.Module):
    def __init__(
        self,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        task_feat_dim: int = TASK_FEAT_DIM,
        agent_feat_dim: int = AGENT_FEAT_DIM,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.task_proj = nn.Linear(task_feat_dim, d_model)
        self.agent_proj = nn.Linear(agent_feat_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=dropout,
        )
        self.self_encoder = nn.TransformerEncoder(layer, num_layers=max(1, n_layers - 1))
        self.cross_a2t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_t2a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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
        t_emb = self.task_proj(task_feats) + self.type_embed.weight[1]
        a_emb = self.agent_proj(agent_feats) + self.type_embed.weight[0]
        tokens = torch.cat([a_emb, t_emb], dim=1)
        pad_mask = torch.cat([agent_mask, task_mask], dim=1)
        h = self.self_encoder(tokens, src_key_padding_mask=pad_mask)
        a_h = h[:, : self.max_agents, :]
        t_h = h[:, self.max_agents :, :]
        a_ctx, _ = self.cross_a2t(a_h, t_h, t_h, key_padding_mask=task_mask, need_weights=False)
        t_ctx, _ = self.cross_t2a(t_h, a_h, a_h, key_padding_mask=agent_mask, need_weights=False)
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


class MLPPairNet(nn.Module):
    def __init__(
        self,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        hidden: int = 128,
        d_model: int = 64,
        task_feat_dim: int = TASK_FEAT_DIM,
        agent_feat_dim: int = AGENT_FEAT_DIM,
        **_kwargs,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        in_dim = task_feat_dim + agent_feat_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
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
        # mean of valid agent/task concat as crude global value features
        am = (~agent_mask).float().unsqueeze(-1)
        tm = (~task_mask).float().unsqueeze(-1)
        a_pool = (agent_feats * am).sum(1) / am.sum(1).clamp(min=1.0)
        t_pool = (task_feats * tm).sum(1) / tm.sum(1).clamp(min=1.0)
        value = self.value_mlp(torch.cat([a_pool, t_pool], dim=-1)).squeeze(-1)
        return logits, value


class PairCostHybrid:
    """Learned or engineered pair scores for Local-Hungarian on WPS."""

    def __init__(
        self,
        use_attention: bool = True,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: Optional[str] = None,
        score_clamp: float = SCORE_CLAMP,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_attention = use_attention
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        self.lr = lr
        self.gamma = gamma
        self.score_clamp = score_clamp
        self.explore_std = 0.15
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.eps = 0.0
        self.n_replans = 0
        self.n_updates = 0
        self.buffer: List[dict] = []
        self.max_buffer = 40_000
        Net = AttPairNet if use_attention else MLPPairNet
        self.net = Net(
            max_tasks=max_tasks,
            max_agents=max_agents,
            d_model=d_model,
            nhead=nhead,
            n_layers=n_layers,
        ).to(self.device)
        self.target = Net(
            max_tasks=max_tasks,
            max_agents=max_agents,
            d_model=d_model,
            nhead=nhead,
            n_layers=n_layers,
        ).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

    def build_tokens(self, env) -> dict:
        return build_pair_tokens(env, self.max_tasks, self.max_agents)

    def _tensors(self, tok: dict):
        tf = torch.tensor(tok["task_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
        tm = torch.tensor(tok["task_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        af = torch.tensor(tok["agent_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
        am = torch.tensor(tok["agent_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        return tf, tm, af, am

    def act(self, tok: dict, explore: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(*self._tensors(tok))
            logits_np = logits[0].cpu().numpy()
        edge_valid = tok["edge_valid"]
        noise = np.zeros_like(logits_np, dtype=np.float32)
        if explore:
            noise = (np.random.randn(*logits_np.shape) * self.explore_std).astype(np.float32)
            noise = noise * edge_valid
        scores = np.tanh(logits_np + noise) * self.score_clamp
        scores = scores * edge_valid
        return scores.astype(np.float32), noise, logits_np.astype(np.float32)

    def edge_score_dict(self, tok: dict, scores: np.ndarray) -> Dict[Tuple[str, int], float]:
        out: Dict[Tuple[str, int], float] = {}
        live = tok["live"]
        task_ids = tok["task_ids"]
        for i, agent in enumerate(live[: self.max_agents]):
            if tok["agent_mask"][i]:
                continue
            for j, tid in enumerate(task_ids):
                if tok["task_mask"][j] or tok["edge_valid"][i, j] < 0.5:
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

    def plan(self, env, hung, events=None, explore: bool = False, force: bool = True, scores=None):
        tok = self.build_tokens(env)
        if scores is None:
            scores, noise, logits = self.act(tok, explore=explore)
        else:
            noise = np.zeros_like(scores)
            logits = scores / max(self.score_clamp, 1e-6)
        edge = self.edge_score_dict(tok, scores)
        result = hung.allocate_tasks(
            env.get_live_agents(),
            tok["open_tasks"],
            time_step=env.time_steps,
            events=events,
            force=force,
            agent_known_ids=tok["vis"],
            edge_scores=edge,
        )
        if result:
            self.n_replans += 1
        selected = self._selected_mask(tok, result)
        return result, tok, scores, noise, logits, selected

    def imitation_loss(self, tok: dict, expert_mask: np.ndarray) -> float:
        """BCE on visible edges: push up expert assignments, down other valid edges."""
        self.net.train()
        tf, tm, af, am = self._tensors(tok)
        logits, _ = self.net(tf, tm, af, am)
        edge_valid = torch.tensor(tok["edge_valid"], dtype=torch.float32, device=self.device).unsqueeze(0)
        target = torch.tensor(expert_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        # only supervise valid edges
        logits = logits.clamp(-8.0, 8.0)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        denom = edge_valid.sum().clamp(min=1.0)
        # weight positives higher (sparse expert edges)
        pos = (target * edge_valid).sum().clamp(min=1.0)
        neg = ((1.0 - target) * edge_valid).sum().clamp(min=1.0)
        w = edge_valid * (target * (neg / pos) + (1.0 - target))
        loss = (bce * w).sum() / denom
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optim.step()
        self.n_updates += 1
        return float(loss.item())

    def push(self, tok, scores, noise, logits, selected, reward, next_tok, done):
        keep = ("task_feats", "task_mask", "agent_feats", "agent_mask", "edge_valid")
        self.buffer.append(
            {
                "tok": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tok.items() if k in keep},
                "scores": np.asarray(scores, dtype=np.float32),
                "noise": np.asarray(noise, dtype=np.float32),
                "logits": np.asarray(logits, dtype=np.float32),
                "selected": np.asarray(selected, dtype=np.float32),
                "reward": float(reward),
                "next_tok": {
                    k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in next_tok.items() if k in keep
                },
                "done": bool(done),
            }
        )
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer :]

    def update(self, batch_size: int = 64) -> Optional[float]:
        if len(self.buffer) < min(batch_size, 16):
            return None
        bs = min(batch_size, len(self.buffer))
        self.net.train()
        idx = np.random.choice(len(self.buffer), size=bs, replace=False)
        batch = [self.buffer[i] for i in idx]

        def stack_tok(key, dtype=torch.float32):
            return torch.tensor(np.stack([b["tok"][key] for b in batch]), dtype=dtype, device=self.device)

        tf = stack_tok("task_feats")
        tm = stack_tok("task_mask", torch.bool).bool()
        af = stack_tok("agent_feats")
        am = stack_tok("agent_mask", torch.bool).bool()
        edge_valid = stack_tok("edge_valid")
        selected = torch.tensor(np.stack([b["selected"] for b in batch]), dtype=torch.float32, device=self.device)
        noise = torch.tensor(np.stack([b["noise"] for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device)
        ntf = torch.tensor(
            np.stack([b["next_tok"]["task_feats"] for b in batch]), dtype=torch.float32, device=self.device
        )
        ntm = torch.tensor(
            np.stack([b["next_tok"]["task_mask"] for b in batch]), dtype=torch.bool, device=self.device
        )
        naf = torch.tensor(
            np.stack([b["next_tok"]["agent_feats"] for b in batch]), dtype=torch.float32, device=self.device
        )
        nam = torch.tensor(
            np.stack([b["next_tok"]["agent_mask"] for b in batch]), dtype=torch.bool, device=self.device
        )

        logits, values = self.net(tf, tm, af, am)
        with torch.no_grad():
            _, next_values = self.target(ntf, ntm, naf, nam)
            target_v = rewards + self.gamma * next_values * (1.0 - dones)
            advantage = (target_v - values).detach().clamp(-5.0, 5.0)

        std = max(self.explore_std * 0.5, 0.05)
        sel_count = selected.sum(dim=(1, 2)).clamp(min=1.0)
        log_prob = -0.5 * ((noise / std) ** 2) * selected
        log_prob = log_prob.sum(dim=(1, 2)) / sel_count
        scores = torch.sigmoid(logits.clamp(-8, 8))
        selected_score = (scores * selected).sum(dim=(1, 2)) / sel_count
        policy_term = log_prob * advantage + 0.5 * selected_score * advantage
        entropy = -(scores.clamp(1e-6, 1 - 1e-6) * torch.log(scores.clamp(1e-6, 1 - 1e-6)))
        entropy = (entropy * edge_valid).sum(dim=(1, 2)) / edge_valid.sum(dim=(1, 2)).clamp(min=1.0)
        value_loss = F.mse_loss(values, target_v)
        loss = -policy_term.mean() + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
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
                "score_clamp": self.score_clamp,
                "kind": "PairCostHybrid",
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        use_attention = bool(ckpt.get("use_attention", self.use_attention))
        max_tasks = int(ckpt.get("max_tasks", self.max_tasks))
        max_agents = int(ckpt.get("max_agents", self.max_agents))
        d_model = int(ckpt.get("d_model", self.d_model))
        nhead = int(ckpt.get("nhead", self.nhead))
        n_layers = int(ckpt.get("n_layers", self.n_layers))
        if (
            use_attention != self.use_attention
            or max_tasks != self.max_tasks
            or max_agents != self.max_agents
            or d_model != self.d_model
        ):
            self.__init__(
                use_attention=use_attention,
                max_tasks=max_tasks,
                max_agents=max_agents,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                lr=float(ckpt.get("lr", self.lr)),
                device=self.device,
                score_clamp=float(ckpt.get("score_clamp", self.score_clamp)),
            )
        self.net.load_state_dict(ckpt["state_dict"])
        self.target.load_state_dict(ckpt["state_dict"])
        self.net.eval()


class UrgencyPair:
    """Non-learned urgency/scarcity/distance edge residuals → Local-Hungarian."""

    def __init__(self, max_tasks: int = DEFAULT_MAX_TASKS, max_agents: int = DEFAULT_MAX_AGENTS):
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.n_replans = 0

    def plan(self, env, hung, events=None, force: bool = True):
        tok = build_pair_tokens(env, self.max_tasks, self.max_agents)
        scores = urgency_edge_scores(env, tok)
        edge = {}
        for i, agent in enumerate(tok["live"][: self.max_agents]):
            if tok["agent_mask"][i]:
                continue
            for j, tid in enumerate(tok["task_ids"]):
                if tok["edge_valid"][i, j] < 0.5:
                    continue
                edge[(agent.name, int(tid))] = float(scores[i, j])
        result = hung.allocate_tasks(
            env.get_live_agents(),
            tok["open_tasks"],
            time_step=env.time_steps,
            events=events,
            force=force,
            agent_known_ids=tok["vis"],
            edge_scores=edge,
        )
        if result:
            self.n_replans += 1
        return result, tok, scores
