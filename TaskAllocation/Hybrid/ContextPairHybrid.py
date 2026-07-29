"""
ContextPair: Att vs MLP context encoders → visibility-masked edge scores → Local-Hungarian.

Matched controls for demonstrating attention advantage on WPS_attn.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from TaskAllocation.Hybrid.AttentionRAH import AGENT_FEAT_DIM, TASK_FEAT_DIM, _urgency
from TaskAllocation.Hybrid.PairCostHybrid import (
    DEFAULT_MAX_AGENTS,
    DEFAULT_MAX_TASKS,
    SCORE_CLAMP,
    PairCostHybrid,
    build_pair_tokens,
)

CONTEXT_DIM = 8
# Raw mode keeps only mission clock: every other summary entry is a hand-computed
# aggregate (urgent count, front split, imbalance) that the encoder should infer itself.
RAW_CONTEXT_DIM = 1


def context_dim(raw: bool = False) -> int:
    return RAW_CONTEXT_DIM if raw else CONTEXT_DIM


def build_context_summary(env, tok: dict, raw: bool = False) -> np.ndarray:
    """Cheap team/situation vector (same for Att and MLP)."""
    if raw:
        return np.asarray(
            [float(env.time_steps) / max(getattr(env, "max_time_steps", 150), 1)], dtype=np.float32
        )
    max_coord = float(getattr(env, "max_coord", 1000.0) or 1000.0)
    mid_x = float(getattr(env, "area_width", max_coord)) * 0.5
    live = tok["live"]
    tasks = tok["open_tasks"]
    n_agents = max(len(live), 1)
    n_tasks = max(len(tasks), 1)
    n_urgent = 0
    left = right = 0
    for t in tasks:
        if _urgency(t, env.time_steps) >= (1.0 - 12.0 / 40.0) and getattr(t, "hard_deadline", None) is not None:
            n_urgent += 1
        if float(t.position[0]) < mid_x:
            left += 1
        else:
            right += 1
    free = sum(1 for a in live if (not a.tasks) or a.tasks[0].id == 0)
    fighters = sum(1 for a in live if str(getattr(a, "type", "")).startswith("F"))
    imbalance = abs(left - right) / n_tasks
    ctx = np.asarray(
        [
            n_urgent / n_tasks,
            min(len(tasks) / float(n_agents), 4.0) / 4.0,
            free / n_agents,
            fighters / n_agents,
            left / n_tasks,
            right / n_tasks,
            imbalance,
            float(env.time_steps) / max(getattr(env, "max_time_steps", 150), 1),
        ],
        dtype=np.float32,
    )
    return ctx


def build_context_pair_tokens(
    env, max_tasks=DEFAULT_MAX_TASKS, max_agents=DEFAULT_MAX_AGENTS, raw: bool = False
) -> dict:
    tok = build_pair_tokens(env, max_tasks=max_tasks, max_agents=max_agents, raw=raw)
    tok["context"] = build_context_summary(env, tok, raw=raw)
    return tok


class AttContextPairNet(nn.Module):
    """Self/cross-attn pair scores with explicit pooled + summary context bias."""

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
        context_dim: int = CONTEXT_DIM,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.d_model = d_model
        self.task_proj = nn.Linear(task_feat_dim, d_model)
        self.agent_proj = nn.Linear(agent_feat_dim, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
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
        # pair + context: 3*d + d = 4d
        self.pair_head = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, task_feats, task_mask, agent_feats, agent_mask, context):
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
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        ctx = self.ctx_proj(context) + pooled
        ctx_exp = ctx.unsqueeze(1).unsqueeze(2).expand(-1, self.max_agents, self.max_tasks, -1)
        a_exp = a_h.unsqueeze(2).expand(-1, -1, self.max_tasks, -1)
        t_exp = t_h.unsqueeze(1).expand(-1, self.max_agents, -1, -1)
        pair = torch.cat([a_exp, t_exp, a_exp * t_exp, ctx_exp], dim=-1)
        logits = self.pair_head(pair).squeeze(-1)
        logits = logits.masked_fill(agent_mask.unsqueeze(2), -1e9)
        logits = logits.masked_fill(task_mask.unsqueeze(1), -1e9)
        value = self.value_head(torch.cat([pooled, ctx], dim=-1)).squeeze(-1)
        return logits, value


class MLPContextPairNet(nn.Module):
    """Matched MLP control: mean-pooled tokens + summary context into pair MLP."""

    def __init__(
        self,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        hidden: int = 192,
        d_model: int = 64,
        task_feat_dim: int = TASK_FEAT_DIM,
        agent_feat_dim: int = AGENT_FEAT_DIM,
        context_dim: int = CONTEXT_DIM,
        **_kwargs,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        # capacity roughly matched to AttContextPairNet
        in_pair = task_feat_dim + agent_feat_dim + task_feat_dim + agent_feat_dim + context_dim
        self.ctx_mlp = nn.Sequential(
            nn.Linear(task_feat_dim + agent_feat_dim + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_pair, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, task_feats, task_mask, agent_feats, agent_mask, context):
        am = (~agent_mask).float().unsqueeze(-1)
        tm = (~task_mask).float().unsqueeze(-1)
        a_pool = (agent_feats * am).sum(1) / am.sum(1).clamp(min=1.0)
        t_pool = (task_feats * tm).sum(1) / tm.sum(1).clamp(min=1.0)
        ctx_h = self.ctx_mlp(torch.cat([a_pool, t_pool, context], dim=-1))
        b, a, _ = agent_feats.shape
        t = task_feats.size(1)
        a_exp = agent_feats.unsqueeze(2).expand(-1, -1, t, -1)
        t_exp = task_feats.unsqueeze(1).expand(-1, a, -1, -1)
        a_p = a_pool.unsqueeze(1).unsqueeze(2).expand(-1, a, t, -1)
        t_p = t_pool.unsqueeze(1).unsqueeze(2).expand(-1, a, t, -1)
        c_exp = context.unsqueeze(1).unsqueeze(2).expand(-1, a, t, -1)
        pair = torch.cat([a_exp, t_exp, a_p, t_p, c_exp], dim=-1)
        logits = self.pair_mlp(pair).squeeze(-1)
        logits = logits.masked_fill(agent_mask.unsqueeze(2), -1e9)
        logits = logits.masked_fill(task_mask.unsqueeze(1), -1e9)
        value = self.value_mlp(ctx_h).squeeze(-1)
        return logits, value


class ContextPairHybrid(PairCostHybrid):
    """Att-ContextPair / MLP-ContextPair policy."""

    def __init__(self, use_attention: bool = True, **kwargs):
        # Build parent then replace nets (parent builds AttPair/MLPPair)
        device = kwargs.get("device")
        kwargs = dict(kwargs)
        kwargs["use_attention"] = use_attention
        super().__init__(**kwargs)
        Net = AttContextPairNet if use_attention else MLPContextPairNet
        net_kwargs = dict(
            max_tasks=self.max_tasks,
            max_agents=self.max_agents,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers,
            task_feat_dim=self.task_feat_dim,
            agent_feat_dim=self.agent_feat_dim,
            context_dim=context_dim(self.raw_features),
        )
        self.net = Net(**net_kwargs).to(self.device)
        self.target = Net(**net_kwargs).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.kind = "AttContextPair" if use_attention else "MLPContextPair"

    def build_tokens(self, env) -> dict:
        return build_context_pair_tokens(env, self.max_tasks, self.max_agents, raw=self.raw_features)

    def _tensors(self, tok: dict) -> Tuple[torch.Tensor, ...]:
        tf = torch.tensor(tok["task_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
        tm = torch.tensor(tok["task_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        af = torch.tensor(tok["agent_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
        am = torch.tensor(tok["agent_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        ctx = torch.tensor(tok["context"], dtype=torch.float32, device=self.device).unsqueeze(0)
        return tf, tm, af, am, ctx

    def act(self, tok: dict, explore: bool = False):
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(*self._tensors(tok))
            logits_np = logits[0].cpu().numpy()
        edge_valid = tok["edge_valid"]
        noise = np.zeros_like(logits_np, dtype=np.float32)
        if explore:
            noise = (np.random.randn(*logits_np.shape) * self.explore_std).astype(np.float32) * edge_valid
        scores = np.tanh(logits_np + noise) * self.score_clamp * edge_valid
        return scores.astype(np.float32), noise, logits_np.astype(np.float32)

    IL_KEYS = PairCostHybrid.IL_KEYS + ("context",)

    def _batch_tensors(self, toks):
        tf, tm, af, am = super()._batch_tensors(toks)
        ctx = torch.tensor(
            np.stack([t["context"] for t in toks]), dtype=torch.float32, device=self.device
        )
        return tf, tm, af, am, ctx

    def push(self, tok, scores, noise, logits, selected, reward, next_tok, done):
        keep = ("task_feats", "task_mask", "agent_feats", "agent_mask", "edge_valid", "context")
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
        import torch.nn.functional as F

        if len(self.buffer) < min(batch_size, 16):
            return None
        bs = min(batch_size, len(self.buffer))
        self.net.train()
        idx = np.random.choice(len(self.buffer), size=bs, replace=False)
        batch = [self.buffer[i] for i in idx]

        def stack_tok(key, dtype=torch.float32):
            return torch.tensor(np.stack([b["tok"][key] for b in batch]), dtype=dtype, device=self.device)

        tf, tm = stack_tok("task_feats"), stack_tok("task_mask", torch.bool).bool()
        af, am = stack_tok("agent_feats"), stack_tok("agent_mask", torch.bool).bool()
        ctx = stack_tok("context")
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
        nctx = torch.tensor(
            np.stack([b["next_tok"]["context"] for b in batch]), dtype=torch.float32, device=self.device
        )

        logits, values = self.net(tf, tm, af, am, ctx)
        with torch.no_grad():
            _, next_values = self.target(ntf, ntm, naf, nam, nctx)
            target_v = rewards + self.gamma * next_values * (1.0 - dones)
            advantage = (target_v - values).detach().clamp(-5.0, 5.0)

        std = max(self.explore_std * 0.5, 0.05)
        sel_count = selected.sum(dim=(1, 2)).clamp(min=1.0)
        log_prob = (-0.5 * ((noise / std) ** 2) * selected).sum(dim=(1, 2)) / sel_count
        scores = torch.sigmoid(logits.clamp(-8, 8))
        selected_score = (scores * selected).sum(dim=(1, 2)) / sel_count
        policy_term = log_prob * advantage + 0.5 * selected_score * advantage
        entropy = -(scores.clamp(1e-6, 1 - 1e-6) * torch.log(scores.clamp(1e-6, 1 - 1e-6)))
        entropy = (entropy * edge_valid).sum(dim=(1, 2)) / edge_valid.sum(dim=(1, 2)).clamp(min=1.0)
        loss = -policy_term.mean() + self.value_coef * F.mse_loss(values, target_v) - self.entropy_coef * entropy.mean()
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
                "raw_features": self.raw_features,
                "kind": self.kind,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        use_attention = bool(ckpt.get("use_attention", self.use_attention))
        self.__init__(
            use_attention=use_attention,
            max_tasks=int(ckpt.get("max_tasks", self.max_tasks)),
            max_agents=int(ckpt.get("max_agents", self.max_agents)),
            d_model=int(ckpt.get("d_model", self.d_model)),
            nhead=int(ckpt.get("nhead", self.nhead)),
            n_layers=int(ckpt.get("n_layers", self.n_layers)),
            lr=float(ckpt.get("lr", self.lr)),
            device=self.device,
            score_clamp=float(ckpt.get("score_clamp", self.score_clamp)),
            raw_features=bool(ckpt.get("raw_features", False)),
        )
        self.net.load_state_dict(ckpt["state_dict"])
        self.target.load_state_dict(ckpt["state_dict"])
        self.net.eval()
