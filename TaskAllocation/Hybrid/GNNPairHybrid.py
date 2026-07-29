"""
GNN-ContextPair: bipartite message passing on visibility-valid edges → Local-Hungarian.

Matched capacity vs Att/MLP ContextPair; pure PyTorch (no torch_geometric).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from TaskAllocation.Hybrid.AttentionRAH import AGENT_FEAT_DIM, TASK_FEAT_DIM
from TaskAllocation.Hybrid.ContextPairHybrid import (
    CONTEXT_DIM,
    ContextPairHybrid,
    context_dim,
)
from TaskAllocation.Hybrid.PairCostHybrid import DEFAULT_MAX_AGENTS, DEFAULT_MAX_TASKS


class BipartiteMPLayer(nn.Module):
    """One a<->t message-passing step restricted to edge_valid."""

    def __init__(self, d_model: int = 64, msg_hidden: int = 96):
        super().__init__()
        self.msg_a2t = nn.Sequential(
            nn.Linear(d_model * 2, msg_hidden),
            nn.ReLU(),
            nn.Linear(msg_hidden, d_model),
        )
        self.msg_t2a = nn.Sequential(
            nn.Linear(d_model * 2, msg_hidden),
            nn.ReLU(),
            nn.Linear(msg_hidden, d_model),
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

    def forward(self, a_h: torch.Tensor, t_h: torch.Tensor, edge_valid: torch.Tensor):
        # a_h: [B,A,d], t_h: [B,T,d], edge_valid: [B,A,T] in {0,1}
        a_exp = a_h.unsqueeze(2).expand(-1, -1, t_h.size(1), -1)
        t_exp = t_h.unsqueeze(1).expand(-1, a_h.size(1), -1, -1)
        pair = torch.cat([a_exp, t_exp], dim=-1)
        w = edge_valid.unsqueeze(-1)

        msg_t = self.msg_a2t(pair) * w
        t_agg = msg_t.sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)
        t_h = self.norm_t(t_h + t_agg)

        msg_a = self.msg_t2a(pair) * w
        a_agg = msg_a.sum(dim=2) / w.sum(dim=2).clamp(min=1e-6)
        a_h = self.norm_a(a_h + a_agg)
        return a_h, t_h


class GNNContextPairNet(nn.Module):
    """Bipartite GNN edge scorer with context-biased pair head."""

    def __init__(
        self,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        d_model: int = 64,
        n_layers: int = 2,
        task_feat_dim: int = TASK_FEAT_DIM,
        agent_feat_dim: int = AGENT_FEAT_DIM,
        context_dim: int = CONTEXT_DIM,
        **_kwargs,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.d_model = d_model
        self.task_proj = nn.Linear(task_feat_dim, d_model)
        self.agent_proj = nn.Linear(agent_feat_dim, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.layers = nn.ModuleList([BipartiteMPLayer(d_model) for _ in range(max(1, n_layers))])
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

    def forward(self, task_feats, task_mask, agent_feats, agent_mask, context, edge_valid=None):
        a_h = self.agent_proj(agent_feats) + self.type_embed.weight[0]
        t_h = self.task_proj(task_feats) + self.type_embed.weight[1]
        if edge_valid is None:
            edge_valid = (~agent_mask).unsqueeze(2).float() * (~task_mask).unsqueeze(1).float()
        else:
            edge_valid = edge_valid.float()
            edge_valid = edge_valid * (~agent_mask).unsqueeze(2).float() * (~task_mask).unsqueeze(1).float()

        for layer in self.layers:
            a_h, t_h = layer(a_h, t_h, edge_valid)

        am = (~agent_mask).float().unsqueeze(-1)
        tm = (~task_mask).float().unsqueeze(-1)
        a_pool = (a_h * am).sum(1) / am.sum(1).clamp(min=1.0)
        t_pool = (t_h * tm).sum(1) / tm.sum(1).clamp(min=1.0)
        pooled = 0.5 * (a_pool + t_pool)
        ctx = self.ctx_proj(context) + pooled
        ctx_exp = ctx.unsqueeze(1).unsqueeze(2).expand(-1, self.max_agents, self.max_tasks, -1)
        a_exp = a_h.unsqueeze(2).expand(-1, -1, self.max_tasks, -1)
        t_exp = t_h.unsqueeze(1).expand(-1, self.max_agents, -1, -1)
        pair = torch.cat([a_exp, t_exp, a_exp * t_exp, ctx_exp], dim=-1)
        logits = self.pair_head(pair).squeeze(-1)
        logits = logits.masked_fill(agent_mask.unsqueeze(2), -1e9)
        logits = logits.masked_fill(task_mask.unsqueeze(1), -1e9)
        logits = logits.masked_fill(edge_valid < 0.5, -1e9)
        value = self.value_head(torch.cat([pooled, ctx], dim=-1)).squeeze(-1)
        return logits, value


class GNNContextPairHybrid(ContextPairHybrid):
    """GNN-ContextPair policy: bipartite MP → edge scores → Local-Hungarian."""

    def __init__(self, use_attention: bool = False, **kwargs):
        kwargs = dict(kwargs)
        kwargs["use_attention"] = False
        # Parent builds MLPContextPair nets; we replace with GNN immediately after.
        super().__init__(**kwargs)
        net_kwargs = dict(
            max_tasks=self.max_tasks,
            max_agents=self.max_agents,
            d_model=self.d_model,
            n_layers=self.n_layers,
            task_feat_dim=self.task_feat_dim,
            agent_feat_dim=self.agent_feat_dim,
            context_dim=context_dim(self.raw_features),
        )
        self.net = GNNContextPairNet(**net_kwargs).to(self.device)
        self.target = GNNContextPairNet(**net_kwargs).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.kind = "GNNContextPair"
        self.use_gnn = True

    def _tensors(self, tok: dict) -> Tuple[torch.Tensor, ...]:
        tf, tm, af, am, ctx = super()._tensors(tok)
        ev = torch.tensor(tok["edge_valid"], dtype=torch.float32, device=self.device).unsqueeze(0)
        return tf, tm, af, am, ctx, ev

    def _batch_tensors(self, toks):
        tf, tm, af, am, ctx = super()._batch_tensors(toks)
        ev = torch.tensor(
            np.stack([t["edge_valid"] for t in toks]), dtype=torch.float32, device=self.device
        )
        return tf, tm, af, am, ctx, ev

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
        nev = torch.tensor(
            np.stack([b["next_tok"]["edge_valid"] for b in batch]), dtype=torch.float32, device=self.device
        )

        logits, values = self.net(tf, tm, af, am, ctx, edge_valid)
        with torch.no_grad():
            _, next_values = self.target(ntf, ntm, naf, nam, nctx, nev)
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
                "use_attention": False,
                "use_gnn": True,
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
        self.__init__(
            use_attention=False,
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
