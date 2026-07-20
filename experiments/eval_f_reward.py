"""Lightweight F_Reward rollout used for checkpoint selection during training."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tianshou.data import Batch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.paper_scenarios import CASE_SPECS, TBTA_E3_FLAGS
from experiments.paper_eval import make_config, run_episode
from mUAV_TA.DroneEnv import MultiUAVEnv
from TaskAllocation.RL_Policies.Tianshou_Policy import _get_model


def eval_f_reward_tbta(
    policy,
    case_id: str = "D3_combined",
    episodes: int = 5,
    env_flags: Optional[Dict[str, Any]] = None,
) -> float:
    """Mean F_Reward for a loaded Tianshou DQN policy on a paper case."""
    flags = dict(env_flags or TBTA_E3_FLAGS)
    scores: List[float] = []
    for ep in range(episodes):
        out = run_episode("TBTA", case_id, ep, flags, policy=policy)
        scores.append(float(out["F_Reward"]))
    return float(np.mean(scores)) if scores else float("nan")


def load_tbta_policy(policy_path: str, case_id: str = "D3_combined"):
    cfg = make_config(CASE_SPECS[case_id], TBTA_E3_FLAGS)
    env = MultiUAVEnv(cfg)
    env.reset(seed=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = _get_model(model="CustomNetMultiHead", env=env, seed=0, algorithm="DQN")
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    policy.set_eps(0.0)
    return policy
