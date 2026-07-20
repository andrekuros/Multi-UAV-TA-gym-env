"""
Reproducible TBTA (DQN) training for RL_EXPERIMENT_PLAN.md experiments.

Usage:
  python experiments/train_tbta.py --exp E0
  python experiments/train_tbta.py --exp E1 --max-epoch 100
  python experiments/train_tbta.py --list
"""
from __future__ import annotations

import argparse
import copy
import datetime
import os
import random
import subprocess
import sys
from typing import Any, Dict, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from TaskAllocation.RL_Policies.Custom_Classes import CustomCollector, CustomParallelToAECWrapper
from TaskAllocation.RL_Policies.CustomClass_MultiHead_Transformer import CustomNetMultiHead
from tianshou.policy import DQNPolicy
from tianshou.data import Batch, to_numpy, to_torch_as
from mUAV_TA.DroneEnv import MultiUAVEnv
from mUAV_TA.MultiDroneEnvUtils import agentEnvOptions


class MaskAwareDQN(DQNPolicy):
    """DQN that respects obs.action_mask when choosing discrete actions."""

    def forward(self, batch, state=None, model="model", input="obs", **kwargs):
        model_fn = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model_fn(obs_next, state=state, info=batch.info)
        action_mask = getattr(obs, "legal_mask", None)
        if action_mask is None:
            action_mask = getattr(obs, "mask", None)
        if action_mask is not None:
            m = np.asarray(action_mask)
            if m.ndim == 1:
                m = m.reshape(1, -1)
            if m.shape[1] > logits.shape[1]:
                m = m[:, : logits.shape[1]]
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - m, logits) * min_value
        if not hasattr(self, "max_action_num"):
            self.max_action_num = logits.shape[1]
        act = to_numpy(logits.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=hidden)


DEFAULT_REWARD = {
    "action": 0.0,
    "distance": 1.0,
    "quality": 1.0,
    "s_quality": 1.0,
    "time": 0.0,
    "alloc": 0.0,
    "time_penaulty": 0.0,
    "step": 0.0,
}

# Scenario presets aligned with UCF eval (see main.py / RL_EXPERIMENT_PLAN.md)
SCENARIO_LEGACY = {"agents": {"F1": 0, "F2": 4, "R1": 0, "R2": 0}, "tasks": {"Att": 25, "Rec": 0, "Hold": 0}}
SCENARIO_UCF_NONE = {"agents": {"F1": 0, "F2": 2, "R1": 0, "R2": 0}, "tasks": {"Att": 15, "Rec": 0, "Hold": 0}}
SCENARIO_UCF_MIXED = {
    "agents": {"F1": 2, "F2": 0, "R1": 4, "R2": 0},
    "tasks": {"Att": 6, "Rec": 12, "Hold": 0},
}


def _rw(**overrides) -> Dict[str, float]:
    w = dict(DEFAULT_REWARD)
    w.update(overrides)
    return w


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "E0": {
        "desc": "Legacy baseline (F2x4 Att25, no env toggles)",
        "scenario": SCENARIO_LEGACY,
        "early_terminate": False,
        "capability_mask": False,
        "saturate_mask": False,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "E1": {
        "desc": "Train-eval match proxy (F1/R1 + Att/Rec, UCF-like)",
        "scenario": SCENARIO_UCF_MIXED,
        "early_terminate": False,
        "capability_mask": False,
        "saturate_mask": False,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "E1_none": {
        "desc": "Match scal_None (F2x2 Att15)",
        "scenario": SCENARIO_UCF_NONE,
        "early_terminate": False,
        "capability_mask": False,
        "saturate_mask": False,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "E2": {
        "desc": "E1 + early_terminate",
        "scenario": SCENARIO_UCF_MIXED,
        "early_terminate": True,
        "capability_mask": False,
        "saturate_mask": False,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "E3": {
        "desc": "E2 + capability/saturate action masks",
        "scenario": SCENARIO_UCF_MIXED,
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "E4c": {
        "desc": "E3 + modest time/alloc shaping",
        "scenario": SCENARIO_UCF_MIXED,
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "reward_weights": _rw(time=0.1, alloc=0.1, time_penaulty=0.25),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "E5a": {
        "desc": "E4c + faster epsilon decay (0.5 to 0.05)",
        "scenario": SCENARIO_UCF_MIXED,
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "reward_weights": _rw(time=0.1, alloc=0.1, time_penaulty=0.25),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.50,
            "tn_eps_min": 0.05,
            "ts_eps_max": 0.0,
            "eps_decay_epochs": 40,
            "prioritized": False,
        },
    },
    "E5b": {
        "desc": "E4c + n-step=8",
        "scenario": SCENARIO_UCF_MIXED,
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "reward_weights": _rw(time=0.1, alloc=0.1, time_penaulty=0.25),
        "dqn": {"discount_factor": 0.98, "estimation_step": 8, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "E5d": {
        "desc": "E4c + prioritized replay",
        "scenario": SCENARIO_UCF_MIXED,
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "reward_weights": _rw(time=0.1, alloc=0.1, time_penaulty=0.25),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 100,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 10,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": True,
        },
    },
    "D3": {
        "desc": "Dynamic D3 train (fail+threats+arrivals) with E3 masks",
        "scenario": {
            "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
            "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        },
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "fail_rate": 0.1,
        "threats_list": [("T1", 3), ("T2", 2)],
        "arrival_rate": 0.02,
        "include_time_windows": True,
        "dynamic_idle_penalty": 0.05,
        "reward_weights": _rw(time=0.1, alloc=0.1, time_penaulty=0.25),
        "dqn": {"discount_factor": 0.98, "estimation_step": 8, "target_update_freq": 3000, "lr": 1e-4},
        "trainer": {
            "max_epoch": 40,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 8,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.50,
            "tn_eps_min": 0.05,
            "eps_decay_epochs": 30,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "E3_MLP": {
        "desc": "E3 scenario with shared MLP (no attention) ablation",
        "scenario": SCENARIO_UCF_MIXED,
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "model": "SharedMLP",
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 150, "target_update_freq": 6000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 40,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 8,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.70,
            "tn_eps_min": 0.007,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "D3ft": {
        "desc": "Fine-tune E3 on D3 dynamics (no time-window obs change)",
        "scenario": {
            "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
            "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        },
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "fail_rate": 0.1,
        "threats_list": [("T1", 3), ("T2", 2)],
        "arrival_rate": 0.02,
        "include_time_windows": False,
        "dynamic_idle_penalty": 0.05,
        "init_policy": "dqn_Custom/policy_TBTA_E3_260715-131109.pth",
        "f_reward_eval_case": "D3_combined",
        "f_reward_eval_every": 5,
        "f_reward_eval_episodes": 5,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 8, "target_update_freq": 3000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 20,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 8,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.25,
            "tn_eps_min": 0.05,
            "eps_decay_epochs": 15,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "CurD1": {
        "desc": "Curriculum stage 1: fine-tune E3 on attrition only",
        "scenario": {
            "agents": {"F1": 2, "F2": 0, "R1": 4, "R2": 0},
            "tasks": {"Att": 6, "Rec": 12, "Hold": 0},
        },
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "fail_rate": 0.1,
        "threats_list": [],
        "arrival_rate": 0.0,
        "include_time_windows": False,
        "dynamic_idle_penalty": 0.05,
        "init_policy": "dqn_Custom/policy_TBTA_E3_260715-131109.pth",
        "f_reward_eval_case": "D1_attrition",
        "f_reward_eval_every": 5,
        "f_reward_eval_episodes": 5,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 8, "target_update_freq": 3000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 15,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 5,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.25,
            "tn_eps_min": 0.05,
            "eps_decay_epochs": 12,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "CurD2": {
        "desc": "Curriculum stage 2: fine-tune from CurD1 on pop-up threats",
        "scenario": {
            "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
            "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        },
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "fail_rate": 0.0,
        "threats_list": [("T1", 3), ("T2", 2)],
        "arrival_rate": 0.0,
        "include_time_windows": False,
        "dynamic_idle_penalty": 0.05,
        "init_policy": "dqn_Custom/policy_TBTA_CurD1_latest.pth",
        "f_reward_eval_case": "D2_popup_threats",
        "f_reward_eval_every": 5,
        "f_reward_eval_episodes": 5,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 8, "target_update_freq": 3000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 15,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 5,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.20,
            "tn_eps_min": 0.05,
            "eps_decay_epochs": 12,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
    "CurD3": {
        "desc": "Curriculum stage 3: fine-tune from CurD2 on combined dynamics",
        "scenario": {
            "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
            "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        },
        "early_terminate": True,
        "capability_mask": True,
        "saturate_mask": True,
        "fail_rate": 0.1,
        "threats_list": [("T1", 3), ("T2", 2)],
        "arrival_rate": 0.02,
        "include_time_windows": False,
        "dynamic_idle_penalty": 0.05,
        "init_policy": "dqn_Custom/policy_TBTA_CurD2_latest.pth",
        "f_reward_eval_case": "D3_combined",
        "f_reward_eval_every": 5,
        "f_reward_eval_episodes": 5,
        "reward_weights": _rw(),
        "dqn": {"discount_factor": 0.98, "estimation_step": 8, "target_update_freq": 3000, "lr": 5e-5},
        "trainer": {
            "max_epoch": 15,
            "step_per_epoch": 3000,
            "step_per_collect": 1500,
            "episode_per_test": 5,
            "batch_size": 1500,
            "update_per_step": 1 / 300,
            "tn_eps_max": 0.15,
            "tn_eps_min": 0.05,
            "eps_decay_epochs": 12,
            "ts_eps_max": 0.0,
            "prioritized": False,
        },
    },
}


def make_env_config(exp: Dict[str, Any]) -> agentEnvOptions:
    sc = exp["scenario"]
    return agentEnvOptions(
        render_speed=-1,
        simulation_frame_rate=0.01,
        action_mode="TaskAssign",
        simulator_module="Internal",
        max_time_steps=150,
        agents=sc["agents"],
        tasks=sc["tasks"],
        multiple_tasks_per_agent=False,
        multiple_agents_per_task=True,
        random_init_pos=False,
        num_obstacles=0,
        hidden_obstacles=False,
        fail_rate=float(exp.get("fail_rate", 0.0) or 0.0),
        threats_list=list(exp.get("threats_list") or []),
        fixed_seed=-1,
        info=exp.get("desc", ""),
        early_terminate=exp["early_terminate"],
        capability_mask=exp["capability_mask"],
        saturate_mask=exp["saturate_mask"],
        reward_weights=copy.deepcopy(exp["reward_weights"]),
        arrival_rate=float(exp.get("arrival_rate", 0.0) or 0.0),
        include_time_windows=bool(exp.get("include_time_windows", False)),
        dynamic_idle_penalty=float(exp.get("dynamic_idle_penalty", 0.0) or 0.0),
    )


def _get_env(config: agentEnvOptions):
    env_parallel = MultiUAVEnv(config=config)
    env = CustomParallelToAECWrapper(env_parallel)
    return PettingZooEnv(env)


def _get_agents(config: agentEnvOptions, dqn_params: Dict[str, Any], same_policy: bool = True, model_name: str = "CustomNetMultiHead"):
    env = _get_env(config)
    agent_name = env.agents[0]
    obs_space = env.observation_space
    # Shapes are only used for Net construction; MultiHead builds features from dict obs.
    state_shape_agent = (
        obs_space["agent_position"].shape[0]
        + obs_space["agent_state"].shape[0]
        + obs_space["agent_type"].shape[0]
        + obs_space["next_free_time"].shape[0]
        + obs_space["position_after_last_task"].shape[0]
    )
    state_shape_task = 31 * 2
    action_shape = env.action_space[agent_name].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "SharedMLP":
        from TaskAllocation.RL_Policies.SharedMLPNet import SharedMLPNet

        net = SharedMLPNet(
            state_shape_agent=state_shape_agent,
            state_shape_task=state_shape_task,
            action_shape=action_shape,
            hidden_sizes=[256, 256],
            device=device,
        ).to(device)
    else:
        net = CustomNetMultiHead(
            state_shape_agent=state_shape_agent,
            state_shape_task=state_shape_task,
            action_shape=action_shape,
            hidden_sizes=[128, 128],
            device=device,
        ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=dqn_params["lr"], weight_decay=0.0, amsgrad=True)
    agent_learn = MaskAwareDQN(
        model=net,
        optim=optim,
        discount_factor=dqn_params["discount_factor"],
        estimation_step=dqn_params["estimation_step"],
        target_update_freq=dqn_params["target_update_freq"],
        reward_normalization=False,
        clip_loss_grad=False,
    )
    agents = [agent_learn for _ in range(len(env.agents))]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def train(
    exp_id: str,
    max_epoch: Optional[int] = None,
    seed: int = 0,
    train_env_num: int = 10,
    step_per_epoch: Optional[int] = None,
    step_per_collect: Optional[int] = None,
):
    if exp_id not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {exp_id}. Use --list.")

    exp = EXPERIMENTS[exp_id]
    dqn_params = exp["dqn"]
    trainer_params = dict(exp["trainer"])
    if max_epoch is not None:
        trainer_params["max_epoch"] = max_epoch
    if step_per_epoch is not None:
        trainer_params["step_per_epoch"] = step_per_epoch
    if step_per_collect is not None:
        trainer_params["step_per_collect"] = step_per_collect

    config = make_env_config(exp)
    tag = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_name = f"TBTA_{exp_id}_{tag}"
    policy_path = os.path.join(ROOT, "dqn_Custom")
    log_path = os.path.join(ROOT, "Logs", "dqn", run_name)
    os.makedirs(policy_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    model_save_path = os.path.join(policy_path, f"policy_{run_name}")

    meta = (
        f"exp={exp_id}\n"
        f"desc={exp['desc']}\n"
        f"git={_git_hash()}\n"
        f"seed={seed}\n"
        f"agents={config.agents}\n"
        f"tasks={config.tasks}\n"
        f"early_terminate={config.early_terminate}\n"
        f"capability_mask={config.capability_mask}\n"
        f"saturate_mask={config.saturate_mask}\n"
        f"reward_weights={config.reward_weights}\n"
        f"dqn={dqn_params}\n"
        f"trainer={trainer_params}\n"
    )
    print(meta)
    with open(os.path.join(log_path, "run_meta.txt"), "w", encoding="utf-8") as f:
        f.write(meta)

    torch.set_grad_enabled(True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_envs = DummyVectorEnv([lambda c=config: _get_env(c) for _ in range(train_env_num)])
    test_envs = DummyVectorEnv([lambda c=config: _get_env(c) for _ in range(train_env_num)])
    train_envs.seed(seed)
    test_envs.seed(seed)

    policy, optim, agents = _get_agents(config, dqn_params, model_name=exp.get("model", "CustomNetMultiHead"))

    init_path = exp.get("init_policy")
    if init_path:
        init_full = init_path if os.path.isabs(init_path) else os.path.join(ROOT, init_path)
        if os.path.exists(init_full):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(init_full, map_location=device)
            policy.policies[agents[0]].load_state_dict(state)
            print(f"Loaded init policy from {init_full}", flush=True)
        else:
            print(f"WARNING: init_policy not found: {init_full}", flush=True)

    if trainer_params.get("prioritized"):
        buffer = PrioritizedVectorReplayBuffer(300_000, len(train_envs), alpha=0.6, beta=0.4)
    else:
        buffer = VectorReplayBuffer(300_000, len(train_envs))

    train_collector = CustomCollector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = CustomCollector(policy, test_envs, exploration_noise=False)

    print("Buffer warming up...")
    warm_rounds = 5
    for _ in range(warm_rounds):
        train_collector.collect(n_episode=train_env_num)
        print(".", end="", flush=True)
    print(f"\nBuffer length (episodes approx): {len(train_collector.buffer) / 150:.1f}")

    writer = SummaryWriter(log_path)
    writer.add_text("Config", meta.replace("\n", "  \n"))
    logger = TensorboardLogger(writer)

    def save_best_fn(pol):
        torch.save(pol.policies[agents[0]].state_dict(), model_save_path + ".pth")
        print(f"Best saved -> {model_save_path}.pth", flush=True)

    def stop_fn(mean_rewards):
        return mean_rewards >= 9939.0

    eps_max = trainer_params["tn_eps_max"]
    eps_min = trainer_params.get("tn_eps_min", eps_max / 100)
    decay_epochs = trainer_params.get("eps_decay_epochs", trainer_params["max_epoch"])

    best_f_reward = float("-inf")
    f_case = exp.get("f_reward_eval_case")
    f_every = int(exp.get("f_reward_eval_every", 0) or 0)
    f_eps = int(exp.get("f_reward_eval_episodes", 5) or 5)
    latest_alias = os.path.join(policy_path, f"policy_TBTA_{exp_id}_latest.pth")
    best_f_path = model_save_path + "_bestF.pth"

    def train_fn(epoch, env_step):
        nonlocal best_f_reward
        t = min(1.0, epoch / max(1, decay_epochs))
        epsilon = eps_max - (eps_max - eps_min) * t
        policy.policies[agents[0]].set_eps(epsilon)
        print(f"[{exp_id}] epoch={epoch}/{trainer_params['max_epoch']} eps={epsilon:.3f} env_step={env_step}", flush=True)
        # Always persist latest weights; test_reward can be poorly scaled for multi-agent.
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path + ".pth")
        torch.save(policy.policies[agents[0]].state_dict(), latest_alias)

        if f_case and f_every > 0 and epoch > 0 and epoch % f_every == 0:
            from experiments.eval_f_reward import eval_f_reward_tbta

            policy.policies[agents[0]].set_eps(0.0)
            # Use shared policy head for rollout
            mean_f = eval_f_reward_tbta(policy.policies[agents[0]], case_id=f_case, episodes=f_eps)
            writer.add_scalar("eval/F_Reward", mean_f, epoch)
            print(f"[{exp_id}] F_Reward@{f_case}={mean_f:.1f} (best={best_f_reward:.1f})", flush=True)
            if mean_f > best_f_reward:
                best_f_reward = mean_f
                torch.save(policy.policies[agents[0]].state_dict(), best_f_path)
                print(f"[{exp_id}] New best F_Reward checkpoint -> {best_f_path}", flush=True)
            policy.policies[agents[0]].set_eps(epsilon)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(trainer_params["ts_eps_max"])

    def reward_metric(rews):
        # Aggregate team return (all agents share the same global reward).
        return rews.mean(axis=1) if getattr(rews, "ndim", 1) > 1 else rews

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=trainer_params["max_epoch"],
        step_per_epoch=trainer_params["step_per_epoch"],
        step_per_collect=trainer_params["step_per_collect"],
        episode_per_test=trainer_params["episode_per_test"],
        batch_size=trainer_params["batch_size"],
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=trainer_params["update_per_step"],
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
        show_progress=False,
    )
    writer.close()
    # Ensure final weights are on disk even if best_reward never improved.
    torch.save(policy.policies[agents[0]].state_dict(), model_save_path + ".pth")
    print(f"\n========== Result ==========\n{result}", flush=True)
    print(f"Checkpoint: {model_save_path}.pth", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="TBTA DQN experiment runner")
    parser.add_argument("--exp", type=str, default="E1", help="Experiment id (see --list)")
    parser.add_argument("--list", action="store_true", help="List experiments")
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-env-num", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=None)
    parser.add_argument("--step-per-collect", type=int, default=None)
    args = parser.parse_args()

    if args.list:
        for k, v in EXPERIMENTS.items():
            print(f"{k:8} {v['desc']}")
        return

    train(
        args.exp,
        max_epoch=args.max_epoch,
        seed=args.seed,
        train_env_num=args.train_env_num,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
    )


if __name__ == "__main__":
    main()
