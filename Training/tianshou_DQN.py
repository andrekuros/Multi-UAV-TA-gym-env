# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:47:47 2023

@author: andre
"""

import torch
import torch.nn as nn
import torch.optim as optim

import tianshou as ts
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.env.pettingzoo_env import PettingZooEnv

from pettingzoo.utils.conversions import parallel_to_aec_wrapper

# Import your custom environment
from DroneEnv import MultiDroneEnv
#from DroneEnv import env

# Define a simple Q-Network
class QNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape)
        )

    def forward(self, x):
        return self.model(x)
def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env_paralell = MultiDroneEnv()
    env = parallel_to_aec_wrapper(env_paralell)
    
    return PettingZooEnv(env)


def train():
    # Set up environment
    env_paralell = MultiDroneEnv()
    env = parallel_to_aec_wrapper(env_paralell)
    env = PettingZooEnv(env)
    env.reset()
    
    state_shape_agent_position = env.observation_space["agent_position"].shape[0]
    state_shape_task_status = env.observation_space["task_status"].shape[0]
    state_shape = state_shape_agent_position + state_shape_task_status
    action_shape = env.action_space.n

    # Create neural network, policy, optimizer, and buffer
    net = QNet(state_shape, action_shape)
    policy = DQNPolicy(net, optim.Adam(net.parameters(), lr=1e-3))
    buffer = ReplayBuffer(size=100, n_step=3)


    # Create training and test environments
    train_envs = SubprocVectorEnv([lambda: _get_env() for _ in range(1)])
    test_envs = SubprocVectorEnv([lambda: _get_env() for _ in range(1)])

    # Create collectors
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    # Train the policy
    offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=2,
        step_per_epoch=10,
        step_per_collect=10,
        episode_per_test=10,
        update_per_step=0.1,
        test_num=10,
        batch_size=1,
        train_fn=None,
        test_fn=None,
        stop_fn=None,
        save_fn=None,
        writer=None,
        log_interval=10,
        verbose=True,
    )

    # Save the policy
    torch.save(policy.state_dict(), 'policy.pth')

if __name__ == '__main__':
    train()
