# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:14:49 2023

@author: andre
"""
import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import copy


def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    
    env = _get_env()
    agent_name = env.agents[0]  # Get the name of the first agent
    agent_observation_space = env.observation_space[agent_name]  # Get the observation_space for that agent
       
    agent_observation_space = env.observation_space["agent0"]  # assuming 'agent0' is a valid agent name
    state_shape_agent_position = agent_observation_space["agent_position"].shape[0]
    state_shape_agent_state = agent_observation_space["agent_state"].n
    state_shape_agent_type = agent_observation_space["agent_type"].n
    state_shape_agent_relay_area = agent_observation_space["agent_relay_area"].shape[0]
    
    state_shape_agent = (state_shape_agent_position +
                     state_shape_agent_state +
                     state_shape_agent_type +
                     state_shape_agent_relay_area
                     )

                     
    
    state_shape_task = env.observation_space["agent0"]["tasks_info"].shape[0]
      
    # rest of the code         
  #  action_shape = len(env.action_space)   
    action_shape = env.action_space[agent_name].n

           
    if agent_learn is None:
        # model
        net = CustomNet(
            state_shape_agent=state_shape_agent,
            state_shape_task=state_shape_task,
            action_shape=action_shape,
            hidden_sizes=[10,10],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=100,
        )
        #if agent_opponent is None:
        #    agent_opponent = RandomPolicy()
        # Use the same DQN policy for all agents
#        agents = [agent_learn] * len(env.agents)
        agents = [agent_learn for _ in range(len(env.agents))]


    #print (agents, agent_learn)

    policy = MultiAgentPolicyManager(agents, env)    
        
    return policy, optim, env.agents

import torch


from Custom_Classes import CustomNet
from Custom_Classes import CustomCollector
from Custom_Classes import CustomParallelToAECWrapper

#from pettingzoo.classic import tictactoe_v3
# Import your custom environment
from DroneEnv import MultiDroneEnv
from tianshou.data import Batch


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env_paralell = MultiDroneEnv()
    #env = parallel_to_aec_wrapper(env_paralell)
    
    env = CustomParallelToAECWrapper(env_paralell)
    
    return PettingZooEnv(env)


#%%%

import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import copy


import torch


from Custom_Classes import CustomNet
from Custom_Classes import CustomCollector
from Custom_Classes import CustomParallelToAECWrapper

#from pettingzoo.classic import tictactoe_v3
# Import your custom environment
from DroneEnv import MultiDroneEnv

from tianshou.data import Batch


def run_simulation(policy, env, max_steps=1000):
    env.reset()
    done = False
    step_count = 0
    total_rewards = {agent: 0 for agent in env.possible_agents}
    
    while not done and step_count < max_steps:
        obs, rewards, done, _ = env.last()

        actions = {}
        for agent_id, agent_obs in obs.items():

            filtered_agent_obs = {k: np.array([v]) if isinstance(v, (int, float)) else np.expand_dims(v, axis=0) for k, v in agent_obs.items() if v is not None}

            agent_batch = Batch(obs=filtered_agent_obs, info={})
            print("Agent batch:", agent_batch)

            action, _ = policy(agent_batch)
            actions[agent_id] = action.item()

        env.step(actions)
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        env.render()
        step_count += 1

    return total_rewards




#env_paralell = MultiDroneEnv()
#env = CustomParallelToAECWrapper(env_paralell)

env = MultiDroneEnv(None)

# Create a new instance of the policy with the same architecture as the saved policy
policy, optim, env.agents = _get_agents()

model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
# Load the saved checkpoint
policy_test = policy.policies['agent0']
policy_test.load_state_dict(torch.load(model_save_path ))

print("Policy object:", policy_test)
print("Policy attributes:", dir(policy_test))

simulation_results = run_simulation(policy_test, env)
print(f"\n==========Simulation Results==========\n{simulation_results}")
