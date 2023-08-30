import warnings
from abc import ABC
from typing import Any, Dict, List, Tuple

import pettingzoo
from gymnasium import spaces
from packaging import version
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo import ParallelEnv

from PPO_CleanRL import Agent

if version.parse(pettingzoo.__version__) < version.parse("1.21.0"):
    warnings.warn(
        f"You are using PettingZoo {pettingzoo.__version__}. "
        f"Future tianshou versions may not support PettingZoo<1.21.0. "
        f"Consider upgrading your PettingZoo version.", DeprecationWarning
    )


class PettingZooEnv(ParallelEnv, ABC):
    """The interface for petting zoo environments.

    Multi-agent environments must be wrapped as
    :class:`~tianshou.env.PettingZooEnv`. Here is the usage:
    ::

        env = PettingZooEnv(...)
        # obs is a dict containing obs, agent_id, and mask
        obs = env.reset()
        action = policy(obs)
        obs, rew, trunc, term, info = env.step(action)
        env.close()

    The available action's mask is set to True, otherwise it is set to False.
    Further usage can be found at :ref:`marl_example`.
    """

    def __init__(self, env: BaseWrapper):
        super().__init__()
        self.env = env
        # agent idx list
        self.agents = self.env.possible_agents
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i

        self.rewards = [0] * len(self.agents)

        # Get first observation space, assuming all agents have equal space
        self.observation_space: Any = self.env.observation_space(self.agents[0])

        # Get first action space, assuming all agents have equal space
        self.action_space: Any = self.env.action_space(self.agents[0])

        assert all(self.env.observation_space(agent) == self.observation_space
                   for agent in self.agents), \
            "Observation spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_observations wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_observations(env)`"

        assert all(self.env.action_space(agent) == self.action_space
                   for agent in self.agents), \
            "Action spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_action_space wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[dict, dict]:
        self.env.reset(*args, **kwargs)

        #observation, reward, terminated, truncated, info = self.env.last(self)
        observation, reward, terminated, truncated, info = self.env.last() 

        #observation_dict = { 'obs': { 'agent_id' : i,
        #                     'obs'   : observation[agent]}  for i, agent in enumerate(self.agents)
        #}
        return observation, info

    def step(self, action: Any) -> Tuple[Dict, List[int], bool, bool, Dict]:
        self.env.step(action)

        observation, rew, term, trunc, info = self.env.last()

        #obs = { 'obs': { 'agent_id' : i,
        #                     'obs'   : observation[agent]}  for i, agent in enumerate(self.agents)

        for agent_id, reward in self.env.rewards.items():
            self.rewards[self.agent_idx[agent_id]] = reward
        
        return obs, self.rewards, term, trunc, info

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        #try:
        #    self.env.seed(seed)
        #except (NotImplementedError, AttributeError):
        self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()