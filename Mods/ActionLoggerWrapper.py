from pettingzoo.utils import wrappers
import wandb
import numpy as np


class ActionLoggerWrapper(wrappers.BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = {agent: [] for agent in self.env.possible_agents}

    def step(self, action):
        agent = self.agent_selection
        self.actions[agent].append(action)
        return self.env.step(action)

    def reset(self, **kwargs):
        # Log actions of each agent
        for agent, actions in self.actions.items():
            if actions:
                flattened_actions = np.concatenate(actions).ravel()
                wandb.log({f"actions_histogram_{agent}": wandb.Histogram(np.histogram(flattened_actions, bins='auto'))})
                self.actions[agent] = []
        return self.env.reset(**kwargs)