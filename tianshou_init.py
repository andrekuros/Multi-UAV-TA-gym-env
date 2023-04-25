from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
import numpy as np

# Import your custom environment
from DroneEnv import MultiDroneEnv

class CustomCollector(Collector):
    def __init__(self, policy, env):
        super().__init__(policy, env)

    def _step(self, policy_output):
        actions = {agent_id: output.act for agent_id, output in policy_output.items()}
        obs, rewards, dones, infos = self.env.step(actions)
        return obs, rewards, dones, infos

    def collect(self, n_episode=1, render=0.1):
        self.reset_env()
        for _ in range(n_episode):
            all_agents_done = False
            while not all_agents_done:
                actions = { 0 for _ in range(10)}#self.policy(self.env)
                obs, rewards, dones, infos = self._step(actions)
                self.env.render(render)
                all_agents_done = all(dones)
                self.env.env.env.observation = obs
                self.env.env.env.done.update(dones)




if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env_paralell = MultiDroneEnv()
    env_fn = parallel_to_aec_wrapper(env_paralell)

    # Step 2: Wrap the environment for Tianshou interfacing
    envPZ = PettingZooEnv(env_fn)

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy() for _ in env_paralell.possible_agents], envPZ)

    # Step 4: Convert the env to vector format
    envPZ = DummyVectorEnv([lambda: envPZ])

    # Step 5: Construct the Custom Collector, which interfaces the policies with the vectorised environment
    custom_collector = CustomCollector(policies, envPZ)   

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = custom_collector.collect(n_episode=1, render=0.1)
