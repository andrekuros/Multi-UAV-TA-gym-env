from pettingzoo.sisl import pursuit_v4
from collections import deque
import numpy as np
from pettingzoo.utils import agent_selector, wrappers

class MemPursuitEnv(pursuit_v4.raw_env):
    
    def __init__(self, att_memory = False, memory_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.memory_size = memory_size
        self.att_memory = att_memory
        # Initialize state memory for each agent
        if att_memory:
            self.state_memories = {agent: deque(maxlen=self.memory_size) for agent in self.agents}
        

    def step(self, action):
        
        
        super().step(action)
        
        if self.att_memory:
            
            # Update state memory agents
            agent = self.agent_selection        
            agent_obs = self.env.safely_observe(self.agent_name_mapping[agent])
            
            # Ensure the observation is correctly formatted for memory storage
            self.state_memories[agent].append(np.swapaxes(agent_obs, 0, 2))
            

    def observe(self, agent):
        current_obs = super().observe(agent)  # This already returns a correctly formatted observation
        
        if self.att_memory:
            state_memory = np.array(list(self.state_memories[agent]))
            if state_memory.shape[0] < self.memory_size:
                # Pad the state memory if it's not yet full
                pad_width = self.memory_size - state_memory.shape[0]
                # print("Before padding:", state_memory.shape)
                state_memory = np.pad(state_memory, ((pad_width, 0), (0, 0), (0, 0), (0, 0)), mode='constant')

            combined_obs = np.concatenate((current_obs[np.newaxis, ...], state_memory), axis=0)
            return combined_obs

        else:

            return current_obs
    
    def reset(self, seed=None, options=None):
        # Call the base environment's reset
        super().reset(seed=seed, options=options)
        
        if self.att_memory:
            # Reset the state memories for each agent
            for agent in self.agents:
                self.state_memories[agent].clear()
                # Optionally, you can initialize the state memory with a starting state
                initial_state = np.zeros((self.observation_spaces[agent].shape[0], self.observation_spaces[agent].shape[1], self.observation_spaces[agent].shape[2]))
                for _ in range(self.memory_size):
                    self.state_memories[agent].append(initial_state)


def env(**kwargs):
    env = MemPursuitEnv(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
