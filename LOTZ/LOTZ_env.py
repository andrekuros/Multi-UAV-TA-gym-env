#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv
from gymnasium import spaces
from pettingzoo.utils import agent_selector, wrappers

class agent():
    """Primarily used to track the best LOTZ score and print it to the screen."""
    def __init__(self, model, index):
        self.model = model
        self.index = index
        self.lastact = 0
        self.best_rew = 0

def env(config = None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = LeadingOnesTrailingZerosEnv(config)

    env = wrappers.OrderEnforcingWrapper(env)
    
    return env


# Define the custom environment for LOTZ
class LeadingOnesTrailingZerosEnv(AECEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Env Constructor
    def __init__(self, string_length=6, n_agents=2, seed=0, m_steps = 128, sp = 0):
        metadata = {"render_modes": ["human"], "name": "multi_agent_env_v0"}
        super().__init__()
        self.env = self
        self.agent = None
        self.string_length = string_length
        self.n_agents = n_agents
        self._action_space = spaces.Discrete(self.string_length)# + 1)  # Each agent can flip any bit
        self._observation_space = spaces.Box(low=0, high=1, shape=(self.string_length,), dtype=np.int32)
        # self.observation_space = spaces.Dict({
        #     'bit_string': spaces.Box(low=0, high=1, shape=(self.string_length,), dtype=np.int32),
        #     'reward': spaces.Box(low=np.float32(-1.0), high=np.float32(1.0), shape=(1,), dtype=np.float32)
        # })
        self.state = np.array([0] * (self.string_length//2) + [1] * (self.string_length//2))
        self.current_step = 0
        self.solved = 0
        self.optimal = 0
        self.epsilon = 1.0
        self.current_epoch = 0
        self._seed = seed
        self.m_steps = m_steps
        self.sp = sp
        # self.tgt = np.array([1] * (self.string_length//2) + [0] * (self.string_length//2))
        # self.prev_lo = self.count_leading_ones()
        # self.prev_tz = self.count_trailing_zeros()
        #self.solved = False
        self.best = 0

        #Updated to handle odd bitstrings
        half_length = self.string_length // 2
        remaining_length = self.string_length - half_length
        self.op_rew = ((half_length * remaining_length))
        self.reward_range = (0, self.op_rew + m_steps)
        # print(f"Bit_String Length {self.string_length}, Optimum Reward {self.op_rew}")

        #KUR Modifications
        self.agents = ["agent" + str(a) for a in range(self.n_agents)]
        self.possible_agents = self.agents[:]        
        self._agent_selector = agent_selector(self.agents)        
        self.agent_selection = self.possible_agents[0] 

        self.observation_spaces = { agent: self._observation_space for agent in self.possible_agents }
        self.action_spaces = { agent: self._action_space for agent in self.possible_agents }

        self.observations = None
        self.rewards = None
        self.terminations = None
        self.truncations = None
        self.infos = None

        self._cumulative_rewards = {0  for agent in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent] 
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def last(self):
        """Return the last observations, rewards, and done status."""
        agent = self.agent_selection
        # print("AG:",agent, "|", self.time_steps)        
        
        return (
            self.observations,
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos,
        )
        #self._generate_observations()
        #return self.observations, self.rewards, self.terminations, self.truncations,self.infos
           
    def seed(self,seed):
        self.reset(seed = seed)
        #print("Call def_seed :", seed)     

    # Env Reset
    def reset(self, seed=0, options=None):
        # super().reset(seed=seed)
        self._seed = seed
        # print(seed)
        info = {"options": options} if options else {}
        self.current_step = 0  # Reset step count at the start of each episode
        self.best_rew = 0
        
        self._cumulative_rewards = {agent : 0  for agent in self.possible_agents}
        # If none seed explicitly generate worst case, 6-bit string would be 000111 - comment out if you use seed
        if False:
            # self.state = np.random.randint(2, size=self.string_length, dtype=np.int32)
            self.state = np.array([0] * (self.string_length // 2) + [1] * (self.string_length // 2), dtype=np.int32)
        else:
            # np.random.seed(seed)
            self.state = np.random.randint(2, size=self.string_length, dtype=np.int32)

        # print(f"START STATE : {self.state}")
        self.rewards =  {agent : 0  for agent in self.possible_agents}
        self.terminations =  {agent : False  for agent in self.possible_agents}
        self.truncations =  {agent : False  for agent in self.possible_agents} 
        self.observations =  {agent : self.state  for agent in self.possible_agents}  

        self.last_lotz = 0 

        # print("Reset: ", self.state)     

        self.agent_selection = self.possible_agents[0] 
        self.infos = {agent: {} for agent in self.possible_agents}    

        return self.observations, self.infos
        

    # Env Step
    def step(self, action):
        #self.last_rew = self.calculate_reward()
        #print(f"Action : {action}")
        agent = self.agent_selection

        self.current_step += 1
        done = False
        truncated = False
        actions = []
        #cls = self.calculate_reward() #current LOTZ score
        reward = 0
        info = {}
        # if self.current_step == 1:
        #     print(f"Initial State : {self.state}")
        # print(f"Action : {action}, Step : {self.current_step}")

        if action < len(self.state):            
            self.state[action] ^= 1
        else:
            print("outAction")

        
        # print(f"New State : {self.state}")

       #Normalized Method##############################################################################################
        # Maximum possible cumulative reward for the whole episode
        max_step_reward = self.op_rew + (self.m_steps - self.string_length)

        # Calculate the LOTZ score
        lotz = self.calculate_lotz()

        # Intermediate reward
        reward = (lotz - self.last_lotz) #* (self.current_step / self.m_steps) #/ max_step_reward
        
        if reward == 0:
            reward -= max_step_reward / self.string_length
        
        self.last_lotz = lotz


        # Check for terminal conditions
        if self.current_step >= self.m_steps:
            # reward = lotz 
            done = True
            
            # print("Final LOTZ:", lotz)
            # print("Final STATE: ", self.state)
            info = {"TimeLimit.truncated": True}
            truncated = True

        # Solved Reward
        if lotz == self.op_rew:
            # print(f"VICTORY!!!, LOTZ : {lotz}, Optimum Reward : {self.op_rew}")
            # Bonus for solving should not exceed the max step reward
            reward = (self.op_rew + (self.m_steps - self.current_step)) #/ max_step_reward
            done = True

            

        #Uncomment to bound the cumulative episode reward to less than 1
        # reward = reward / m_steps
        # Subtract the step penalty from the scaled reward
        #reward -= self.sp   #self.sp = 1/m_steps
        #reward += bonus
        # Normalized Method#############################################################################################

        # Update the best reward seen so far, if applicable
        if lotz > self.best:
            self.best = lotz
            #print(f"Best = {self.best_rew}")

        
        # rewards for all agents are placed in the rewards dictionary to be returned

        self.observations = {agent : self.state for agent in self.possible_agents}

        self.rewards = {_agent : (reward if _agent == agent else 0) for _agent in self.possible_agents } #Rand -28           
        # print("Step Rewards: ", self.rewards)
        self._cumulative_rewards[agent] += reward                                                                                      
        
        self.terminations = { agent : done for agent in self.possible_agents}
                            
        self.truncations = {agent : truncated for agent in self.possible_agents}     
        
        self.infos = {agent: {} for agent in self.possible_agents}  

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def calculate_reward(self, a=0.5, b=0.5):
        half = (len(self.state) + 1) // 2  # Handles odd-length arrays by rounding up
        leading_ones = np.sum(self.state[:half] == 1)
        trailing_zeros = np.sum(self.state[half:] == 0)
        #print(f"LO : {leading_ones} , TZ : {trailing_zeros}")
        return a * (leading_ones / len(self.state)) + b * (trailing_zeros / len(self.state))

    def calculate_lotz(self) :
        """Calculate and return the LOTZ Score"""
        return np.argmax(self.state == 0) * np.argmax(self.state[::-1] == 1)

    def render(self, mode='human'):
        # print(f"Reward = {self.calculate_reward()}, Optimal Reward = {self.op_rew}")
        if self.calculate_reward() == self.op_rew:
            print(f"Optimal Configuration Reached in {self.current_step} steps")
            print("Current State:", ''.join(map(str, self.state)))
            self.solved += 1
            if self.current_step <= self.string_length: self.optimal += 1

    def count_leading_ones(self):
        """Count the number of leading ones in the current state."""
        lo = 0
        for bit in self.state:
            if bit == 1:
                lo += 1
            else:
                break
        return lo

    def count_trailing_zeros(self):
        """Count the number of trailing zeros in the current state."""
        tz = 0
        for bit in reversed(self.state):
            if bit == 0:
                tz += 1
            else:
                break
        return tz

    def check_improvement(self):
        """Check if the number of leading ones or trailing zeros has increased."""
        # Count current leading ones and trailing zeros
        current_lo = self.count_leading_ones()
        current_tz = self.count_trailing_zeros()

        # Check for improvement
        if current_lo > self.prev_lo or current_tz > self.prev_tz:
            # Update previous counts
            self.prev_lo = current_lo
            self.prev_tz = current_tz
            return True
        else:
            self.prev_lo = current_lo
            self.prev_tz = current_tz
            return False

# HELPER FUNCTIONs
def switch_agent_in_wrapper(wrapped_env, agent):
    """Switch the agent in the StableBaselinesWrapper."""
    wrapped_env.agent = agent
#
def make_env():
    """Used to help create multiple vector environments."""
    return LeadingOnesTrailingZerosEnv(string_length=sl, n_agents=n_agents, seed=my_seed, m_steps=m_steps, sp=sp)

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func