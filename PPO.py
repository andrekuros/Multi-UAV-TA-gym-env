import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from DroneEnv import MultiDroneEnv
import DroneEnv as mdenv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)
    
class PPO:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, gamma, eps_clip, K_epochs):
        self.policy = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state):
        # Convert the state dictionary to a tensor
        print(state)
        agent_position = state["agent_position"]
        target_status = state["target_status"]
    
        # You may need to modify the following line depending on how you want to
        # represent the state as a tensor.
        state_tensor = torch.tensor(np.concatenate((agent_position, target_status)), dtype=torch.float32)
    
        # Get the action probabilities from the policy network
        action_probs = self.policy(state_tensor)
        
        # Sample an action using the probabilities
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
    
        # Save the action's log probability
        self.policy.saved_log_probs.append(action_distribution.log_prob(action))
    
        return action.item()

    def update(self, states, actions, log_probs, rewards, dones):
        for _ in range(self.K_epochs):
            new_log_probs, state_values = self.evaluate(states, actions)
            ratios = torch.exp(new_log_probs - log_probs)

            advantages = rewards + self.gamma * state_values * (1 - dones) - state_values

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surrogate1, surrogate2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
def train_ppo(env, num_episodes, input_dim, hidden_dim, output_dim, lr, gamma, eps_clip, K_epochs):
    ppo = PPO(input_dim, hidden_dim, output_dim, lr, gamma, eps_clip, K_epochs)
    rewards = []

    for episode in range(num_episodes):
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        state = env.reset()
        done = False

        while not done:
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        rewards_sum = sum(rewards)
        rewards.append(rewards_sum)
        print("Episode: {}, Reward: {}".format(episode, rewards_sum))

        ppo.update(states, actions, log_probs, rewards, dones)

    return ppo


num_drones = 15
num_targets = 50
episodes = 10
render_speed = -1


# Hyperparameters
input_dim = 2 * (num_drones + num_targets)
hidden_dim = 128
env = mdenv.env(action_mode= "TaskAssign", render_speed=render_speed, num_drones=num_drones, num_targets=num_targets)
output_dim = env.NUM_DRONES * env.NUM_TARGETS
lr = 0.001
gamma = 0.99
eps_clip = 0.2
K_epochs = 10
num_episodes = 1000

# Train the agent
trained_ppo_agent = train_ppo(env, num_episodes, input_dim, hidden_dim, output_dim, lr, gamma, eps_clip, K_epochs)

# Save the trained model
torch.save(trained_ppo_agent.policy.state_dict(), 'ppo_trained_multi_drone_env.pth')
