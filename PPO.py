import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from DroneEnv import DroneEnv

# Criar o ambiente DroneEnv no modo "TaskAssign"
env = DroneEnv(action_mode="TaskAssign",  render_enabled=False)

# Verificar a validade do ambiente
check_env(env)

#%%
# Envolva o ambiente em um vetor DummyVecEnv para compatibilidade com Stable Baselines
vec_env = DummyVecEnv([lambda: env])

# Crie e treine o agente PPO
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=200000)

# Teste o agente treinado
num_episodes = 100
total_rewards = []
for i in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    
    total_rewards.append(episode_reward)

print("Recompensa média em 100 episódios:", np.mean(total_rewards))
