#%%%

import numpy as np
from pettingzoo.sisl import pursuit_v4
import random

# import wandb


def get_direction_to_closest_enemy(observation):
    # Assuming the agent is at the center of the observation grid
    
    grid_center = np.array(observation.shape[:2]) // 2
    
    # Assuming enemy_channel is the second channel in observation
    enemy_channel = observation[:, :, 2]
    # print(enemy_channel)

    if np.sum(enemy_channel) == 0:
        return random.randint(0,3)
    # Calculate the positions of the enemies relative to the agent
    enemy_positions = np.argwhere(enemy_channel > 0) - grid_center

    # print(enemy_positions)
    # Calculate distances to all enemies
    distances = np.linalg.norm(enemy_positions, axis=1)

    # print(distances)

    # Get the closest enemy's relative position
    closest_enemy_rel_pos = enemy_positions[np.argmin(distances)]
    
    # Determine the direction to the closest enemy
    if np.abs(closest_enemy_rel_pos[0]) > np.abs(closest_enemy_rel_pos[1]) or random.random() > 0.95:
        # Move vertically
        action = 2 if closest_enemy_rel_pos[0] > 0 else 3
    else:
        # Move horizontally
        action = 1 if closest_enemy_rel_pos[1] > 0 else 0

    return action

seed = 10
runs = 10

SISL_Config = {
    "max_cycles": 150,         # default: 500
    "x_size": 10,              # default: 16
    "y_size": 10,              # default: 16
    "shared_reward": True,     # default: True
    "n_evaders": 10,           # default: 30
    "n_pursuers": 5,           # default: 10
    "obs_range": 7,            # default: 7
    "n_catch": 2,              # default: 2
    "freeze_evaders": False,   # default: False
    "tag_reward": 0.01,        # default: 0.01
    "catch_reward": 5.0,       # default: 5.0
    "urgency_reward": -0.1,    # default: -0.1
    "surround": True,         # default: True
    "constraint_window": 1.0,   # default: 1.0
    "render_mode" : "human"       # default: None
}


# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="SISL_EVAL01",
    
#     # track hyperparameters and run metadata
#     config={
#     "size": size,
#     "architecture": "CNN",
#     "dataset": "SISL_Pursuer",
#     "repetitions": 10,
#     }
# )
       
env = pursuit_v4.env(
            max_cycles=SISL_Config["max_cycles"],
            x_size=SISL_Config["x_size"],
            y_size=SISL_Config["y_size"],
            shared_reward=SISL_Config["shared_reward"],
            n_evaders=SISL_Config["n_evaders"],
            n_pursuers=SISL_Config["n_pursuers"],
            obs_range=SISL_Config["obs_range"],
            n_catch=SISL_Config["n_catch"],
            freeze_evaders=SISL_Config["freeze_evaders"],
            tag_reward=SISL_Config["tag_reward"],
            catch_reward=SISL_Config["catch_reward"],
            urgency_reward=SISL_Config["urgency_reward"],
            surround=SISL_Config["surround"],
            constraint_window=SISL_Config["constraint_window"],
            render_mode=SISL_Config["render_mode"]
        )

# env = pursuit_v4.env()


rewards = []
temp_rews = []
total_rews = 0

n_agents = SISL_Config["n_pursuers"]
print("Starting Simulations")
for i in range(runs):
    env.reset(seed=seed + 1)
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        idx_agent = env.agents.index(agent)
                
        if idx_agent == n_agents -1:
            temp_rews.append(reward)
            # log metrics to wandb
            
            rewards.append(temp_rews)
            total_rews += np.mean(temp_rews)

            # wandb.log({"rews": temp_rews, "total": total_rews})

            temp_rews = []
            
        else:
            temp_rews.append(reward)
        
        if termination or truncation:
            action = None
        else:
            # print(observation.shape)
            # this is where you would insert your policy                
            action = get_direction_to_closest_enemy(observation) 
            # action = env.action_space(agent).sample()
            # action=0
        
        # print(f'{idx_agent} | {action}')    
        
        env.step(action) 

           
env.close()
print(total_rews/(i+1))

