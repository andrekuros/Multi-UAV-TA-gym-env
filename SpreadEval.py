#%%%

import numpy as np
from pettingzoo.mpe import simple_spread_v3
import random
import pygame

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Define a function to find the closest landmark to an agent
def find_closest_landmark(agent_pos, landmark_positions):
    closest_landmark = None
    min_distance = float('inf')
    for landmark_pos in landmark_positions:
        dist = np.linalg.norm(np.array(landmark_pos))
        if dist < min_distance:
            min_distance = dist
            closest_landmark = landmark_pos
    return closest_landmark


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0

    def update(self, position_error, speed, dt=1):
        # Incorporate speed into the error calculation
        alpha = -0.2  # Weighting factor for speed
        combined_error = position_error + alpha * speed

        self.integral += combined_error * dt
        derivative = (combined_error - self.prev_error) / dt
        output = self.Kp * combined_error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = combined_error
        return output

# Initialize PID controllers for x and y directions
pid_x = PIDController(Kp=0.5, Ki=-0.0000, Kd=-0.0, setpoint=0)
pid_y = PIDController(Kp=0.5, Ki=-0.0000, Kd=-0.0, setpoint=0)


def choose_actionPID(agent_index, landmark_positions, agent_speed):
    threshold = 0.01
    pid_output_x = pid_controllers_x[agent_index].update(landmark_positions[0], agent_speed[0])
    pid_output_y = pid_controllers_y[agent_index].update(landmark_positions[1], agent_speed[1])
  
    
    threshold = 0.01
    # pid_output_x = pid_x.update(landmark_positions[0], agent_speed[0])
    # pid_output_y = pid_y.update(landmark_positions[1], agent_speed[1])

        

    # Discretize the PID controller outputs
    if abs(pid_output_x) > abs(pid_output_y):                
        if pid_output_x > threshold:
            action = 2  # Move right
        elif pid_output_x < -threshold:
            action = 1  # Move left
        else:
            action = 0  # No action            
    else:        
        if pid_output_y > threshold:
            action = 4  # Move down
        elif pid_output_y < -threshold:
            action = 3  # Move up
        else:
            action = 0  # No action            
    
    return action


# Define a function to choose a discrete action based on the direction to the closest landmark
def choose_action(closest_landmark_pos):
    # Calculate the direction vector towards the closest landmark
    # direction_vector = np.array(closest_landmark_pos) - np.array(agent_pos)
    # print(closest_landmark_pos)
    # print(direction_vector)
    action = 0

    # if action == -1 or (abs(closest_landmark_pos[0]) < 0.1 and abs(closest_landmark_pos[1])):
    #     action = 0  # No action
    disRef = 0.1

    
    # Determine the action based on the direction vector
    if abs(closest_landmark_pos[0]) > abs(closest_landmark_pos[1]):

        if abs(closest_landmark_pos[0]) > disRef:
            if closest_landmark_pos[0] > 0 :
                action = 2  # Move right
            elif closest_landmark_pos[0] < 0:
                action = 1  # Move left
    else:
        
        if abs(closest_landmark_pos[1]) > disRef:
            if closest_landmark_pos[1] > 0:
                action = 4  # Move down
            elif closest_landmark_pos[1] < 0:
                action = 3  # Move up
     
        
    # print(action)
    return action

def direction_to_avoid_target(agent_position, agent_speed, target_position):
    """
    Calculate a direction perpendicular to the agent's velocity vector to avoid the target.
    This direction is chosen based on the agent's current movement relative to the target.
    """
    dx = target_position[0] - agent_position[0]
    dy = target_position[1] - agent_position[1]

    vx, vy = agent_speed

    # Check if the agent is moving towards the target
    moving_towards_target = (dx * vx + dy * vy) > 0

    if not moving_towards_target or (dx == 0 and dy == 0):
        return -1  # Stay or random direction if preferred

    # Determine the perpendicular direction based on agent's velocity vector
    if abs(vx) > abs(vy):
        # If the agent is moving more in the x direction, move in the y direction
        return 3 if vx > 0 else 4  # Up (3) if moving right, Down (4) if moving left
    else:
        # If the agent is moving more in the y direction, move in the x direction
        return 2 if vy > 0 else 1  # Right (2) if moving down, Left (1) if moving up


seed = 0
runs = 30

Spread_Config = {
    "N": 3,                      # Default = 3
    "local_ratio": 0.5,          # Default = 0.5
    "max_cycles": 25,            # Default = 25
    "continuous_actions": False, # Default = False
    "render_mode": None#"human"          # Default = None 
}

rewards = []
temp_rews = []
total_rews = 0

n_agents = Spread_Config["N"]
print("Starting Simulations")

if Spread_Config['render_mode'] == "human":
    # Initialize Pygame
    pygame.init()

    # Set the desired frame rate (e.g., 30 frames per second)
    desired_fps = 15
    clock = pygame.time.Clock()

env = simple_spread_v3.env(
        max_cycles=Spread_Config["max_cycles"],
        local_ratio=Spread_Config["local_ratio"],
        N=Spread_Config["N"],
        continuous_actions=Spread_Config["continuous_actions"],
        render_mode=Spread_Config["render_mode"]
    )    


env.reset(seed=seed)
import re


for i in range(runs):
        
    env.reset(seed=seed + i)       

    num_agents = Spread_Config["N"]

    pid_controllers_x = [PIDController(Kp=0.5, Ki=-0.0000, Kd=-0.0, setpoint=0) for _ in range(num_agents)]
    pid_controllers_y = [PIDController(Kp=0.5, Ki=-0.0000, Kd=-0.0, setpoint=0) for _ in range(num_agents)]

    reward_run = 0 
    for step, agent in enumerate(env.agent_iter()):
                
        observation, reward, termination, truncation, info = env.last()
        ind = int((re.search(r'\d+', agent)).group())
        #ind = 0
        agent_index = int(re.search(r'\d+', agent).group())

        reward_run += reward / 3

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            # action = 0#env.action_space(agent).sample()
            # print(observation)   
            # closest_landmark_pos = find_closest_landmark(agent_pos, landmark_positions)

            agent_speed = np.array([observation[0], observation[1]])
            agent_pos = np.array([observation[2], observation[3]])
            landmark_positions = np.array([observation[4 + agent_index * 2], observation[5 + agent_index * 2]])
            # landmark_positions = np.array([observation[4 ], observation[5]])
            # landmark_positions = find_closest_landmark(agent_pos, [[observation[i*2],observation[i*2+1]] for i,p in enumerate(observation[4:9])])            
            action = -1
            # print(observation)
            # predicted_position = agent_pos + agent_speed
            
            dist_agents = distance([0,0] + agent_speed, [observation[-4], observation[-3]])
            # dist_agents = distance([0,0], [observation[-4], observation[-3]])
                        
            if dist_agents < 5:
                action = direction_to_avoid_target([0,0], agent_speed ,[observation[-4], observation[-3]])
                print(dist_agents, "Avoiding" , action)

            if action == -1:
                action = choose_actionPID(agent_index, landmark_positions, agent_speed)
            #action = 0

        env.step(action)
        if Spread_Config['render_mode'] == 'human':
            clock.tick(desired_fps)
    
    rewards.append(reward_run)  

        
env.close()         
print(rewards)
print(np.sum(rewards) / runs)
print(np.std(rewards))
print(np.max(rewards))
print(np.min(rewards))


