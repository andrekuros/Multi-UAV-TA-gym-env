from ssl import get_default_verify_paths
from turtle import distance
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils import wrappers
import numpy as np
import torch
import time
import math
import random
from collections import deque


from gymnasium import spaces

import pandas as pd
from IPython.display import display, clear_output
from IPython.display import HTML

import pygame
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.sisl.pursuit.manual_policy import ManualPolicy
from pettingzoo.sisl.pursuit.pursuit_base import Pursuit as _env
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn



__all__ = ["ManualPolicy", "env", "parallel_env", "raw_env"]


TASK_TYPES = {
    'go_landmark': [1, 0, 0],
    'avoid_colision': [0, 1, 0],    
    'explore': [0, 0, 1],
    'stay': [0, 0, 0]
}

class state:
    def __init__(self, position):
        self.p_pos = position

class refPoint:
    def __init__(self, position):
        self.current_pos = position
        self.state = state(position)

class Task:
    _id_counter = 0  # Class variable for generating unique IDs

    def __init__(self, task_type, target_object, forced_action = 0, num_agents=0):
        self.id = Task._id_counter
        # print(self.id, " - ", task_type)
        Task._id_counter += 1
        self.type = task_type
        self.target_object = target_object        

        self.forced_action = forced_action
        #self.distance = distance
                
        self.slots = [0]

        self.refExp = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
            

    def calculate_default_action(self):
        # Return a default action (e.g., stay)
        return 0  # Stay    

        
    def determine_desired_action(self, agent_position, agent_speed):
        """
        Determine the action to take based on the task type and agent's position.
        """
        if self.type == 'go_landmark':            
            
            action = self.direction_to_target(agent_position, self.target_object.state.p_pos)                       
            return action
        
        elif self.type == 'avoid_colision':

            action = self.direction_to_avoid_target(agent_position, agent_speed, self.target_object.state.p_pos)
            return action
        
        elif self.type == 'explore':
                                    
            action =  self.direction_to_target(agent_position, self.target_object.state.p_pos)
            return action
                                
        else:            
            return self.calculate_default_action()
    
            
    def direction_to_target(self, agent_position, target_position):
        """
        Calculate the direction from the agent's position to the target's position.
        """
        dx = target_position[0] - agent_position[0]
        dy = target_position[1] - agent_position[1]

        if dx == 0 and dy == 0:
            return 4 #self.random_direction()


        if abs(dx) > abs(dy):
            # Move in x direction
            return 1 if dx < 0 else 2  # Left (1) if dx is negative, Right (2) otherwise
        elif abs(dx) < abs(dy):
            # Move in y direction
            return 3 if dy < 0 else 4  # Up (3) if dy is negative, Down (4) otherwise
        else:
            #case the distances are the same random choice hor or vert move
            options = [1 if dx < 0 else 2 , 3 if dy < 0 else 4]
            return np.random.choice(options)
        

    def direction_to_avoid_target(self,agent_position, agent_speed, target_position):
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

        
    def getDistSinCos(self, point_a, point_b):
        # Unpack the points
        x1, y1 = point_a
        x2, y2 = point_b
        
        # Calculate the differences
        dx = x2 - x1
        dy = y2 - y1

        # Euclidean distance
        distance = math.sqrt(dx**2 + dy**2)

        # Calculate the angle in radians
        angle = math.atan2(dy, dx)

        # Sine and Cosine of the angle
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)

        return distance, sin_angle, cos_angle
    
    def get_dist(self, posA, posB):
        return math.sqrt((posB[0] - posA[0])**2 + (posB[1] - posA[1])**2)
    
    def distance_bearing(self, pos1, pos2):
        """Calculate Euclidean distance between two points."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def random_direction(self):
        """
        Return a random direction.
        """
        # Assuming actions 0, 1, 2, 3 correspond to Up, Down, Left, Right
        return np.random.choice([0, 1, 2, 3])
    
    def increment_slot(self):
        self.slots[0] += 1

    def decrement_slot(self):
        if self.slots[0] > 0:
            self.slots[0] -= 1
    
def get_dist(posA, posB):
        return math.sqrt((posB[0] - posA[0])**2 + (posB[1] - posA[1])**2)

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


def choose_actionPID(pidX, pidY, landmark_positions, agent_speed):
    threshold = 0.01
    pid_output_x = pidX.update(landmark_positions[0], agent_speed[0])
    pid_output_y = pidY.update(landmark_positions[1], agent_speed[1])
      
    threshold = 0.01
   
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

class TaskSpreadEnv(simple_spread_v3.raw_env):
    
    def __init__(self, *args, **kwargs):
                        
        super().__init__(*args, **kwargs)

        n_agents = len(self.world.agents)
                                         
        self.agents = ["agent_" + str(a) for a in range(n_agents)]
        self.possible_agents = self.agents[:]
        #Env modification for Task based policy
        self.max_tasks = n_agents #* 2 - 1  + 4 # Maximum number of tasks
        self.act_space = spaces.Discrete(self.max_tasks)
        self._action_space = [self.act_space for _ in range(n_agents)]
        self.action_spaces = dict(zip(self.agents, self._action_space))

        # spaces
        self.n_act_agents = n_agents
        
        # self.observation_spaces = dict(zip(self.world.agents, self.observation_space))                        
        
        self.refGeneric = refPoint([0,0])
                
        self.exploreRefs = [refPoint(pos) for pos in  np.array([[-1,0], [1,0], [0,1], [0,-1]])]        
        self.tasks_explore = [ Task("explore", refExplore, forced_action = i) for i,refExplore in enumerate(self.exploreRefs) ]
        # self.tasks_explore = [ Task("explore", self.refGeneric, forced_action = 4)]         
        
        self.tasks_basic = [Task('stay', self.refGeneric, forced_action = 0)] 

        self.last_tasks = {agent : [] for agent in self.agents}

        self.tasks_landmarks = []
        self.tasks_allies = []
        
        self.tasks = []
        self.tasks_map = {}

        self.pid_controllers_x = [PIDController(Kp=0.5, Ki=-0.0000, Kd=-0.0, setpoint=0) for _ in range(self.n_act_agents)]
        self.pid_controllers_y = [PIDController(Kp=0.5, Ki=-0.0000, Kd=-0.0, setpoint=0) for _ in range(self.n_act_agents)]

        self.current_task = {agent: None for agent in self.agents}
       

    def reset(self, seed=None, options=None):
                
        # Call the base environment's reset
        super().reset(seed=seed, options=options)
             
        self.tasks_landmarks = []
        self.tasks_allies = []        
        
        self.tasks = []   
        self.tasks_map = {}

        self.last_tasks = {agent : [] for agent in self.agents}
        
        self.tasks = self.generate_tasks()    
        self.update_tasks()

        self.pid_controllers_x = [PIDController(Kp=0.5, Ki=-0.0000, Kd=-0.0, setpoint=0) for _ in range(self.n_act_agents)]
        self.pid_controllers_y = [PIDController(Kp=0.5, Ki=-0.0000, Kd=-0.0, setpoint=0) for _ in range(self.n_act_agents)]

        self.current_task = {agent: None for agent in self.agents}


    def action_space(self, agent):
        return self.action_spaces[agent]   

    def observe(self, agent):
               
        current_obs = super().observe(agent)

        agent_idx = agent.split("_")[-1]
        agent_idx = int(agent_idx)
        agent_obj = self.world.agents[agent_idx]
       
        available_tasks = [task for task in self.tasks if task.target_object != agent_obj]
        task_features = [self.get_feature_vector(agent_obj, agent, task) for task in available_tasks]
        
        self.last_tasks[agent] = available_tasks
        
        #self.infos[agent]["mask"] = mask
        observation = {"observation" : task_features}#, "action_mask" : mask, "info" : mask}

        return observation


    
    def get_one_hot(self, task):
        return TASK_TYPES[task.type]
    
    def get_feature_vector(self, agent, agent_name,  task):
        
    #     if task.type == "explore":
    #         channel = 0
    #     elif task.type == "coordinate":
    #         channel = 1        
    #     else:
    #         channel = 2

    #     # stats = self.calculate_statistics( current_obs, channel )
        
    #     #historic = 0 if self.last_actions[pusuer_idx][0] != task else self.task_historic[pusuer_idx] 

        task_type_one_hot = self.get_one_hot(task)  # One-hot encoding of task type
        slot_count = [task.slots[0]]

        if self.last_tasks[agent_name] == task:
            slot_count = slot_count - 1

        
    #     if task.type != 'explore':
    #         task_pos = task.target_object.current_pos
    #     else:
    #         task_pos = np.array([ agent_position[0] + task.target_object.current_pos[1], agent_position[1] + task.target_object.current_pos[0]])
        
        # feature_vector = task_type_one_hot +  list(task.getDistSinCos(agent.state.p_pos, task.target_object.state.p_pos)) +\
        feature_vector = slot_count  + list([get_dist(agent.state.p_pos, task.target_object.state.p_pos)])#+\
    #                      [                             
    #                          agent.state.p_vel[0],
    #                          agent.state.p_vel[1],  
    # #                        agent_position[1]/15,
    # #                        sum(task.slots)/4,
    # #                        #historic / 5,
    # #                        #aliies_in_sight,
    #                      ] #+ self.action_historic[pusuer_idx]#+ stats
        
        
        return feature_vector
        
    
    def step(self, action):
        
        agent = self.agent_selection

        agent_idx = agent.split("_")[-1]
        agent_idx = int(agent_idx)
        agent_obj = self.world.agents[agent_idx]               
        
        selected_task = self.last_tasks[agent][action]

        # print(self.tasks[0].slots, self.tasks[1].slots)

        # If the agent selected a different task, update slots
        if self.current_task[agent] != selected_task:
            if self.current_task[agent] is not None:
                self.current_task[agent].decrement_slot()
            selected_task.increment_slot()
            self.current_task[agent] = selected_task    
                         
        if selected_task.type == "go_landmark":
            action = choose_actionPID(self.pid_controllers_x[agent_idx], self.pid_controllers_y[agent_idx], selected_task.target_object.state.p_pos - agent_obj.state.p_pos, agent_obj.state.p_vel)

        else:
            action = self.convert_task2action(agent_obj, selected_task ) 
            print("HAHAHA")

        super().step(action)
    

    def generate_tasks(self):
                        
        for landmark in self.world.landmarks:  # world.entities:
                                          
            new_task = Task('go_landmark', landmark)            
            self.tasks_landmarks.append(new_task)
            self.tasks_map[new_task.id] = new_task

        for agent in self.world.agents:  # world.entities:
                                          
            new_task = Task('avoid_colision', agent)            
            self.tasks_allies.append(new_task)
            self.tasks_map[new_task.id] = new_task               
                                            
    def convert_task2action(self, agent_obj, task):
                
        # Determine the desired action based on the task type
        return task.determine_desired_action(agent_obj.state.p_pos, agent_obj.state.p_vel)

    def update_tasks(self):
                
    #     active_evaders = [self.tasks_evaders[i] for i in range(len(self.tasks_evaders)) if not self.env.evaders_gone[i]]        
    
        self.tasks = self.tasks_landmarks #+ self.tasks_allies + self.tasks_explore

def env(**kwargs):
    environment = TaskSpreadEnv(**kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment

