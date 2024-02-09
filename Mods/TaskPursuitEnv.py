from ssl import get_default_verify_paths
from pettingzoo.sisl import pursuit_v4
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
    'chase_evader': [1, 0, 0],
    'coordinate': [0, 1, 0],    
    'explore': [0, 0, 1],
    'stay': [0, 0, 0]
}

class refPoint:
    def __init__(self, position):
        self.current_pos = position

class Task:
    _id_counter = 0  # Class variable for generating unique IDs

    def __init__(self, task_type, target_object, forced_action = 4, num_agents=0):
        self.id = Task._id_counter
        # print(self.id, " - ", task_type)
        Task._id_counter += 1
        self.type = task_type
        self.target_object = target_object        

        self.forced_action = forced_action
        #self.distance = distance
        
        self.slot_offset = np.array([[-1,0], [1,0], [0,1], [0,-1]])
        self.slots = [0, 0, 0, 0]

        self.refExp = np.array([[3, 3], [3, 13], [13, 3], [13, 13]])
            

    def calculate_default_action(self):
        # Return a default action (e.g., stay)
        return 4  # Stay    


    
    def find_closest_point(self, my_position, points_array):
        
        # Calculate the differences between each point and your position
        differences = points_array - my_position
        
        # Calculate the Euclidean distance for each point
        distances = np.sqrt(np.sum(differences**2, axis=1))
        
        # Find the index of the minimum distance
        closest_point_index = np.argmin(distances)
        
        # Return the closest point
        return points_array[closest_point_index]
    
    def determine_desired_action(self, agent_position, slot, wall_channel):
        """
        Determine the action to take based on the task type and agent's position.
        """
        if self.type == 'chase_evader':
            # Task to chase an evader: Move towards the evader's position            
            action = self.calculate_direction_to_target(agent_position, self.target_object.current_pos + self.slot_offset[slot])
            
            if action != 4 and wall_channel[3 + self.slot_offset[action][1], 3 + self.slot_offset[action][0]] == 1 :
                slot = self.get_newSlot(agent_position, slot, exclusion = slot)
                action = self.calculate_direction_to_target(agent_position, self.target_object.current_pos + self.slot_offset[slot])

            return action
        
        elif self.type == 'coordinate':
            # Task to coordinate with another agent: Move towards the other agent's position
            return self.calculate_direction_to_target(agent_position, self.target_object.current_pos + self.slot_offset[slot])
        
        elif self.type == 'explore':
            
            exp_pos = None
            # Task to explore: This could be a predefined direction or a random action
            if agent_position[0] <= 3: 
                
                if agent_position[1] < 13:
                     exp_pos = np.array([3,13])
                else:
                     exp_pos = np.array([13,13])
            
            if agent_position[0] >= 13: 
                
                if agent_position[1] > 3:
                    exp_pos = np.array([13,3])
                else:
                    exp_pos = np.array([3,3])

            if exp_pos is None:
                if agent_position[0] > 3 and agent_position[1] <= 3: 
                    exp_pos = np.array([3,3])
                
                if agent_position[0] > 3 and agent_position[1] >= 13: 
                    exp_pos = np.array([13,13])


            if exp_pos is None:
                exp_pos = self.find_closest_point(agent_position, self.refExp)
            
            direction =  self.calculate_direction_to_target(agent_position, exp_pos)
            return direction
            # return self.calculate_direction_to_target(np.array([0,0]), self.target_object.current_pos[::-1] )
        
        else:
            # Default action (e.g., 'stay')
            return self.calculate_default_action()
    
    
    #Randonly select an more empty slot
    def get_newSlot(self, agent_position, current_slot, exclusion = 99):
                                
        
        #exclusions = self.check_slots(wall_channel)
        
        slots = [slot if idx != exclusion else 999 for idx,slot in enumerate(self.slots)]
                            
        minV = min(slots) 
        
        dists = [self.get_dist(agent_position, self.target_object.current_pos + self.slot_offset[i]) if slot == minV else 999 for i,slot in enumerate(slots)]
        
        #occurs = [index for index, element in enumerate(self.slots) if element == minV]                
        #idxMin = np.random.choice(occurs)
        idxMin = dists.index(min(dists))

        if exclusion == 99:
            self.slots[idxMin] += 1

        return idxMin
    
    def update_slots(self):        
        
        return 0
    
    def check_slots(self, wall_layer):
                
        #need to invert slots considering wall is inverted x,y
        wall_check = [wall_layer[ 1, 0] == 1, # Above center
                      wall_layer[ 1, 2] == 1, # Below center
                      wall_layer[ 2, 1] == 1, # Left of center
                      wall_layer[ 0, 1] == 1  # Right of center
                    ]

        
        return wall_check


    
    def calculate_direction_to_target(self, agent_position, target_position):
        """
        Calculate the direction from the agent's position to the target's position.
        """
        dx = target_position[0] - agent_position[0]
        dy = target_position[1] - agent_position[1]

        if dx == 0 and dy == 0:
            return 4 #self.random_direction()


        if abs(dx) > abs(dy):
            # Move in x direction
            return 0 if dx < 0 else 1  # Left (0) if dx is negative, Right (1) otherwise
        elif abs(dx) < abs(dy):
            # Move in y direction
            return 3 if dy < 0 else 2  # Up (2) if dy is negative, Down (3) otherwise
        else:
            #case the distances are the same random choice hor or vert move
            options = [0 if dx < 0 else 1 , 3 if dy < 0 else 2]
            return np.random.choice(options)

        
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

class TaskPursuitEnv(pursuit_v4.raw_env):
    
    def __init__(self, *args, **kwargs):
                
        EzPickle.__init__(self, *args, **kwargs)
                
        self.env = _env(*args, **kwargs)        

        
        self.fps = 5
        self.render_mode = kwargs.get("render_mode")
        if self.render_mode  == "human":
            pygame.init()

                
        self.agents = ["pursuer_" + str(a) for a in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)   

        self.removed_evader = 0   
          
        
        #Env modification for Task based policy
        self.max_tasks = 30  # Maximum number of tasks
        self.env.act_space = spaces.Discrete(self.max_tasks)
        self.env.action_space = [self.env.act_space for _ in range(self.env.n_pursuers)]
        self.env.action_spaces = dict(zip(self.agents, self.env.action_space))

        # spaces
        self.n_act_agents = self.env.act_dims[0]
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.steps = 0
        self.closed = False

        self.refGeneric = refPoint([0,0])
        
        self.exploreRefs = [refPoint(pos) for pos in  np.array([[-2,0], [2,0], [0,2], [0,-2]])]
        
        # self.tasks_explore = [ Task("explore", refExplore, forced_action = i) for i,refExplore in enumerate(self.exploreRefs) ]
        self.tasks_explore = [ Task("explore", self.refGeneric, forced_action = 4)] 
        
        self.tasks_basic = [Task('stay', self.refGeneric, forced_action = 4)] 

        self.tasks_evaders = []
        self.tasks_allies = []
        self.last_tasks = {agent : [] for agent in self.agents}
        self.last_len_tasks = {agent : [] for agent in self.agents}

        self.task_historic = [0 for _ in self.agents]
        self.action_historic = [[0] * 5 for _ in self.agents]
        self.tasks = []

        self.last_actions = [(self.tasks_basic[0], self.tasks_basic[0].forced_action, -1, 1 ,0 ) for _ in self.agents]
        self.raw_observations = [None for _ in self.agents]

        # self.last_actions_memory = [ deque([0] * 5, maxlen=5) for _ in self.agents]

        self.tasks_map = {}
        self.allocation_table = []


    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        # Call the base environment's reset
        super().reset(seed=seed, options=options)
         
        # tasks
        self.tasks_evaders = []
        self.tasks_allies = []
        self.tasks_map = {}
        self.allocation_table = []

        self.tasks = self.generate_tasks()
        self.update_tasks()
       # self.tasks_explore = [ Task("explore", refExplore, forced_action = i) for i,refExplore in enumerate(self.exploreRefs) ]
        self.tasks_explore = [ Task("explore", self.refGeneric, forced_action = 4)] 
        self.tasks_basic = [Task('stay', self.refGeneric, forced_action = 4)] 

        self.last_tasks = {agent : [] for agent in self.agents}
        self.last_len_tasks = {agent : [] for agent in self.agents}
        
        self.last_actions = [[self.tasks_basic[0], self.tasks_basic[0].forced_action, -1, 1, 0] for _ in self.agents]
                
        self.task_historic = [0 for _ in self.agents]
        self.action_historic = [[0] * 5 for _ in self.agents]

        self.raw_observations = [None for _ in self.agents]
        
    def is_valid_explore_task(self,task, layer):
            
            # Calculate the target position for the explore task
            target_position = [3 + task.target_object.current_pos[0], 3 + task.target_object.current_pos[1]] 
            
            #print(layer[target_position[1], target_position[0]])

            # Check if the target position is a wall
            return layer[target_position[0], target_position[1]] == 0
    
    
    def observe(self, agent):
        
        current_obs = super().observe(agent)

    
        # print("Observation: ",current_obs)
        
        if np.sum(self.env.evaders_gone) != self.removed_evader:
            
            self.removed_evader = np.sum(self.env.evaders_gone)
            self.update_tasks() 
            # print("task_updated")


        pusuer_idx = agent.split("_")[-1]
        pusuer_idx = int(pusuer_idx)
        pursuer_obj = self.env.pursuers[pusuer_idx]

        self.raw_observations[pusuer_idx]  = current_obs

        #agent_positions = self.get_agent_positions()
        agent_position = self.env.pursuer_layer.get_position(pusuer_idx)

        # Calculate the bounds of the observation box
        range_ref = (self.env.obs_range-1)/2
        x_lower_bound = agent_position[0] - range_ref
        x_upper_bound = agent_position[0] + range_ref
        y_lower_bound = agent_position[1] - range_ref
        y_upper_bound = agent_position[1] + range_ref
        
        wall_channel = current_obs[..., 0]  # Assuming the first channel represents walls        
        allies_channel = current_obs[...,1]  # Assuming the first channel represents walls        
        enemies_channel = current_obs[...,2]  # Assuming the first channel represents walls        
               
        last_tasks = []               
        
        if self.last_actions[pusuer_idx][0].type == "explore" and sum(sum(enemies_channel)) == 0:
            
            current_task = self.last_actions[pusuer_idx][0]
            
            if self.task_historic[pusuer_idx] <= 1:
                if self.is_valid_explore_task(current_task, wall_channel):                    
                    last_tasks = [current_task]
                
                
        # Filter tasks within the observation box
        if last_tasks == []:
            tasks_in_range = [task for task in self.tasks if x_lower_bound <= task.target_object.current_pos[0] <= x_upper_bound and y_lower_bound <= task.target_object.current_pos[1] <= y_upper_bound and task.target_object != pursuer_obj]
            
            # Filter out explore tasks that lead to a wall
            tasks_to_explore = [task for task in self.tasks_explore if self.is_valid_explore_task(task, wall_channel)] 

            random.shuffle(tasks_to_explore)


            last_tasks = tasks_in_range + tasks_to_explore
        
        self.last_len_tasks[agent] = len(last_tasks)
        
        mask = [True for _ in last_tasks]
        mask.extend([False] * (self.max_tasks - len(last_tasks)))
                
        #Pad or truncate the task list to ensure a fixed number of tasks
        if len(last_tasks) < self.max_tasks:
            last_tasks.extend([self.tasks_basic[0] for _ in range( self.max_tasks - len(last_tasks) )])
            # last_tasks.extend([np.random.choice(self.tasks_explore) for _ in range( self.max_tasks - len(last_tasks) )])
        else:
            last_tasks = self.last_tasks[agent][:self.max_tasks]

        self.last_tasks[agent] = last_tasks
        # Convert tasks to tensor
                      
        task_features = [self.get_feature_vector(pusuer_idx, current_obs,  agent_position, task) for task in last_tasks]
        
        # task_tensor = torch.tensor(task_features, dtype=torch.float32).to("cuda")
        
        self.infos[agent]["mask"] = mask
        observation = {"observation" : task_features, "action_mask" : mask, "info" : mask}
        
        
        return observation 
    
    
    def get_one_hot(self, task):
        return TASK_TYPES[task.type]
    
    def get_feature_vector(self, pusuer_idx, current_obs, agent_position, task):
        
        if task.type == "explore":
            channel = 0
        elif task.type == "coordinate":
            channel = 1        
        else:
            channel = 2

        # stats = self.calculate_statistics( current_obs, channel )
        
        #historic = 0 if self.last_actions[pusuer_idx][0] != task else self.task_historic[pusuer_idx] 

        task_type_one_hot = self.get_one_hot(task)  # One-hot encoding of task type
        
        if task.type != 'explore':
            task_pos = task.target_object.current_pos
        else:
            task_pos = np.array([ agent_position[0] + task.target_object.current_pos[1], agent_position[1] + task.target_object.current_pos[0]])
        
        feature_vector = task_type_one_hot +  list(task.getDistSinCos(agent_position, task_pos)) +\
                        [                             
                           agent_position[0]/15, 
                           agent_position[1]/15,
                           sum(task.slots)/4,
                           #historic / 5,
                           #aliies_in_sight,
                        ] #+ self.action_historic[pusuer_idx]#+ stats
        
        
        return feature_vector

    
    def calculate_statistics(self, observation, channel):
        # Supondo que a observação seja uma matriz NumPy ou tensor PyTorch de forma [7, 7, 3]
        stats = []
        
        # Exemplo: suponha que o primeiro canal representa inimigos
        selected_channel = observation[..., channel]
        #print(enemy_channel)

        # Contagem de inimigos
        enemy_count = np.sum(selected_channel) / 4

        # Densidade de inimigos
        enemy_density = enemy_count / (7 * 7) * 7

        # Centro de massa dos inimigos (média da posição)
        enemy_positions = np.argwhere(selected_channel)
#         print(enemy_positions)
        if len(enemy_positions) > 0:
            center_of_mass = np.mean(enemy_positions, axis=0)
        else:
            center_of_mass = [3, 3]  # Ou outro valor representativo quando não houver inimigos

        dist = self.distance(center_of_mass, [3,3])
        # Adicionando as estatísticas à lista
        stats.extend([center_of_mass[0], center_of_mass[1], dist, enemy_density])

        return stats
    
    def step(self, action):
                                
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection

        pusuer_idx = agent.split("_")[-1]
        pusuer_idx = int(pusuer_idx)

                   
        wall_channel = self.raw_observations[pusuer_idx][..., 0]  # Assuming the first channel represents walls        
        # allies_channel = current_obs[...,1]  # Assuming the first channel represents walls        
        # enemies_channel = current_obs[...,2]  # Assuming the first channel represents walls    
        
        agent_position = self.env.pursuer_layer.get_position(pusuer_idx)        
               
        task = self.last_tasks[agent][action]

        last_task_data = self.last_actions[pusuer_idx]                                          
        
        if task != self.last_actions[pusuer_idx][0]:                                                                                    
                                    
            last_task_data[0].slots[last_task_data[2]] -= 1

            slot = task.get_newSlot(agent_position, last_task_data[2])
            self.last_actions[pusuer_idx] = [task, action, slot, self.last_len_tasks[agent], 1]
            self.task_historic[pusuer_idx] = 1
        
        else:
                                                
            #self.last_actions[pusuer_idx][2] = slot            
            self.task_historic[pusuer_idx] += 1
            self.last_actions[pusuer_idx][4] += 1
              
        
        action = self.convert_task2action(self.last_tasks[agent], action , agent_position, self.last_actions[pusuer_idx][2], wall_channel) 
                        
        self.env.step(
            action, self.agent_name_mapping[agent], self._agent_selector.is_last()
        )

        #Acreate a memory of actions
        self.action_historic[pusuer_idx][action] += 1         
        for a in range(5):
            if a != action:
                self.action_historic[pusuer_idx][a] /= 2
                                             
            
        for k in self.terminations:
            if self.env.frames >= self.env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = self.env.is_terminal
        for k in self.agents:
            self.rewards[k] = self.env.latest_reward_state[self.agent_name_mapping[k]]
        self.steps += 1

        
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self._agent_selector.is_last() and self.render_mode == "html":
            self.print_grid()  

        if self.render_mode == "human":
            self.render()

    
    def generate_tasks(self):
                        
        for evader in self.env.evaders:

            new_task = Task('chase_evader', evader)
            self.tasks_evaders.append(new_task)
            self.tasks_map[new_task.id] = new_task           
            
        for pursuer in self.env.pursuers:
            
            new_task = Task('coordinate', pursuer)
            self.tasks_allies.append(new_task)
            self.tasks_map[new_task.id] = new_task  
                            

    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
   
    def convert_task2action(self, last_tasks, task_num, agent_position, slot, wall_channel):
        # Get the selected task
        
        selected_task = last_tasks[task_num] if task_num < len(last_tasks) else None

        if selected_task is None:
            # Default action if no task is selected or if the task number is out of range
            return 4  # Assuming 4 is the default 'stay' action        

        # Determine the desired action based on the task type
        return selected_task.determine_desired_action(agent_position, slot, wall_channel)

    def update_tasks(self):
                
        active_evaders = [self.tasks_evaders[i] for i in range(len(self.tasks_evaders)) if not self.env.evaders_gone[i]]
        
        self.tasks = active_evaders + self.tasks_allies #+ self.tasks_basic 

        # self.tasks = active_evaders + self.tasks_basic 
        # self.tasks = self.tasks_basic.copy()

 
    def print_grid(self):
        clear_output(wait=True)  # Clear the output of the current cell

        grid = [[' ' for _ in range(self.env.x_size)] for _ in range(self.env.y_size)]

        # Place evaders and pursuers on the grid    
        for evader in self.env.evaders:            
            x, y = evader.current_pos
            # Color evaders in red
            grid[y][x] = f'<span style="color: red;">O</span>'

        # Place pursuers on the grid
        for i, pursuer in enumerate(self.env.pursuers):
            x, y = pursuer.current_pos
            # Color pursuers in blue
            grid[y][x] = f'<span style="color: blue;">{i}</span>'

        # Convert the grid to an HTML table with enhanced visibility
        grid_html = '<table style="border-collapse: collapse;">'
        for row in grid:
            grid_html += '<tr>' + ''.join([f'<td style="width: 16px; height: 16px; text-align: center; border: 1px solid black;">{cell if cell.strip() != "" else "&nbsp;"}</td>' for cell in row]) + '</tr>'
        grid_html += '</table>'

        # Convert last_actions to an HTML table
        actions_html = ''
        if hasattr(self, 'last_actions'):
            # Extract task type, action, and slot from each entry in last_actions
            formatted_actions = [(task.type, task.id, action, len_last, task.slots,  hist) for task, action, slot, len_last, hist in self.last_actions]
            
            # Create a DataFrame from the formatted list
            actions_df = pd.DataFrame(formatted_actions, columns=['Task Type', 'id', 'action', 'nTasks', 'Slots' , 'Hist'])
            actions_html = '<td>' + actions_df.to_html(index=True, border=0) + '</td>'

        # Display grid and actions table side by side within a parent table
        parent_table_html = f'<table><tr><td style="vertical-align: top;">{grid_html}</td>{actions_html}</tr></table>'
        display(HTML(parent_table_html))
        time.sleep(1/self.fps)
        
        


def env(**kwargs):
    environment = TaskPursuitEnv(**kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment

