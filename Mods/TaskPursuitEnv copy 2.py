from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils import wrappers
import numpy as np
import torch
import time
import random

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
                
        self.slot_offset = np.array([[1,0], [0,1], [-1,0], [0,-1]])
        self.slots = [0, 0, 0, 0]
        
        
        
   
    def get_one_hot(self):
        return TASK_TYPES[self.type]
    
    def get_feature_vector(self, agent_position):
        # Example feature vector; adjust according to your task attributes
        task_type_one_hot = self.get_one_hot()  # One-hot encoding of task type
        # desired_action = self.determine_desired_action(agent_position)
        
        feature_vector = task_type_one_hot + list(self.target_object.current_pos) + [sum(self.slots)/4] 
        return feature_vector
    

    def calculate_default_action(self):
        # Return a default action (e.g., stay)
        return 4  # Stay

 
    def determine_desired_action(self, agent_position, slot):
        """
        Determine the action to take based on the task type and agent's position.
        """
        if self.type == 'chase_evader':
            # Task to chase an evader: Move towards the evader's position            
            return self.calculate_direction_to_target(agent_position, self.target_object.current_pos + self.slot_offset[slot])
        
        elif self.type == 'coordinate':
            # Task to coordinate with another agent: Move towards the other agent's position
            return self.calculate_direction_to_target(agent_position, self.target_object.current_pos )#+ self.slot_offset[slot])
        
        elif self.type == 'explore':
            # Task to explore: This could be a predefined direction or a random action
            return self.forced_action if self.forced_action != 4 else self.random_direction()
        
        else:
            # Default action (e.g., 'stay')
            return self.calculate_default_action()
    
    def get_newSlot(self, agent_position):

        #create a function that select between the 4 possible slots considering the closest and the ocupied one
        minV = min(self.slots)
        occurs = [index for index, element in enumerate(self.slots) if element == minV]
        idxMin = np.random.choice(occurs)
        return idxMin


    
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

        self.render_mode = kwargs.get("render_mode")
        
        if self.render == "human":
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
        
        self.tasks_explore = [ Task("explore", self.refGeneric, forced_action = i) for i in range(4) ]
        self.tasks_basic = [Task('stay', self.refGeneric, forced_action = 4)] 

        self.tasks_evaders = []
        self.tasks_allies = []
        self.last_tasks = {agent : [] for agent in self.agents}
        self.last_len_tasks = {agent : [] for agent in self.agents}
        self.tasks_inertia = {agent : [] for agent in self.agents}
        
        self.tasks = []

        self.last_actions = [(self.tasks_basic[0], self.tasks_basic[0].forced_action, 0 ) for _ in self.agents]

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

        self.last_tasks = {agent : [] for agent in self.agents}
        self.last_len_tasks = {agent : [] for agent in self.agents}
        self.last_actions = [(self.tasks_basic[0], self.tasks_basic[0].forced_action, 0) for _ in self.agents]

    def observe(self, agent):
        
        current_obs = super().observe(agent)
        
        if np.sum(self.env.evaders_gone) != self.removed_evader:
            
            self.removed_evader = np.sum(self.env.evaders_gone)
            self.update_tasks() 
            # print("task_updated")


        pusuer_idx = agent.split("_")[-1]
        pusuer_idx = int(pusuer_idx)
        pursuer_obj = self.env.pursuers[pusuer_idx]

        #agent_positions = self.get_agent_positions()
        agent_position = self.env.pursuer_layer.get_position(pusuer_idx)

        # Calculate the bounds of the observation box
        range_ref = (self.env.obs_range-1)/2
        x_lower_bound = agent_position[0] - range_ref
        x_upper_bound = agent_position[0] + range_ref
        y_lower_bound = agent_position[1] - range_ref
        y_upper_bound = agent_position[1] + range_ref

        # Filter tasks within the observation box
        tasks_in_range = [task for task in self.tasks if x_lower_bound <= task.target_object.current_pos[0] <= x_upper_bound and y_lower_bound <= task.target_object.current_pos[1] <= y_upper_bound and task.target_object != pursuer_obj]

        last_tasks = tasks_in_range + self.tasks_explore
        self.last_len_tasks[agent] = len(last_tasks)

        #Pad or truncate the task list to ensure a fixed number of tasks
        if len(last_tasks) < self.max_tasks:
#             last_tasks.extend([self.tasks_basic[0] for _ in range( self.max_tasks - len(last_tasks) )])
            last_tasks.extend([np.random.choice(self.tasks_explore) for _ in range( self.max_tasks - len(last_tasks) )])
        else:
            last_tasks = self.last_tasks[agent][:self.max_tasks]

        self.last_tasks[agent] = last_tasks
        # Convert tasks to tensor
        task_features = [task.get_feature_vector(agent_position) for task in self.last_tasks[agent]]
        task_tensor = torch.tensor(task_features, dtype=torch.float32).to("cpu")
        
        return task_tensor

    
    def step(self, action):
                
        debug = False
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection

        pusuer_idx = agent.split("_")[-1]
        pusuer_idx = int(pusuer_idx)        
        
        agent_position = self.env.pursuer_layer.get_position(pusuer_idx)        
               
        task = self.last_tasks[agent][action]
                          
        if task != self.last_actions[pusuer_idx][0]:
                                                
            slot = task.get_newSlot(agent_position)
            task.slots[slot] += 1 

            last_task_data = self.last_actions[pusuer_idx]
            last_task_data[0].slots[last_task_data[2]] -= 1
                        
            self.last_actions[pusuer_idx] = (task, action, slot, self.last_len_tasks[agent])


        else:
            
            if random.random() > 0.80:
                
                last_task_data = self.last_actions[pusuer_idx]
                last_task_data[0].slots[last_task_data[2]] -= 1

                slot = task.get_newSlot(agent_position)
                task.slots[slot] += 1 
            
                self.last_actions[pusuer_idx] = (task, action, slot, self.last_len_tasks[agent])
            else:
                slot = self.last_actions[pusuer_idx][2]
                self.last_actions[pusuer_idx] = (task, action, slot, self.last_len_tasks[agent])
                    
            
        
        # print(self.last_tasks[agent])
        action = self.convert_task2action(self.last_tasks[agent], action , agent_position, self.last_actions[pusuer_idx][2])        
        
        self.env.step(
            action, self.agent_name_mapping[agent], self._agent_selector.is_last()
        )
                     
        if self._agent_selector.is_last() and debug:
            self.print_grid()                        
            
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
   
    def convert_task2action(self, last_tasks, task_num, agent_position, slot):
        # Get the selected task
        
        selected_task = last_tasks[task_num] if task_num < len(last_tasks) else None

        if selected_task is None:
            # Default action if no task is selected or if the task number is out of range
            return 4  # Assuming 4 is the default 'stay' action        

        # Determine the desired action based on the task type
        return selected_task.determine_desired_action(agent_position, slot)

    def update_tasks(self):
                
        active_evaders = [self.tasks_evaders[i] for i in range(len(self.tasks_evaders)) if not self.env.evaders_gone[i]]
        
        self.tasks = active_evaders + self.tasks_allies

#         self.tasks = active_evaders + self.tasks_basic 

from IPython.display import HTML, clear_output
import time
import pandas as pd

class TaskPursuitEnv(pursuit_v4.raw_env):
    # ... [existing code] ...

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
            formatted_actions = [(task.type, action, task.slots, len_last) for task, action, slot, len_last in self.last_actions]
            
            # Create a DataFrame from the formatted list
            actions_df = pd.DataFrame(formatted_actions, columns=['Task Type', 'Action', 'Slots', 'nTasks'])
            actions_html = '<td>' + actions_df.to_html(index=True, border=0) + '</td>'

        # Display grid and actions table side by side within a parent table
        parent_table_html = f'<table><tr><td style="vertical-align: top;">{grid_html}</td>{actions_html}</tr></table>'
        display(HTML(parent_table_html))
        time.sleep(0.3)

        
        


def env(**kwargs):
    environment = TaskPursuitEnv(**kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment

