from concurrent.futures import thread
from operator import length_hint
from typing import List, Optional
from ctypes import util
from email import utils
from pkgutil import extend_path
import gym
from gym import spaces
from gym.spaces import Dict, Discrete, MultiDiscrete, Box
import numpy as np
import random
import copy
import sys
import math

import matplotlib.pyplot as plt
import seaborn as sns
#from pygame_screen_recorder import pygame_screen_recorder as pgr

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.utils import agent_selector

from .DroneEnvComponents import UAV, SquareArea, Task, Obstacle, Threat
from .MultiDroneEnvData import SceneData 
from .MultiDroneEnvUtils import agentEnvOptions, EnvUtils, ACMIExporter

import pygame

MAX_INT = sys.maxsize
EPS = 1e-12


def env(config = None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    #internal_render_mode = config.render_mode if config.render_mode != "ansi" else "human"
    #env = raw_env(internal_render_mode, action_mode,  render_speed, max_time_steps, n_agents, n_tasks)
    env = MultiUAVEnv(config)
    # This wrapper is only for environments which print results to the terminal
    #if render_mode == "ansi":
    #    env = wrappers.CaptureStdoutWrapper(env)
    
    # this wrapper helps error handling for discrete action spaces
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    
    return env


def raw_env(config = None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = MultiUAVEnv(config)
    env = parallel_to_aec(env)
    return env

class MultiUAVEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multi_agent_env_v0"}
    
    def __init__(self, config=None ):
        
        super(MultiUAVEnv, self).__init__()        
        
        self.exporter = ACMIExporter()

        if config is None:
            self.config = agentEnvOptions()
        else:
            self.config = config            
                    
        self.fixed_seed = self.config.fixed_seed
        if self.fixed_seed != -1:
            self._seed = self.fixed_seed
        else:
            self._seed = 0

        self.rndGen = random.Random(self._seed)
        
        self.sceneData = SceneData()
        self.area_width = self.sceneData.GameArea[0]
        self.area_height = self.sceneData.GameArea[1]
        self.max_coord = max(self.area_height, self.area_width)
        self.bases = self.sceneData.Bases
        
        self.max_time_steps = self.config.max_time_steps
        self.time_steps = 0 
        self.simulation_frame_rate = self.config.simulation_frame_rate
        self.conclusion_time = self.max_time_steps + 1

        self.info = self.config.info

        self.render_speed = self.config.render_speed        
        self.render_enabled = self.config.render_speed != -1 #Render is activated if speed != -1
        self.render_mode = self.config.render_mode        
        
        self.action_mode = self.config.action_mode
        self.reached_tasks = set()
        
        # Define Agents
        self.agents_config = self.config.agents
        self.n_agents = sum(self.config.agents.values()) 
        self.random_init_pos = self.config.random_init_pos
        self.max_agents = 20                
        
        self.possible_agents = [] 
        for agent_type, n_agents in self.agents_config.items():
            for i in range(n_agents):                
                self.possible_agents.append(f'{agent_type[0]}_agent{i}')

        self.agent_selection = self.possible_agents[0] 
        self.agents_obj = None
        #self.possible_agents = ["agent" + str(i) for i in range(self.n_agents)]          
        
        #self.agent_name_mapping = dict(
        #    zip(self.possible_agents, list(range(len(self.possible_agents))))
        #)
        self.agent_by_name = {}

        #print(self.possible_agents)
        
        self.agents = self.possible_agents

        self.agent_selector = agent_selector(self.possible_agents)
        self.current_agent = self.agent_selector.next()
                
        self.max_tasks = 30 + 1
        self.n_tasks = sum(self.config.tasks.values()) + 1
        
        self.tasks_config = self.config.tasks               
        
        self.tasks: List[Optional[Task]] = []
        self.concluded_tasks: List[Optional[Task]] = []
        self.last_tasks_info = None
        self.task_idle = None

        self.multiple_tasks_per_agent = self.config.multiple_tasks_per_agent
        self.multiple_agents_per_task = self.config.multiple_agents_per_task
        
        #Define Threats
        self.threats = []  # List to store active threats
        self.threat_generation_probability = 0.7  / self.simulation_frame_rate * 0.02#TODO: Frame rate Adjust as needed        
        self.max_threats = self.config.max_threats
        self.n_threats = self.max_threats

        self.n_mission_areas = 3
        self.mission_areas = None
        
        #Dynamic Conditions        
        self.hidden_obstacles = self.config.hidden_obstacles        
        
        self.num_obstacles = self.config.num_obstacles        
        self.obstacles = None
        
        self.fail_rate = self.config.fail_rate         
        
        self.tasks_current_quality = None        
        self.allocation_table = None
                                               
        #Table with the quality of each agent for each task
        self.quality_table = None
                
        self.time_steps = 0
        self.total_distance = 0
        self.agent_distances = None
        self.agent_directions = None 

        self.F_Reward = 0 
        self.step_reward = 0

        self.event_list = []      
                
        self.previous_agents_positions = None
        self.previous_agents_positions = None
                        
        # Inicializar o Pygame
        self.screen = None      
        self.recrdr = None
        
        if self.render_enabled:
            import pygame
            pygame.init()
            self.clock = pygame.time.Clock()
            # Configurar o MovieWriter
            #self.recrdr = pgr("Test_Tessi.gif") # init recorder object
              
        self._observation_space = Dict({
            "agent_position": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "agent_state": Box(low=0, high=1, shape=(5,), dtype=np.float32),  # Assuming 5 possible states (0, 1, 2, 3 and 4)
            "agent_type": Box(low=0, high=1, shape=(6,), dtype=np.float32),  # One-hot encoded agent types  # Assuming 2 possible types
            "next_free_time": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "position_after_last_task": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            #"agent_relay_area": Box(low=0, high=max(self.area_width, self.area_height), shape=(2,), dtype=np.float32),           
            "tasks_info": Box(low=0, high=1, shape=((self.max_tasks ) * 3,), dtype=np.float32),
            "agents_info": Box(low=0, high=1, shape=((self.max_agents ) * 5,), dtype=np.float32),
            #"task_type": Discrete(2)  # Assuming 2 possible types
        }) 
        
        self.rewards = {agent : 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent : 0 for agent in self.possible_agents}
        self.terminations = {agent : False for agent in self.possible_agents}
        self.truncations = {agent : False for agent in self.possible_agents}
        self.infos = {agent : {} for agent in self.possible_agents}
        self.state = {agent : None for agent in self.possible_agents}
        
        self._observation_spaces = {
            agent: self._observation_space
            for agent in self.possible_agents
        }
        
        self.observations = { agent: { } for agent in self.possible_agents  }    

    def get_task_info(self, agent: UAV):
                
        task_values = [ {
                
                "id": task.id,
                "position": task.position / self.max_coord,                
                "status": task.status,                
                "current_reqs": task.currentReqs,                
                "alloc_reqs":  task.allocatedReqs,
                "init_time" : (task.initTime - self.time_steps) / self.max_time_steps,            
                "end_time" : (task.doneTime - self.time_steps) / self.max_time_steps
                }  for task in self.tasks
                if task.status != 2 #status = 2 is concluded 
            ]          
       

        mask = [True for _ in task_values]
        mask.extend([False] * (self.max_tasks - len(task_values)))

        # Pad the task_values array to match the maximum number of tasks
        task_values.extend([{"status": -1} for _ in range(self.max_tasks - len(task_values))])
        #task_values = np.array(task_values, dtype=np.float32)               
              
        return task_values, mask

    def get_agents_info(self):

        agents_values = []
         
        for agent in self.agents_obj:
            
            #One-hot expansion for the agent type
            #agents_values.extend(self._one_hot(agent.typeIdx, len(self.sceneData.self.UavTypes)))
                        
            agents_values.extend([
                agent.typeIdx,
                agent.next_free_position[0] / self.max_coord,
                agent.next_free_position[1] / self.max_coord,
                agent.next_free_time / self.max_time_steps,
                agent.currentCap2Task,
                agent.tasks[0].id
                ])
            
        #Pad with -1 if the number of agents is smaller than the max_agents
        agents_values.extend([-1] * (self._observation_spaces["agent0"]["agents_info"].shape[0] - len(agents_values)))
        
        agents_values = np.array(agents_values, dtype=np.float32)  

        return agents_values
                
    def observation_space(self, agent):
           return self._observation_spaces[agent]  
    
    def _generate_observations(self):
                        
        #agents_info = self.get_agents_info()        
        tasks_info, mask = self.get_task_info(self.agents_obj[0])        
                                                    
        self.observations = {
            agent.name : {
                #Add Own Data for relative features    
                "agent_type": agent.typeIdx,              
                "agent_position": agent.position / self.max_coord,                
                "agent_caps": agent.currentCap2Task,
                "agent_attack_cap": agent.attackCap / 4,
                "next_free_time": agent.next_free_time / self.max_time_steps,
                "position_after_last_task": agent.next_free_position / self.max_coord,                                
                "alloc_task": agent.tasks[0].id, 

                #Complete data for tasks and agents
                "tasks_info": tasks_info,
                "mask": mask#,   
                #"agents_info": agents_info,              
                
            }
            for agent in self.agents_obj
        }

        self.last_tasks_info = [task for task in self.tasks if task.status != 2]            

    #@functools.lru_cache(maxsize=None)
    def action_space(self, agent):
               
        #return Dict({agent:MultiDiscrete([5, 5, 5]) for agent in self.possible_agents})
        
        return Dict({agent: Box(low=0, high=1, shape=(self.max_tasks,), dtype=np.float32) for agent in self.possible_agents})
        #return Discrete(self.n_tasks )

    def last(self):
        """Return the last observations, rewards, and done status."""
        agent = self.agent_selection
        print("AG:",agent, "|", self.time_steps)
        assert agent
        observation = self.observe(agent) if self.observe else None        
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )
        #self._generate_observations()
        #return self.observations, self.rewards, self.terminations, self.truncations,self.infos

    def seed(self,seed):
        self.reset(seed = seed)
        #print("Call def_seed :", seed)

    def reset(self, seed = None , return_info = True, options={"options":1}):
               
        #print("Seed:", seed)
        if seed is None:                
            self._seed = random.randint(0, MAX_INT)
        else:
            self._seed = seed    
        
        if self.fixed_seed != -1:
            self._seed = self.fixed_seed           
                

        #print("Call reset_seed :", self._seed)
        self.rndAgentGen = random.Random(self._seed)
        self.rndObsGen = random.Random(self.rndAgentGen.randint(0, MAX_INT))
        self.rndTgtGen = random.Random(self.rndAgentGen.randint(0, MAX_INT))        
        self.rndMissionGen = random.Random(self.rndAgentGen.randint(0, MAX_INT))
        #print("Seed:",self._seed)
        #print (self.rndAgentGen.random(),self.rndObsGen.random(),self.rndTgtGen.random())
                                
        if self.agents_obj is not None:
           self.agents_obj.clear() 
        if self.tasks is not None:
            self.tasks.clear() 
        if self.obstacles is not None:
            self.obstacles.clear()
        
        
        self.conclusion_time = self.max_time_steps + 1
        self.F_Reward = 0 
        
        self.threats = []  # List to store active threats         
        self.threats: List[Optional[Threat]] = []
        self.n_threats = self.max_threats
                                  
                        
        #-------------------  Define Obstacles  -------------------#
        self.obstacles = []
        for _ in range(self.num_obstacles):
            size = self.rndObsGen.randint(30, 100)
            position = self.random_position(self.rndObsGen, obstacles = self.obstacles, own_range = size , contact_line = True)            
            self.obstacles.append(Obstacle(position, size))

                                                    
        #-------------------  Init Agents  -------------------#
        self.agents_obj = []
        self.agent_by_name = {}
        self.task_idle = Task(0 , np.array([0,0]), "Hold", {"Hold" : 0.0}, (0, self.max_time_steps), self.sceneData, self.max_time_steps) 
             
        agents_list = list(range(self.n_agents))
        
        self.rndAgentGen.shuffle(agents_list)
        self.agents = self.possible_agents.copy()        
        self.rndAgentGen.shuffle(self.agents)
        
        self.agent_selector = agent_selector(self.agents)
        self.current_agent = self.agent_selector.next()

        # Initialize agents_obj with None placeholders
        #self.agents_obj = [None] * self.n_agents
        self.agents_obj: List[Optional[UAV]] = [None] * self.n_agents

        for agent_type, n_agents in self.agents_config.items():
            for i in range(n_agents):
                agent_id = agents_list.pop(0)
                agent = UAV(agent_id, f'{agent_type[0]}_agent{i}', 
                                                self.random_position(self.rndAgentGen, obstacles=self.obstacles) if self.random_init_pos else self.bases[0],
                                                agent_type,  
                                                self) 
                self.agents_obj[agent_id] = agent 
                agent.max_speed = agent.max_speed / self.simulation_frame_rate * 0.02               
                self.agent_by_name[agent.name] = agent

        
        #-------------------  Define Fail Condition  -------------------#
        for agent in self.agents_obj:
            if self.rndAgentGen.random() < self.fail_rate * agent.fail_multiplier:                
                agent.fail_event = self.rndAgentGen.randint(1, 1000 if self.max_time_steps == -1 else self.max_time_steps )
                #print (f'Agent({agent.id}): {agent.fail_event}')
               
        
        #-------------------  Define Missions Areas  -------------------#
        self.mission_areas = []

        for i in range(self.n_mission_areas): 
            
            area_width  = self.sceneData.GameArea[0] * self.rndMissionGen.randint(10,20)/100  
            area_height = self.sceneData.GameArea[1] * self.rndMissionGen.randint(10,20)/100  

            self.mission_areas.append(
                SquareArea( self.random_position(
                                self.rndMissionGen, min_distance=max(area_width,area_height)), 
                            area_width, 
                            area_width
                            )          
            )

        #-------------------  Define Tasks  -------------------#       

        task_list = list(range(1, self.n_tasks))
        
        self.rndAgentGen.shuffle(task_list)                
        #task_idle.type = "Idle"       
        self.tasks: List[Optional[Task]] = []
        self.tasks.append(self.task_idle)
       
        hold_tasks_num = 0
        
        for task_type, n_tasks in self.tasks_config.items(): 
            for i in range(n_tasks):
                
                selected_mission = None
                
                if self.n_mission_areas > 0:
                    selected_mission = self.rndMissionGen.choice(self.mission_areas)
                
                task_id = task_list.pop(0)
                
                if task_type != "Hold":
                    task_position = self.random_position(self.rndTgtGen, obstacles = self.obstacles, contact_line = True, mission_area=selected_mission)
                    task_req = {task_type : 1.0} 
                                   
                else:
                    task_position = np.array([int((hold_tasks_num + 1) * self.sceneData.GameArea[0] / 5), int (self.sceneData.GameArea[1] / 4) ])                
                    hold_tasks_num += 1                      
                    task_req = {"Def" : 1.0, "Hold" : 1.0}        

                self.tasks.append(  Task(task_id, 
                                        task_position, 
                                        task_type, 
                                        task_req,
                                        (20, self.max_time_steps), 
                                        self.sceneData,
                                        self.max_time_steps)
                                        )
    
        #Tasks that each agent is doing                        
        self.allocation_table = [set() for _ in range(self.max_time_steps + 1)]       
                
        self.previous_agents_positions = None        
                
        self.time_steps = 0        
        self.agent_distances = np.zeros(len(self.agents_obj))
        self.reached_tasks.clear()
        self.total_distance = 0
        self.agent_directions = np.zeros(self.n_agents)
                              
        self.all_agent_positions = np.vstack([agent.position for agent in self.agents_obj])
        self.task_position = np.vstack([task.position for task in self.tasks])
        
        self.rewards = {agent.name : 0 for agent in self.agents_obj}
        self._cumulative_rewards = {agent.name : 0 for agent in self.agents_obj}
        self.terminations = {agent.name : False for agent in self.agents_obj}
        self.truncations = {agent.name : False for agent in self.agents_obj}
        self.infos = {agent.name : {} for agent in self.agents_obj}
        self.state = {agent.name : None for agent in self.agents_obj}
                        
        self.current_agent = self.agent_selector.reset()
                                    
        self._generate_observations()
                       
        return self.observations, self.infos
    
    def get_initial_state(self):
       # return a dictionary with the initial state information
       return  { 
                   "agents" : copy.deepcopy(self.agents),
                   "tasks" : copy.deepcopy(self.tasks),
                   "quality_table" : copy.deepcopy(self.quality_table) ,
                   "events" : []
                   }

#------------------ STEP FUNTION --------------------------#    
    def step(self, actions):                                  
        action_reward = 0
        distance_reward = 0
        quality_reward = 0
        S_quality_reward = 0
        time_reward = 0
        self.step_reward = 0

        done_events = []  

        #print(self.time_steps, end="")    
                                                                          
        #Select next available agent
        self.agent_selection = self.agent_selector.next()
       
        #while self.agents_obj[self.agent_name_mapping[self.agent_selection]].state == -1:
        #self.agent_selection = self.agent_selector.next()            
        
        self.current_agent = self.agent_selection        
                        
        if self.action_mode == "TaskAssign":
                                 
            self.time_steps += 1
            self.previous_agents_positions = np.copy([agent.position for agent in self.agents_obj])                                          
           
            #Process Shared Events
            if len(self.event_list) != 0:                
                event = self.event_list.pop()                                
                if event == "Reset_Allocation":
                    self.releaseAllTasks()
                    print("Allocation_RESET")
                    done_events.append(event)
                        
            #------------  TASK ALLOCATION  ----------------#
            notAllocatedTasks = len(self.unallocated_tasks())
            
            if isinstance(actions,dict):               
                                                                
                #print(actions)
                for agent_name, obs_task_ids in actions.items(): 
                                            
                    agent : UAV = self.agent_by_name[agent_name]
                    agent_index = agent.id                      
                    
                    #Agent Out of Order
                    if agent.state == -1:                                                
                        continue
                                        
                    if not isinstance(obs_task_ids, list):
                        obs_task_ids = [obs_task_ids]
                    
                    for obs_task_idx in obs_task_ids:
                                                                                                                                                                                                                    
                        if obs_task_idx < len(self.last_tasks_info):
                            
                            #Take the ID for the task in with the index "obs_task_idx" in observation                                                        
                            task: Task = self.last_tasks_info[obs_task_idx]
                            
                            if task is None:                            
                                action_reward += -5                                
                                continue                                                           
                        else:                            
                            #Reward: Penaulty for allocate task with wrong index
                            action_reward += -5
                            continue                                

                        #Multi agent per task and one task per agent
                        if not self.multiple_tasks_per_agent and self.multiple_agents_per_task:

                            
                            if len(agent.tasks) > 0:

                                #If the agent change the current task receive a small penaulty                            
                                if agent.tasks[0].id != task.id:
                                    
                                    if agent.tasks[0].id != 0:                                    
                                        S_quality_reward -= 0.05                                                                        
                                        S_quality_reward -= agent.currentCap2Task[task.typeIdx]                                    
                                else:
                                    continue                               
                                                            
                            EnvUtils.desallocateAll([agent], self)
                                                            
                            if task.id == 0:
                                
                                agent.tasks = [self.task_idle]                                
                                agent.next_free_position = agent.position
                                agent.next_free_time = self.time_steps
                                agent.state = 0
                                continue
                                                       
                            if agent.allocate(task) :                                                                            
                                
                                #self.tasks_current_quality[task_id] = self.quality_table[agent_index][task_id]                                                                      
                                self.allocation_table[task.id].add(agent.name)
                                                                                                     
                                agentCap = agent.currentCap2Task[task.typeIdx] 
                                missingCapBefore = task.currentReqs[task.typeIdx] - (task.allocatedReqs[task.typeIdx] - agentCap)  
                                missingCapBefore = missingCapBefore if missingCapBefore > 0 else 0                                
                                
                                addedCap =  agentCap if missingCapBefore > 0 else missingCapBefore
                                
                                S_quality_reward += addedCap
                                #print(f'ACAP:{agentCap},Missing: {missingCapBefore} Rwd {S_quality_reward} / {addedCap}' )
                                task.status = 1

                                #task.addAgentCap(agent)
                                #action_reward += 1.0 
                                #quality_reward += self.agents_obj[agent_index].fit2Task[task.typeIdx]   # noqa: E501
                                distance_reward += self.calculate_agent_expected_reward(agent)                                                  
                                                                                                
                                #Agent state as doing task
                                if agent.state != 1 and agent.state != -1: 
                                    agent.state = 1
                            #else:
                            #    print("Wrong Allocation")
                                                
                        if self.multiple_tasks_per_agent and not self.multiple_agents_per_task:
                            
                            if task.status == 0:
                                
                                aloccation = self.agents_obj[agent_index].allocate(task, self.time_steps, self.max_time_steps)                                    
                                                                
                                if aloccation > 0:                                                                            
                                    #self.tasks_current_quality[task_id] = self.quality_table[agent_index][task_id]                                                                      
                                    self.allocation_table[task.id] = agent_index
                                    task.status = 1                                        
                                    action_reward += 5.0 #* self.agents_obj[agent_index].fit2Task[task.typeIdx]#
                                    S_quality_reward += self.agents_obj[agent_index].fit2Task[task.typeIdx]  # noqa: E501
                                    distance_reward += self.calculate_agent_expected_reward(self.agents_obj[agent_index])                                                  
                                    
                                    if len(self.unallocated_tasks()) == 0:                                            
                                        time_reward = max(self.agents_obj, key=lambda obj: obj.next_free_time).next_free_time / 200
                                    
                                    if self.agents_obj[agent_index].state != 1 and self.agents_obj[agent_index].state != -1: 
                                        self.agents_obj[agent_index].state = 1
                            
                            else:                                
                                if notAllocatedTasks > 0:
                                    action_reward += 0
                                                                                
            reached_tasks_this_step = set()    
                                   
            if self.hidden_obstacles:            
                self.detect_segments()
            
            # Calcular as novas posições dos agents 
            for i, agent in enumerate(self.agents_obj):
                
                if isinstance(agent, UAV):                                       
                    #Check if agent is operational
                    if agent.state == -1: 
                        continue
                                        
                    if agent.fail_event == self.time_steps:
                        agent.state = -1
                        agent.tasks = []
                        agent.desallocateAll(self.time_steps)
                        self.event_list.append("Reset_Allocation")                    
                        #print("Fail:" , agent.id, agent.fail_event )
                        #print(self.agents_obj[agent.id].tasks)
                        continue
                    
                    movement = np.array([0,0])
                    avoid_vector = np.array([0,0]) 

                    #----------------IDLE-----------------#
                    if agent.state == 0: 
                        if len(self.reached_tasks) == self.n_tasks:
                            agent.state = 3                                     
                
                    #------------------- HAS TASK ------------------#                                                                                
                    if len(agent.tasks) > 0:
                        
                        current_task = agent.tasks[0]

                        if current_task.id != 0:

                            #----------------NAVIGATING-----------------#
                            if agent.state == 1: 
                            
                                dir_task = current_task.position - agent.position
                                distance_task  = np.linalg.norm(dir_task) 
                                
                                if abs(distance_task) < EPS:                                    
                                    dir_task_norm = 0  
                                else:
                                    dir_task_norm = dir_task / distance_task
                                          
                                # Limit to consider Task Aerea Reached
                                if distance_task < agent.max_speed:                      
                                    
                                    agent.state = 2
                                    #agent.task_start = self.time_steps
                                    #print(f'Agent{agent.id} in Task {current_task.id}')  

                                movement = dir_task_norm 
                                avoid_vector = agent.avoid_obstacles(agent, self.obstacles, movement, self.sceneData)                                                  
                                    
                            #----------------IN TASK-------------------#
                            elif agent.state == 2: 
                                                                                
                                #Just Started the Task
                                if agent.task_start == -1:                                            
                                    
                                    agent.task_start = self.time_steps
                                    agent.position = current_task.position
                                    #print(f'Agent{agent.id} in Task {current_task.id}')
                                    
                                else:
                                    
                                    #Check if TASK is CONCLUDED
                                    if (self.time_steps - agent.task_start) >= current_task.task_duration and \
                                        current_task.id != 0 and \
                                        current_task.type != "Hold" and\
                                        current_task.status != 2:
                                                                           
                                        task = agent.tasks.pop(0)
                                        #task = current_task
                                                                                                                                                                                                      
                                        agent.task_start = -1                                                                                                                 
                                        agent.tasks_done[task.id] = agent.currentCap2Task
                                        task.doneReqs += agent.currentCap2Task
                                        task.currentReqs -= agent.currentCap2Task

                                        if task.type == "Att":                
                                                                                
                                            agent.attackCap -= 1
                                            if agent.attackCap <= 0:
                                                agent.currentCap2Task[task.typeIdx] = 0                                    

                                        #print(f'Done: {task.doneReqs[task.typeIdx]} | {task.orgReqs[task.typeIdx]}')                                      
                                        if task.doneReqs[task.typeIdx] >= task.orgReqs[task.typeIdx]:                                                                                            
                                            reached_tasks_this_step.add(task.id)
                                            self.reached_tasks.add(task.id) 
                                            
                                            #Reward if just concluded the task
                                            if task.status != 2:                                            
                                                quality_reward += task.orgReqs[task.typeIdx] * 2
                                                task.status = 2   
                                                #print(f'Concluded Task {task.id} | {task.type} -> Agent: {agent.type}')                                             
                                        else:
                                            quality_reward += agent.currentCap2Task[task.typeIdx]     
                                                                                    
                                        if len(agent.tasks) >= 1:
                                            if agent.tasks[0].id != 0:
                                                agent.state = 1                                    
                                            else:
                                                agent.state = 0
                                        else:                                        
                                            
                                            agent.tasks = [self.task_idle]  
                                            agent.next_free_time = -1
                                            agent.next_free_position = agent.position   
                                            
                                            if len(self.reached_tasks) == self.n_tasks:
                                                self.conclusion_time = self.time_steps
                                                agent.state = 3 
                                            else:
                                                agent.state = 0  
                                    else:

                                        movement = agent.doTask(self.agent_directions[i], None, None, current_task.type)                            


                    #----------------RETURNING BASE (NO TASK)--------------------
                    if agent.state == 3:
                                                                
                        if np.linalg.norm(agent.position - self.bases[0]) < agent.max_speed + 5:
                            agent.state = 0
                        else:
                            movement = EnvUtils.norm_vector(self.bases[0]  - agent.position)
                            avoid_vector = agent.avoid_obstacles(agent, self.obstacles, movement, self.sceneData)     
                                                                                                                        
                    movement = EnvUtils.norm_vector(movement + avoid_vector) * agent.max_speed
                                                                                
                    agent.position = agent.position + movement 

                    agent.position = np.clip(agent.position, 0, [self.area_width, self.area_height])

                    self.exporter.add_drone_state(self.time_steps, agent) 
         
            self.all_agent_positions = np.vstack([agent.position for agent in self.agents_obj])
                                       
            self.agent_directions = [ direction for direction in (self.all_agent_positions - self.previous_agents_positions)]
                                                            
            dists = np.linalg.norm(self.all_agent_positions - self.previous_agents_positions, axis=1)           
            
            self.agent_distances += dists    
            self.total_distance += np.sum(dists)
                                                                                              
            time_penaulty = -(self.n_tasks - len(self.reached_tasks))/self.n_tasks * (self.time_steps / self.max_time_steps)
            
            if self.time_steps > self.n_tasks + 1:
                alloc_reward = - len(self.unallocated_tasks())
            else:
                alloc_reward = 0
            
            self.generate_threat()
            self.update_threats()   
                                                                                                                                                                                                       
            # rewards for all agents are placed in the rewards dictionary to be returned
            self.rewards = {agent.name :  0.0 * action_reward  +   #Rand +50
                                          0.0 * distance_reward +  #Rand -4
                                          1.0 * quality_reward +   #Rand +6
                                          0.1 * S_quality_reward +   #Rand +6                                                                                    
                                          0.0 * self.n_tasks * time_reward +      #Rand -9
                                          0.0 * alloc_reward  +
                                          0.0 * time_penaulty + 
                                          0.0 * self.step_reward for agent in self.agents_obj} #Rand -28 
                                                               
            #self._cumulative_rewards["agent0"] += self.rewards["agent0"]
                                                
            #done = (len(self.reached_tasks) == self.n_tasks or ((self.time_steps >= self.max_time_steps) and (self.max_time_steps > 0)))
            done = ((self.time_steps >= self.max_time_steps) and (self.max_time_steps > 0))
            
            #Only for speed up traning without Dynamic conditions
            #done = done or (self.allocation_table.count(-1) == 0 and self.time_steps >= 50)
                                  
            #self.terminations = { agent.name : (done or (agent.state < 0 )) for agent in self.agents_obj}
            self.terminations = { agent.name : False for agent in self.agents_obj}
                        
            #env_truncation = done#(self.time_steps >= self.max_time_steps) if self.max_time_steps > 0 else done             
            env_truncation = (self.time_steps >= self.max_time_steps) if self.max_time_steps > 0 else done             


            self.truncations = {agent.name: env_truncation for agent in self.agents_obj}     
            
            self.infos = {agent.name: {} for agent in self.agents_obj}            
            self.infos['selected'] = self.agent_selection
            self.infos['events'] = done_events            

            self._generate_observations()                                    

            if done:
                metrics =  self.calculate_metrics() 
                self.infos['metrics'] = metrics                
                #self.rewards = {agent.name :  self.F_Reward for agent in self.agents_obj} #Rand -28                
                
                return self.observations, self.rewards, self.terminations, self.truncations, self.infos #metrics
            else:                
                return self.observations, self.rewards, self.terminations, self.truncations, self.infos
                                        
    
    def observe(self, agent):
   
        # Get the observation for the given agent   
        observation = self.observations[agent]
        observation['agent_id'] = agent
        return observation
 
    def calculate_agent_expected_reward(self, agent):
        
        # Calculate distance between last task and base position
        if len(agent.tasks) >= 2:
            total_distance = np.linalg.norm(agent.next_free_position - agent.tasks[-2].position)
        else:
            total_distance = np.linalg.norm(agent.next_free_position - agent.position)
        
        #print(total_distance)
    
        # Calculate reward (you can adjust the penalty factor as needed)
        reward = -1.0 * total_distance / self.max_coord
    
        return reward  

    def calculate_metrics(self):
                
        total_distance = self.total_distance       
        #load_balancing = self.calculate_load_balancing()
        load_balancing_std = self.calculate_load_balancing_std()
        
        # F_quality = np.mean([0 if t.final_quality == -1 else t.final_quality  for t in self.tasks])        
        # F_Time = 1 / (self.conclusion_time / self.max_time_steps * 5)
        # F_distance = 1 / (total_distance/self.n_agents / self.max_coord)
        
        # self.F_Reward = 0.25 * F_distance + 0.6 * F_quality + 0.15 * F_Time

        # if self.conclusion_time == self.max_time_steps:
        #     F_quality = 0
        #     F_Time = 0
        #     F_distance = 0
        #     self.F_Reward = 0

        F_quality = np.mean([0 if t.final_quality == -1 else t.final_quality  for t in self.tasks])        
        F_Time = 1 / self.conclusion_time * self.max_time_steps
                
        F_distance = 1 / total_distance * self.max_coord
        reward_weigths = [1.0, 1.0, 0.0]#[0.25, 0.6, 0.15]
        
        self.F_Reward = reward_weigths[0] * F_distance/0.06 + reward_weigths[1] * F_quality/0.9 + reward_weigths[2] * F_Time / 1.4

        self.F_Reward = self.F_Reward * self.max_time_steps

        if self.conclusion_time == (self.max_time_steps + 1):
            #print("Fail to Conclude")
            self.F_Reward = 0
            F_Time = 0

        #print( f'Q:{F_quality}|D:{F_distance}|T:{F_Time}|F:{self.F_Reward}|Rem:{self.allocation_table.count(-1)}' )

        return { 
            "F_time": F_Time ,
            "F_distance": F_distance ,
            #"load_balancing": load_balancing,
            "F_load": 1 / load_balancing_std,
            "F_quality": F_quality,
            "F_Reward": self.F_Reward 
        }

    def _one_hot(self, idx, num_classes):
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[idx] = 1
        return one_hot_vector    

    def detect_segments(self, detection_distance=30):
        num_segments = 12
        arc_angle = 2 * np.pi / num_segments
    
        for obstacle in self.obstacles:
            for i in range(num_segments):
                start_angle = i * arc_angle
                stop_angle = (i + 1) * arc_angle
                mid_angle = (start_angle + stop_angle) / 2
    
                mid_point_x = obstacle.position[0] + obstacle.size * np.cos(mid_angle)
                mid_point_y = obstacle.position[1] - obstacle.size * np.sin(mid_angle)  # Inverta o sinal do componente y
    
                mid_point = np.array([mid_point_x, mid_point_y])
    
                detected = False
                for agent in self.agents_obj:
                    agent_to_mid_point = mid_point - np.array(agent.position)
                    distance = np.linalg.norm(agent_to_mid_point)
    
                    if distance <= detection_distance:
                        detected = True
                        break  # Não é necessário verificar outros agents para este segmento
    
                if detected:
                    obstacle.detected_segments.append((start_angle, stop_angle))
                        
    def random_position(self,  rndGen, obstacles = None, min_distance = 20, own_range = 3, contact_line = False, mission_area: SquareArea = None):
          
        if contact_line:
            limit_line = self.sceneData.ContactLine
        else:
            limit_line = 0
            
        tries = 0
        while tries < 100:            
                        
            if mission_area is not None:                
                         
                # Generate a random x coordinate
                x = rndGen.uniform(mission_area.top_left[0], mission_area.top_left[0] + mission_area.width)                
                y = rndGen.uniform(mission_area.top_left[1], mission_area.top_left[1] + mission_area.height)
            
            else:                        
                x = rndGen.uniform(own_range + min_distance, self.area_width - own_range - min_distance)
                y = rndGen.uniform(own_range + min_distance, self.area_height - own_range - min_distance - ((self.area_height-limit_line) if limit_line != 0 else 0))
            
            point = np.array([x, y])
    
            if  obstacles != None:
                # Verificar a distância entre o ponto e todos os obstáculos
                valid_point = True
                for obstacle in obstacles:
                    distance_to_obstacle = np.linalg.norm(point - obstacle.position) - own_range
                    if distance_to_obstacle < obstacle.size + min_distance :
                        valid_point = False
                        break
    
                if valid_point:
                    
                    return point                    
            else:                
                return point
             
            tries +=  1
        
        raise ValueError(f'Error to build a valid scenario, can´t find no space for more {len(obstacles)-1} then obstacles')        
                
    def calculate_capacities_table(self, agents, tasks, SceneData):
        
        # Cria uma matriz de capacidade inicializada com zeros
        capacity_matrix = [[0 for _ in tasks] for _ in agents]
        
        # Preenche a matriz de capacidade com os valores de adequação de sensores
        for i, agent in enumerate(agents):
            for j, task in enumerate(tasks):
                # Para cada combinação de agent e alvo, encontra o valor máximo de adequação
                max_capacity = 0
                #for sensor_type in agent.sensors: 
                    #print(sensor_type, " -- ", task.task_type)
                agent_capacity = agent.fit2Task[task.typeIdx]
                    
                if agent_capacity > max_capacity:
                    max_capacity = agent_capacity
                
                # Armazena o valor máximo de adequação na matriz de capacidade
                capacity_matrix[i][j] = max_capacity
        
        return capacity_matrix                
   
    def unallocated_tasks(self):                
        
        #allocated_tasks = set(task for agent in self.agents_obj for task in agent.tasks)
        #all_tasks = set(range(self.n_tasks))        
        unallocated = [idx for idx, agents in enumerate(self.allocation_table) if not agents]

        return unallocated
    
    def releaseAllTasks(self):

        available_agents = set()        
        for agent in self.agents_obj:
            
            agent.desallocateAll(self.time_steps)
            
            if agent.state != -1:                
                agent.state = 0
                available_agents.add(agent.typeIdx)
        
        
        for task in self.tasks:
            if task.status != 2:
                              
                cum_cap = 0
                for agentType in available_agents:
                    cum_cap =+ task.fit2Agent[agentType]                
                
                if cum_cap == 0:                   
                    task.status = 2 #No capability to DO it
                    #print("No CAP")
                    
                    # Adicionar o alvo par a lista alcançados por não ser mais possível
                    # E não prejudicar algoritmos que alocam errado
                    if task.id not in self.reached_tasks:                        
                        self.reached_tasks.add(task.id) 
                        task.status = 2    

                        if len(self.reached_tasks) == self.n_tasks:
                            self.conclusion_time = self.time_steps

                else:
                    task.status = 0                        
                    self.allocation_table[task.id] = set()                            
                 
        self.tasks_current_quality = {i: -1 for i in range(self.n_tasks)}  
    
    def get_live_agents(self):

        return [agent for agent in self.agents_obj if agent.state != -1]  
       

####---------------------Dynamic Conditions ----------------------------------###

    def generate_threat(self):
        if self.n_threats > 0 and self.time_steps % 10 == 0:
            if self.rndAgentGen.random() < self.threat_generation_probability:
                start_position = np.array([self.rndAgentGen.randint(0, self.sceneData.GameArea[0]), 0])
                speed = self.sceneData.maxSpeeds["T1"]
                engageRange = self.sceneData.engage_range["T1"]
                attack = self.sceneData.UavCapTable["T1"][2]
                defence = self.sceneData.UavCapTable["T1"][3]            
                #threat_id = len(self.threats)
                new_task_id = len(self.tasks) + 1
                
                new_threat = Threat(new_task_id, start_position, speed, engageRange, attack, defence)
                
                #TODO-> Correct frame rate handle
                new_threat.max_speed = new_threat.max_speed / self.simulation_frame_rate * 0.02  
                
                new_threat.target_agent = self.get_closest_agent(start_position)                                                                
                
                relative_task = self.TaskFromThreat(new_task_id, new_threat)
                
                self.tasks.append(relative_task)
                self.allocation_table.append(set())
                
                new_threat.relative_task = relative_task
                self.threats.append(new_threat)

                self.n_threats -= 1


    def get_closest_agent(self, position):
        
        min_distance_F = float('inf')
        min_distance_W = float('inf')
        
        closest_F_agent = None #Figther Agents
        closest_W_agent = None #Weaker Agents        

        for agent in self.agents_obj:
            if agent.state != -1 and agent.state != 4:
                distance = np.linalg.norm(np.array(agent.position) - np.array(position))
                
                if agent.type == "F1" or agent.type == "F2":
                    if distance < min_distance_F:
                        min_distance_F = distance
                        closest_F_agent = agent
                else:
                    if distance < min_distance_W:
                        min_distance_W = distance
                        closest_W_agent = agent

        #Initial Enemy Behavior (chase closest)
        if True:
            if min_distance_F < min_distance_W:
                return closest_F_agent
            else: 
                return closest_W_agent
        
        #Version 2 Enemy Behavior (chase closest not Fighter)
        if closest_W_agent is not None:
            return closest_W_agent
        else:
            return closest_F_agent

    def update_threats(self):
        for threat in self.threats:
            if threat.status == 0 or threat.target_agent is None:
                direction = np.array([0,-1])                
                threat.position += threat.max_speed * direction
            else:          
                #direction = np.array(threat.target_agent.position) - np.array(threat.position)
                #direction = direction / np.linalg.norm(direction)
                direction = EnvUtils.norm_vector(threat.target_agent.position  - threat.position)
                threat.position = threat.position + threat.max_speed * direction

                if np.linalg.norm(np.array(threat.target_agent.position) - np.array(threat.position)) < threat.engage_range:
                    self.handle_threat_engagement(threat)
            threat.relative_task.position = threat.position
                

    def handle_threat_engagement(self, threat: Threat):
        
        attDiff = threat.target_agent.currentCap2Task[2] / threat.attack
        defDiff = threat.target_agent.currentCap2Task[3] / threat.defence
        engageDiff = threat.target_agent.engage_range / threat.engage_range

        avg_diff = (attDiff + defDiff + engageDiff) / 3                
        neutralize_prob = avg_diff / (avg_diff + 1)  

        #print(neutralize_prob)
                
        if self.rndAgentGen.random() < neutralize_prob:        
            
            threat.relative_task.status = 2
            threat.target_agent.attackCap -= 1
            if threat.target_agent.attackCap <= 0:
                threat.target_agent.currentCap2Task[3] = 0
            
            self.threats.remove(threat)       
            self.step_reward += 1.0 
            # print(self.step_reward)                  
            # Provide reward or update agent state if necessary
        else:            
            threat.target_agent.outOfService()                       
            threat.attackCap -= 1              
            
            self.step_reward -= 1.0  
            # print(self.step_reward)      
    
            if threat.attackCap <= 0:
                threat.status = 0
                threat.relative_task.status = 2
            else:
                threat.target_agent = self.get_closest_agent(threat.position)
            
            # Apply penaulty or update environment state if necessary  
    
    def TaskFromThreat(self, task_id, threat: Threat):

        new_task = Task( task_id, threat.position, 
                        "Att", 
                        {"Att" : threat.defence, "Def" : threat.attack}, 
                        (self.time_steps, self.max_time_steps), 
                        self.sceneData, 
                        self.max_time_steps,  
                        is_active=True )
        return new_task
          


####----------------------Metrics Calculation-----------------------------------
    def calculate_load_balancing_std(self):
        load_balancing_std = np.std(self.agent_distances)
        return load_balancing_std

    def calculate_load_balancing(self):
        min_distance = np.min(self.agent_distances)
        max_distance = np.max(self.agent_distances)
        load_balancing = max_distance - min_distance
        return load_balancing
   
    

    def plot_metrics(self, df, n_agents, n_tasks):
        # Group data by algorithm and calculate means and standard deviations
        grouped = df.groupby('Algorithm', sort = False)
        means = grouped.mean()
        std_devs = grouped.std()
        
        std_devs = std_devs / means.loc['Random']
        means = means / means.loc['Random']        
        
        # Calculate the number of algorithms and metrics
        num_algorithms = len(grouped)
        num_metrics = len(df.columns) - 1
    
        palette = sns.color_palette("Set1",n_colors=num_algorithms)
              
        # Create a single plot
        fig, ax = plt.subplots(figsize=(10, 5))
    
        # Define the bar width and the spacing between groups of bars
        bar_width = 0.7 / num_algorithms
        group_spacing = 1.2
    
        # Create a bar chart for each algorithm
        for i, (algo, data) in enumerate(grouped):
            index = np.arange(num_metrics) * group_spacing + i * bar_width
            ax.bar(index, means.loc[algo], bar_width, alpha=0.8, label=algo, yerr=std_devs.loc[algo], capsize=5, color=palette[i])
    
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title(f'Task Allocation: ({n_agents} agents, {n_tasks} tasks)')
        ax.set_xticks(np.arange(num_metrics) * group_spacing + (bar_width * (num_algorithms - 1) / 2))
        ax.set_xticklabels(list(grouped.columns)[:-1])
        ax.legend()
        ax.set_ylim(0, 2.0)
    
        plt.tight_layout()
        plt.show()
    
    def plot_convergence(self,df, n_agents, n_tasks, algorithm):
        
        cumulative_means = df.expanding().mean() / df.mean()
    
        palette = sns.color_palette("Set1",n_colors=len(df)-1)
              
        fig, ax = plt.subplots()
        auxDf = cumulative_means.reset_index()
        
        for i, metric in enumerate(auxDf.columns[1:]):
            ax.plot(auxDf[metric], label=metric, color = palette[i])
    
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Cummulative Means')
        ax.set_title(f'Convergence {algorithm} : ({n_agents} uavs, {n_tasks} tasks)')
        ax.legend()
        ax.set_ylim(0, 1.0)
            
        plt.tight_layout()
        plt.show()

#####--------------- Rendering in PyGame -------------------###################

    def draw_rotated_triangle(self, surface, x, y, size, angle, agent_type, state):
        
        mod_size = size / 2
        color = (0, 0, 255) if state >= 1 else (100, 100, 100)
                
        if agent_type == "R1":
            mod_size = size / 1.5
            color = (0, 200, 200) if state >= 1 else color 
        
        if agent_type == "C1":
            mod_size = size / 1.5
            color = (230, 230, 230) if state >= 1 else color 

        if agent_type == "T1":
            mod_size = size / 1.0
            color = (250, 0, 0) if state >= 1 else color 
                            
        angle_rad = np.radians(angle)    
        dx1 = mod_size  * np.cos(angle_rad - np.pi / 0.7)
        dy1 = mod_size  * np.sin(angle_rad - np.pi / 0.7)    
        dx2 = mod_size * np.cos(angle_rad)
        dy2 = mod_size * np.sin(angle_rad)        
        dx3 = mod_size  * np.cos(angle_rad + np.pi / 0.7)
        dy3 = mod_size  * np.sin(angle_rad + np.pi / 0.7)           
        points = [(x + dx1, y + dy1), (x + dx2, y + dy2), (x + dx3, y + dy3)]


        if agent_type == "R1":
            pygame.draw.polygon(surface, color, points )
        else:
            pygame.draw.polygon(surface, color, points )
        
        if state == -1:
            line_width = 3
            pygame.draw.line(surface, (230, 0, 0), (x - 10, y - 10), (x + 10, y + 10), line_width)
            pygame.draw.line(surface, (230, 0, 0), (x - 10, y + 10), (x + 10, y - 10), line_width)


    def draw_rotated_x(self,surface, x, y, size, angle, agent_type):
        half_size = size / 2
        angle_rad = np.radians(angle)
    
        dx1 = half_size * np.cos(angle_rad - np.pi / 4)
        dy1 = half_size * np.sin(angle_rad - np.pi / 4)
    
        dx2 = half_size * np.cos(angle_rad + np.pi / 4)
        dy2 = half_size * np.sin(angle_rad + np.pi / 4)
    
        pygame.draw.line(surface, (0, 0, 255), (x - dx1, y - dy1), (x + dx1, y + dy1), 2)
        pygame.draw.line(surface, (0, 0, 255), (x - dx2, y - dy2), (x + dx2, y + dy2), 2)
    

    def render(self, show_lines = True):
               
        if self.screen is None:
            
            pygame.init()
            self.screen = pygame.display.set_mode((self.area_width, self.area_height))
            pygame.display.set_caption('Multi agent Task Allocation')
        
        # Desenhar fundo
        agents_surface = pygame.Surface((self.area_width, self.area_height))        
        
        comm_surface = pygame.Surface((self.area_width, self.area_height))
        comm_surface.fill((0, 0, 0))
        comm_surface.set_alpha(100)
        
        pygame.draw.line(agents_surface, (0, 0, 90), (0, self.sceneData.ContactLine),(self.area_width, self.sceneData.ContactLine) , 3)

        #Draw Base
        base_size = 50
        base_x = self.bases[0][0]
        base_y = self.bases[0][1]
        
        pygame.draw.rect(agents_surface, (10,80,30), (base_x-base_size/2, base_y-base_size/2, base_size, base_size))
        
        font = pygame.font.Font(None, 18)
        font2 = pygame.font.Font(None, 24)

        #Draw misison Areas

        # Inside the render function:
        for area in self.mission_areas:

            # Draw dashed rectangle
            self.draw_dashed_line(agents_surface, (200, 200, 200), area.top_left, area.top_right) # top side
            self.draw_dashed_line(agents_surface, (200, 200, 200), area.top_right, area.bottom_right) # right side
            self.draw_dashed_line(agents_surface, (200, 200, 200), area.bottom_right, area.bottom_left) # bottom side
            self.draw_dashed_line(agents_surface, (200, 200, 200), area.bottom_left, area.top_left) # left side
        
        # Draw Tasks
        for task in self.tasks:
            
        
            if task.type == "Rec":
                color = (0, 255, 0) if task.status == 2 else (40, 80, 40)           
                pygame.draw.circle(agents_surface, color, (int(task.position[0]), int(task.position[1])), 8)
            elif task.type == "Hold":
                color = (80, 0, 80)           
                pygame.draw.circle(agents_surface, color, (int(task.position[0]), int(task.position[1])), 10)               
            else:
                if task.info == "Threat":
                    if task.status != 2:
                        color = (200, 0, 0) if task.status == 2 else (80, 40, 40)           
                        pygame.draw.circle(agents_surface, color, (int(task.position[0]), int(task.position[1])), 10) 
                    else:                    
                        line_width = 3
                        pygame.draw.line(agents_surface, (230, 0, 0), (int(task.position[0]) - 10, (int(task.position[1]) - 10)), (int(task.position[0]) + 10, (int(task.position[1]) + 10)), line_width)
                        pygame.draw.line(agents_surface, (230, 0, 0), (int(task.position[0]) - 10, (int(task.position[1]) + 10)), (int(task.position[0]) + 10, (int(task.position[1]) - 10)), line_width)
                else:
                    color = (200, 0, 0) if task.status == 2 else (80, 40, 40)           
                    pygame.draw.circle(agents_surface, color, (int(task.position[0]), int(task.position[1])), 10) 
                
            # Renderizar o número do alvo e desenhá-lo no centro do círculo
            task_number_text = font.render(str(task.id), True, (30, 30, 30))  # Renderizar o texto (preto)
            text_rect = task_number_text.get_rect(center=(int(task.position[0]), int(task.position[1])))  # Centralizar o texto no círculo
            agents_surface.blit(task_number_text, text_rect)
                        
        # Draw agents
        for i,agent in enumerate(self.agents_obj):
                        
            pygame.draw.circle(comm_surface, (40, 40, 40), (int(agent.position[0]), int(agent.position[1])), agent.relay_area)
            
            if show_lines:
                if len(agent.tasks) > 0 and agent.tasks[0].id != 0: 
                    # Desenhar linha entre o agent e seu alvo atual            
                    pygame.draw.line(agents_surface, (210, 210, 210), agent.position, agent.tasks[0].position, 1)
            
                # Desenhar linha entre o alvo atual e o próximo (mais clara)
                if len(agent.tasks) > 1 :                                    
                    pygame.draw.line(agents_surface, (100, 100, 100), self.tasks[agent.tasks[0]].position, agent.tasks[1].position, 1)
            
            #pygame.draw.circle(self.screen, (0, 0, 255), (int(agent.position[0]), int(agent.position[1])), 7)
            #self.draw_rotated_x(self.screen, int(agent.position[0]), int(agent.position[1]), 10, self.agent_directions[i])
            if agent.state != 0:
                self.draw_rotated_triangle(agents_surface, int(agent.position[0]), int(agent.position[1]), 20, np.degrees(np.arctan2(self.agent_directions[i][1],self.agent_directions[i][0])) , agent.type, agent.state)            
                
                if agent.type == "F1" or agent.type == "F2":
                    task_text = font.render(str(agent.attackCap), True, (200, 200, 200))  # Renderizar o texto (preto)                    
                    text_rect = task_text.get_rect(center=(int(agent.position[0]), int(agent.position[1])))  # Centralizar o texto no círculo
                    agents_surface.blit(task_text, text_rect)

            else:
                self.draw_rotated_triangle(agents_surface, int(agent.position[0]), int(agent.position[1]), 20, -90 , agent.type, agent.state)            
                                        
               
        # Desenhar obstáculos
        for obstacle in self.obstacles:
            obstacle_color = (150, 40, 40)  # Cor vermelha clara
            
            if not self.hidden_obstacles:
                pygame.draw.circle(comm_surface, obstacle_color, (int(obstacle.position[0]), int(obstacle.position[1])), obstacle.size)
                # Renderizar o texto "No Fly Zone" e desenhá-lo no centro do círculo
                obstacle_text = font.render("NFZ", True, (0, 0, 0))  # Renderizar o texto (preto)
                text_rect = obstacle_text.get_rect(center=(int(obstacle.position[0]), int(obstacle.position[1])))  # Centralizar o texto no círculo
                comm_surface.blit(obstacle_text, text_rect)        
            else:
                self.draw_detected_circle_segments(comm_surface, (255, 70, 70) ,obstacle)
                                                    
                                  
        # Drawing threats and their directions
        for threat in self.threats:            
            # Drawing threat's direction as a line (or arrow) towards its target
            line_length = threat.engage_range  # or any suitable length
            if threat.target_agent != None and threat.attackCap > 0:
                direction = np.array(threat.target_agent.position) - np.array(threat.position)
                direction = direction / np.linalg.norm(direction) * line_length
            else:
                direction = np.array([0,-1])
            
            end_position = (threat.position[0] + direction[0], threat.position[1] + direction[1])
            pygame.draw.line(agents_surface, (255, 0, 0), threat.position, end_position, 2)

            self.draw_rotated_triangle(agents_surface, 
                                       int(threat.position[0]), 
                                       int(threat.position[1]), 
                                       10, 
                                       np.degrees(np.arctan2(direction[1],direction[0])), 
                                       "T1", 1) 
            #Attack Missiles
            text_att = font.render(str(threat.attackCap), True, (200, 200, 200))  # Renderizar o texto (preto)
            text_rect = text_att.get_rect(center=(int(threat.position[0]), int(threat.position[1]))) # Centralizar o texto no círculo
            agents_surface.blit(text_att, text_rect)           


        
        self.screen.blit(agents_surface, (0,0))
        self.screen.blit(comm_surface, (0,0))
        
        texto = font.render(str(self.time_steps), True, (200,200,200))
        self.screen.blit(texto, (self.sceneData.GameArea[0] - 35, self.sceneData.GameArea[1] - 20))
        
        info = font2.render(self.info, True, (250,250,250))
        self.screen.blit(info, (20, 20))
        
        pygame.display.flip()
        
        # Salvar a imagem atual do jogo
        if self.recrdr != None:
            self.recrdr.click(self.screen) # save frame as png to _temp_/ folder
        
        # Limitar a taxa de quadros
        self.clock.tick(self.render_speed * 10)
        
        # Verificar se a janela está fechada
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True
   
    def close(self):
        if self.recrdr != None:
            self.recrdr.save() 
        #pygame.quit()

    def draw_dashed_line(self, surface, color, start_pos, end_pos, dash_length=10, space_length=5):
       
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx = x2 - x1
        dy = y2 - y1
        distance = int(math.sqrt(dx**2 + dy**2))
        dashes = distance // (dash_length + space_length)
        
        for i in range(dashes):
            start = (x1 + (dx * i / dashes), y1 + (dy * i / dashes))
            end = (x1 + (dx * (i + 0.5) / dashes), y1 + (dy * (i + 0.5) / dashes))
            pygame.draw.line(surface, color, start, end, 1)

    
    def draw_detected_circle_segments(self, surface, color, obstacle, width=1):
        
        radius = obstacle.size
        center = obstacle.position
        
        for start_angle, stop_angle in obstacle.detected_segments:
            rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            pygame.draw.arc(surface, color, rect, start_angle, stop_angle, width)



