from concurrent.futures import thread
from operator import length_hint
from typing import List, Optional
from ctypes import util
from email import utils
from pkgutil import extend_path
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Box
import numpy as np
import random
import copy
import sys
import math
import socket
import json

import matplotlib.pyplot as plt
import seaborn as sns
#from pygame_screen_recorder import pygame_screen_recorder as pgr

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.utils.agent_selector import agent_selector
from torch import pinverse

from .DroneEnvComponents import UAV, SquareArea, Task, Obstacle, Threat
from .MultiDroneEnvData import SceneData 
from .MultiDroneEnvUtils import agentEnvOptions, EnvUtils, ACMIExporter

# pygame removed
import core_sim

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
        
        self.side_panel_width = 300

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
 
        self.debuger_data = self.render_enabled
        
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
                self.possible_agents.append(f'{agent_type[0:2]}_agent{i}')

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
                
        self.max_tasks = 40 + 1
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
        self.threats_groups = [] # Threats before activated
        self.threat_generation_probability = 0.7  / self.simulation_frame_rate * 0.02#TODO: Frame rate Adjust as needed        
        self.threats_list = self.config.threats_list
        self.threat_wide = self.sceneData.GameArea[0] / 10
        #self.n_threats = self.max_threats

        self.n_mission_areas = 3
        self.mission_areas = None
        
        #Dynamic Conditions        
        self.hidden_obstacles = self.config.hidden_obstacles        
        
        self.num_obstacles = self.config.num_obstacles        
        self.obstacles = None
        
        self.fail_rate = self.config.fail_rate
        self.early_terminate = getattr(self.config, "early_terminate", False)
        self.capability_mask = getattr(self.config, "capability_mask", False)
        self.saturate_mask = getattr(self.config, "saturate_mask", False)
        self.reward_weights = getattr(self.config, "reward_weights", None) or {
            "action": 0.0,
            "distance": 1.0,
            "quality": 1.0,
            "s_quality": 1.0,
            "time": 0.0,
            "alloc": 0.0,
            "time_penaulty": 0.0,
            "step": 0.0,
        }
        self.arrival_rate = float(getattr(self.config, "arrival_rate", 0.0) or 0.0)
        self.include_time_windows = bool(getattr(self.config, "include_time_windows", False))
        self.dynamic_idle_penalty = float(getattr(self.config, "dynamic_idle_penalty", 0.0) or 0.0)
        # WPS knobs
        self.sense_radius = float(getattr(self.config, "sense_radius", 0.0) or 0.0)
        self.threat_delay = int(getattr(self.config, "threat_delay", 0) or 0)
        self.hard_windows = bool(getattr(self.config, "hard_windows", False))
        self.window_length = int(getattr(self.config, "window_length", 30) or 30)
        self.burst_mode = bool(getattr(self.config, "burst_mode", False))
        self.burst_size = int(getattr(self.config, "burst_size", 3) or 3)
        self.miss_penalty = float(getattr(self.config, "miss_penalty", 25.0) or 0.0)
        self.on_time_bonus = float(getattr(self.config, "on_time_bonus", 10.0) or 0.0)
        self.dual_region_bursts = bool(getattr(self.config, "dual_region_bursts", False))
        self.share_knowledge = bool(getattr(self.config, "share_knowledge", True))
        self.commit_horizon = int(getattr(self.config, "commit_horizon", 0) or 0)
        self.reassign_penalty = float(getattr(self.config, "reassign_penalty", 0.0) or 0.0)
        self.escort_enabled = bool(getattr(self.config, "escort_enabled", False))
        self.escort_radius = float(getattr(self.config, "escort_radius", 70.0) or 70.0)
        self.escort_requirement = float(getattr(self.config, "escort_requirement", 1.2) or 1.2)
        self.escort_intercept_radius = float(
            getattr(self.config, "escort_intercept_radius", 100.0) or 100.0
        )
        self.mutual_support_radius = float(
            getattr(self.config, "mutual_support_radius", 80.0) or 80.0
        )
        self.escort_agent_types = tuple(
            getattr(self.config, "escort_agent_types", ("F1", "F2")) or ("F1", "F2")
        )
        self._burst_region_toggle = 0
        self._next_task_id = 1  # 0 reserved for idle
        self._next_threat_id = 0
        self.n_reallocations = 0
        self.n_task_switches = 0
        self.n_arrivals = 0
        self._pending_reset = False
        self.agent_known_tasks = {}  # agent_name -> set of task ids
        self.pending_reveals = []  # (reveal_time, task_id) for threat_delay
        self.n_missed_windows = 0
        self.n_on_time = 0
        self.n_windowed_tasks = 0
        self._idle_reserve_steps = 0
        # Escort metrics
        self.escort_requests = 0
        self.escort_completed = 0
        self.escort_failed = 0
        self.escort_required_steps = 0
        self.escort_covered_steps = 0
        self.protection_breaches = 0
        self.threats_intercepted = 0
        self.recon_losses = 0
        self.escort_losses = 0
        self.mutual_support_engagements = 0
        self.protected_rec_completed = 0
        self._escort_by_recon = {}  # recon_name -> escort Task
        
        # self.tasks_current_quality = None        
        self.allocation_table = None
                                               
        #Table with the quality of each agent for each task
        self.quality_table = None
                
        self.time_steps = 0
        self.total_distance = 0
        self.agent_distances = None
        self.agent_directions = None 

        self.F_Reward = 0 
        self.step_reward = 0
        
        

        self.final_rew_factor   = 1
        self.reward_multiplifier = 1000

        self.reward_norm_factor = 1

        self.event_list = []      
                
        self.previous_agents_positions = None
        self.previous_agents_positions = None
                        
        # Inicializar o Pygame
        self.screen = None 
        self.debug_panel = None     
        self.recrdr = None
        self.scale_factor = 1.0
        
        if self.render_mode == "human":
            self.screen = None

        elif self.render_mode == "rgb_array":
            self.screen = None
            self.recrdr = None# Configurar o MovieWriter
            #self.recrdr = pgr("Test_Tessi.gif") # init recorder object
        if self.debuger_data:
            # Set up a socket connection
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.host = 'localhost'
            self.port = 65432

            try:
                 self.client_socket.connect(( self.host,  self.port))
            except  self.client_socket.error as e:
                print(str(e))
                sys.exit()
        
        self.simulation_running = True
              
        self._observation_space = Dict({
            "agent_position": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "agent_state": Box(low=0, high=1, shape=(5,), dtype=np.float32),  # Assuming 5 possible states (0, 1, 2, 3 and 4)
            "agent_type": Box(low=0, high=1, shape=(6,), dtype=np.float32),  # One-hot encoded agent types  # Assuming 2 possible types
            "next_free_time": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "position_after_last_task": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            #"agent_relay_area": Box(low=0, high=max(self.area_width, self.area_height), shape=(2,), dtype=np.float32),           
            "tasks_info": Box(low=0, high=1, shape=((self.max_tasks ) * 12,), dtype=np.float32),
            #"agents_info": Box(low=0, high=1, shape=((self.max_agents ) * 5,), dtype=np.float32),
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

    def _alloc_task_id(self) -> int:
        tid = int(self._next_task_id)
        self._next_task_id = tid + 1
        return tid

    def _alloc_threat_id(self) -> int:
        tid = int(self._next_threat_id)
        self._next_threat_id = tid + 1
        return tid

    def _ensure_allocation_bucket(self, task_id: int):
        if self.allocation_table is None:
            self.allocation_table = []
        while len(self.allocation_table) <= task_id:
            self.allocation_table.append(set())

    def _is_task_action_valid(self, agent, task):
        """Return whether task is a legal discrete action for this agent."""
        if task is None or task.status == 2:
            return False

        # Keep current allocation selectable so the agent can hold it.
        if len(agent.tasks) > 0 and agent.tasks[0].id == task.id:
            return True

        eligible = getattr(task, "eligible_agent_types", None)
        if eligible is not None:
            if isinstance(eligible, str):
                eligible = {eligible}
            if agent.type not in eligible:
                return False

        if self.capability_mask and agent.currentCap2Task[task.typeIdx] <= 0:
            return False

        if self.saturate_mask and task.allocatedReqs[task.typeIdx] >= task.orgReqs[task.typeIdx]:
            return False

        return True

    def get_task_info(self, agent: UAV):
        open_tasks = [task for task in self.tasks if task.status != 2]
        task_values = []
        for task in open_tasks:
            info = {
                "id": task.id,
                "position": task.position / self.max_coord,
                "status": task.status,
                "current_reqs": task.currentReqs,
                "alloc_reqs": task.allocatedReqs,
            }
            if self.include_time_windows:
                info["init_time"] = (getattr(task, "initTime", 0) - self.time_steps) / max(self.max_time_steps, 1)
                info["end_time"] = (getattr(task, "doneTime", self.max_time_steps) - self.time_steps) / max(self.max_time_steps, 1)
                info["type_idx"] = float(task.typeIdx) / 6.0
            # Dynamic features (safe extras; encoders may ignore)
            unmet = float(np.maximum(task.currentReqs[task.typeIdx] - task.allocatedReqs[task.typeIdx], 0.0))
            info["unmet"] = unmet / max(float(task.orgReqs[task.typeIdx]), 1e-6)
            created = float(getattr(task, "created_at", getattr(task, "initTime", 0)) or 0)
            info["age"] = min((self.time_steps - created) / max(self.max_time_steps, 1), 1.0)
            task_values.append(info)

        if len(task_values) == 0:
            task_values = [{
                "id": self.task_idle.id,
                "position": self.task_idle.position / self.max_coord,
                "status": self.task_idle.status,
                "current_reqs": self.task_idle.currentReqs,
                "alloc_reqs": self.task_idle.allocatedReqs,
            }]
            pad_mask = [True]
            action_mask = [True]
        else:
            # Pad mask must stay contiguous for the transformer encoder packing.
            pad_mask = [True] * len(task_values)
            action_mask = [self._is_task_action_valid(agent, task) for task in open_tasks]
            if not any(action_mask):
                current_id = agent.tasks[0].id if len(agent.tasks) > 0 else -1
                for i, task in enumerate(open_tasks):
                    if task.id == current_id:
                        action_mask[i] = True
                        break
                else:
                    action_mask[0] = True

        pad = self.max_tasks - len(task_values)
        pad_mask.extend([False] * pad)
        action_mask.extend([False] * pad)
        task_values.extend([{"status": -1} for _ in range(pad)])

        return task_values, pad_mask, action_mask

    def _event_flag_vector(self):
        """Global [fail, new_threat, reset, time_frac, n_open_norm] for dynamic conditioning."""
        fail = threat = reset = 0.0
        for ev in getattr(self, "event_list", []) or []:
            tag = ev[0] if isinstance(ev, (list, tuple)) and ev else ev
            if tag == "Agent_Fail":
                fail = 1.0
            elif tag == "New_Threat":
                threat = 1.0
            elif tag == "Reset_Allocation":
                reset = 1.0
        open_n = sum(1 for t in self.tasks if t.id != 0 and t.status != 2)
        return np.asarray(
            [
                fail,
                threat,
                reset,
                self.time_steps / max(self.max_time_steps, 1),
                open_n / max(self.max_tasks, 1),
            ],
            dtype=np.float32,
        )

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
                        
        # tasks_info + pad mask are shared (contiguous). action_mask is per-agent.
        shared_tasks_info, shared_pad_mask, _ = self.get_task_info(self.agents_obj[0])
        self.observations = {}
        for agent in self.agents_obj:
            _, _, action_mask = self.get_task_info(agent)
            if agent.state == 2:
                action_mask = [
                    True if task.get("status", -1) != -1 and task.get("id") == agent.tasks[0].id else False
                    for task in shared_tasks_info
                ]
            self.observations[agent.name] = {
                "agent_position": agent.position / self.max_coord,
                "agent_caps": agent.currentCap2Task,
                "alloc_task": agent.tasks[0].id,
                "tasks_info": shared_tasks_info,
                "mask": shared_pad_mask,
                # Named legal_mask (not action_mask) so PettingZoo/Tianshou
                # does not treat the obs Dict as the reserved action-mask layout.
                "legal_mask": action_mask,
                "event_flags": self._event_flag_vector(),
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
        self.n_reallocations = 0
        self.n_task_switches = 0
        self.n_arrivals = 0
        self._pending_reset = False
        self.agent_known_tasks = {}
        self.pending_reveals = []
        self.n_missed_windows = 0
        self.n_on_time = 0
        self.n_windowed_tasks = 0
        self._idle_reserve_steps = 0
        self._burst_region_toggle = 0
        self._next_task_id = 1
        self._next_threat_id = 0
        self.escort_requests = 0
        self.escort_completed = 0
        self.escort_failed = 0
        self.escort_required_steps = 0
        self.escort_covered_steps = 0
        self.protection_breaches = 0
        self.threats_intercepted = 0
        self.recon_losses = 0
        self.escort_losses = 0
        self.mutual_support_engagements = 0
        self.protected_rec_completed = 0
        self._escort_by_recon = {}
                        
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
        #self.rndAgentGen.shuffle(self.agents)
        
        self.agent_selector = agent_selector(self.agents)
        self.current_agent = self.agent_selector.next()

        # Initialize agents_obj with None placeholders        
        self.agents_obj: List[Optional[UAV]] = [None] * self.n_agents
        
        for agent_type, n_agents in self.agents_config.items():
            for i in range(n_agents):
                agent_id = agents_list.pop(0) #Agent_id is the position in the agents_obj
                agent = UAV(agent_id, f'{agent_type[0:2]}_agent{i}', 
                                                self.random_position(self.rndAgentGen, obstacles=self.obstacles) if self.random_init_pos else self.bases[0],
                                                agent_type,  
                                                self) 
                self.agents_obj[agent_id] = agent 
                agent.max_speed = agent.max_speed / self.simulation_frame_rate * 0.02            
                self.agent_by_name[agent.name] = agent #Agent.name is a nique name with type letter and order inside type
        
                               
        #-------------------  Define Fail Condition  -------------------#
        for agent in self.agents_obj:
            if self.rndAgentGen.random() < self.fail_rate * agent.fail_multiplier:                
                agent.fail_event = self.rndAgentGen.randint(1, 1000 if self.max_time_steps == -1 else self.max_time_steps )                               
        
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
        self.tasks: List[Optional[Task]] = []
       
        hold_tasks_num = 0
        
        for task_type, n_tasks in self.tasks_config.items(): 
            for i in range(n_tasks):
                
                selected_mission = None
                
                if self.n_mission_areas > 0:
                    selected_mission = self.rndMissionGen.choice(self.mission_areas)
                
                task_id = self._alloc_task_id()
                
                if task_type != "Hold":
                    task_position = self.random_position(self.rndTgtGen, obstacles = self.obstacles, contact_line = True, mission_area=selected_mission)
                    task_req = {task_type : 1.0} 
                                   
                else:
                    task_position = np.array([int((hold_tasks_num + 1) * self.sceneData.GameArea[0] / 5), int (self.sceneData.GameArea[1] / 4) ])                
                    hold_tasks_num += 1                      
                    task_req = {"Hold" : 1.0}        

                self.tasks.append(  Task(task_id, 
                                        task_position, 
                                        task_type, 
                                        task_req,
                                        (20, self.max_time_steps), 
                                        self.sceneData,
                                        self.max_time_steps)
                                        )
    
                
        prossible_rews = 0
        
        for task in self.tasks:
            prossible_rews += task.orgReqs[task.typeIdx] 
        
        self.reward_norm_factor = (prossible_rews * self.final_rew_factor + prossible_rews) / self.reward_multiplifier

        #---------------- Define Threats  -----------------#
          
        self.threats = [] # List to store active threats         
        self.threats_groups = [] #Store possible threats by group
        
        max_horz = self.sceneData.GameArea[0]
        max_vert = self.sceneData.GameArea[1]
        
        for ng, threat_group in enumerate(self.threats_list):

            group_pos = np.array([self.rndAgentGen.randint(int(0 + self.threat_wide), int(max_horz - self.threat_wide)), 0])
            group_type = threat_group[0]
            self.threats_groups.append([])
            
            #Create Defensive task for threat group
            task_id = self._alloc_task_id()
            task_position = np.array( [ group_pos[0] , max_vert/5 ])
            
            detect_task = Task(task_id, 
                                    task_position, 
                                    "Det", 
                                    {"Det" : threat_group[1]},
                                    (10, self.max_time_steps), 
                                    self.sceneData,
                                    self.max_time_steps)
                             
            self.tasks.append(detect_task) 
        
            
            for _ in range(threat_group[1]): 

                start_position = np.array([self.rndAgentGen.randint(int(group_pos[0] - self.threat_wide), int(group_pos[0] + self.threat_wide)), 0])
                speed = self.sceneData.maxSpeeds[group_type]
                engageRange = self.sceneData.engage_range[group_type]
                attack = self.sceneData.UavCapTable[group_type][2]
                defence = self.sceneData.UavCapTable[group_type][3]                            
                
                new_threat = Threat(
                    self._alloc_threat_id(),
                    start_position,
                    speed,
                    engageRange,
                    attack,
                    defence,
                    group_number=ng,
                    threat_type=group_type,
                )
                #TODO-> Correct frame rate handle
                new_threat.max_speed = new_threat.max_speed / self.simulation_frame_rate * 0.02 

                new_threat.relative_detect_task = detect_task
                
                self.threats_groups[ng].append(new_threat)   

        
        #Tasks that each agent is doing                        
        max_tid = max((t.id for t in self.tasks), default=0)
        self.allocation_table = [set() for _ in range(max_tid + 1)]       
                
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

        # Static / initial tasks are known to everyone; dynamic pop-ups use sensing/delay.
        static_ids = {t.id for t in self.tasks if t.id != 0}
        self.agent_known_tasks = {a.name: set(static_ids) for a in self.agents_obj}
                                    
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
           
            #Process Shared Events (drain entire queue so New_Threat / Agent_Fail are not dropped)
            done_events = []
            while self.event_list:
                event = self.event_list.pop(0)
                done_events.append(event)
                if event[0] == "Reset_Allocation":
                    self.releaseAllTasks(event[1])
                        
            #------------  TASK ALLOCATION  ----------------#
            notAllocatedTasks = len(self.unallocated_tasks())
            
            if isinstance(actions,dict):               
                                                                
                # print(f'Step{self.time_steps} - {actions}' )
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
                                action_reward += -1                                
                                continue                                                           
                        else:                            
                            #Reward: Penaulty for allocate task with wrong index
                            action_reward += -1
                            continue                                

                        #Multi agent per task and one task per agent
                        #if not self.multiple_tasks_per_agent and self.multiple_agents_per_task:
                        if self.multiple_agents_per_task:
                            
                            if len(agent.tasks) > 0:

                                #If the agent change the current task receive a small penaulty                            
                                if agent.tasks[0].id != task.id:                                    
                                    if agent.tasks[0].id != 0:                                    
                                        
                                        S_quality_reward -= 0.1                                                                        
                                        #Remove caps from last allocation
                                        S_quality_reward -= agent.currentCap2Task[agent.tasks[0].typeIdx]
                                        self.n_reallocations += 1
                                        # Open→open switch (not Hold): counts toward S_WPS rematch term
                                        if task.id != 0:
                                            self.n_task_switches += 1
                                            agent.commit_until = 0

                                        dist_old = np.linalg.norm(agent.position - agent.tasks[0].position)
                                        dist_new = np.linalg.norm(agent.position - task.position)
                                        distance_reward += (dist_old - dist_new) / self.max_coord
                                        #print(f'Srew:{S_quality_reward} | taskType {task.typeIdx} | Acgap {agent.currentCap2Task}' )                                    
                                    else: 
                                        S_quality_reward += 0.05
                                        if self._pending_reset and self.dynamic_idle_penalty:
                                            S_quality_reward -= self.dynamic_idle_penalty

                                else:

                                    if agent.tasks[0].id != 0:                                    
                                        S_quality_reward += 0.05    
                                    else:                                        
                                        S_quality_reward -= 0.50                                

                                    continue
                                                            
                            if not self.multiple_tasks_per_agent:
                                EnvUtils.desallocateAll([agent], self)
                                                            
                            if task.id == 0:
                                
                                if not self.multiple_tasks_per_agent or len(agent.tasks) == 0:
                                    S_quality_reward -= 0.05 
                                    agent.tasks = [self.task_idle]                                
                                    agent.next_free_position = agent.position
                                    agent.next_free_time = self.time_steps
                                    agent.state = 0
                                else:
                                    agent.tasks.append(self.task_idle)

                                continue

                            if not self._is_task_action_valid(agent, task):
                                action_reward += -1
                                continue
                                                            
                            if agent.allocate(task, self.time_steps) :                                                                            
 
                                
                                #self.tasks_current_quality[task_id] = self.quality_table[agent_index][task_id]                                                                      
                                self._ensure_allocation_bucket(task.id)
                                self.allocation_table[task.id].add(agent.name)
                                                                                                     
                                agentCap = agent.currentCap2Task[task.typeIdx] 
                                missingCapBefore = task.currentReqs[task.typeIdx] - (task.allocatedReqs[task.typeIdx] - agentCap)  
                                missingCapBefore = max(missingCapBefore, 0 )
                                
                                addedCap =  missingCapBefore -  np.maximum(missingCapBefore - agentCap, 0) #AddedCap
                               
                                if addedCap <= 0:
                                    S_quality_reward -= 1.5

                                
                                S_quality_reward += addedCap
                                # print(f'ACAP:{agentCap},Missing: {missingCapBefore} Rwd {S_quality_reward} / {addedCap}' )
                                task.status = 1                                

                                #task.addAgentCap(agent)
                                #action_reward += 1.0 
                                #quality_reward += self.agents_obj[agent_index].fit2Task[task.typeIdx]   # noqa: E501
                                distance_reward += self.calculate_agent_expected_reward(agent)                                                  
                                                                                                
                                #Agent state as doing task
                                if agent.state != 1 and agent.state != -1: 
                                    agent.state = 1

                                if (
                                    self.escort_enabled
                                    and task.type == "Rec"
                                    and agent.type in ("R1", "R2")
                                    and agent.name not in self._escort_by_recon
                                ):
                                    self._create_escort_for(agent, task)
                                                
                        if False: #self.multiple_tasks_per_agent and not self.multiple_agents_per_task:
                            
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
                                        time_reward = max(self.agents_obj, key=lambda obj: obj.next_free_time).next_free_time / self.max_time_steps
                                    
                                    if self.agents_obj[agent_index].state != 1 and self.agents_obj[agent_index].state != -1: 
                                        self.agents_obj[agent_index].state = 1
                            
                            else:                                
                                if notAllocatedTasks > 0:
                                    action_reward += 0
                                                                                                                               
            if self.hidden_obstacles:            
                self.detect_segments()
                
            obs_array = [[obs.position[0], obs.position[1], obs.size] for obs in self.obstacles]
            
            # Calcular as novas posições dos agents 
            for i, agent in enumerate(self.agents_obj):
                
                if isinstance(agent, UAV):                                       
                    #Check if agent is operational
                    if agent.state == -1: 
                        continue
                                        
                    if agent.fail_event == self.time_steps:
                        agent.state = -1
                        #agent.tasks = []
                        agent.desallocateAll()
                        self.event_list.append(["Reset_Allocation", -1])
                        self.event_list.append(["Agent_Fail", agent.id])
                        self._pending_reset = True
                        #print("Fail:" , agent.id, agent.fail_event )
                        #print(self.agents_obj[agent.id].tasks)
                        continue
                    
                    movement = np.array([0,0])
                    avoid_vector = np.array([0,0]) 

                    #----------------IDLE / RETURN TO BASE-----------------#
                    if agent.state == 0 and not agent.re_eval:
                        # Once an agent finishes its mission (no assigned task) it flies
                        # back to base. It stays available: if the allocator hands it a new
                        # task, allocate() switches it to state 1 and it diverts to the task.
                        idle_task = len(agent.tasks) == 0 or agent.tasks[0].id == 0
                        if idle_task and np.linalg.norm(agent.position - self.bases[0]) > agent.max_speed + 5:
                            agent.state = 3                                     
                
                    #------------------- HAS TASK ------------------#                                                                                
                    if len(agent.tasks) > 0 or agent.re_eval:
                        
                        if agent.re_eval:
                            #Hold last task up to new allocation
                            current_task = agent.last_task
                        else:
                            current_task = agent.tasks[0]

                        if current_task.status == 2:                            
                            agent.desAllocate(current_task)
                            agent.re_eval = False
                            agent.last_task = None
                        
                        elif current_task.id != 0:

                            #----------------NAVIGATING-----------------#
                            if agent.state == 1: 
                            
                                dir_task = current_task.position - agent.position
                                distance_task  = np.linalg.norm(dir_task) 
                                
                                if abs(distance_task) < EPS:                                    
                                    dir_task_norm = 0  
                                else:
                                    dir_task_norm = dir_task / distance_task
                                          

                                # Limit to consider Task Aerea Reached
                                #print(f'Agent{agent.id} in Task {current_task.id} dist {distance_task}')
                                if current_task.type == "Int":
                                    
                                    if distance_task < agent.engage_range:
                                        agent.state = 2
                                        current_task.relative_threat.target_agent = agent
                                        agent.task_start = self.time_steps
                                    else:
                                        movement = dir_task_norm 
                                        avoid_vector = core_sim.SimCore.avoid_obstacles(agent.position.tolist(), obs_array, movement.tolist())
                                        avoid_vector = np.array(avoid_vector)
                                        
                                
                                elif distance_task < agent.max_speed:                      
                                    
                                    agent.state = 2
                                    #agent.task_start = self.time_steps
                                    agent.task_start = self.time_steps
                                    agent.position = current_task.position  
                                    #print(f'Agent{agent.id} Started {current_task.id}')  
                                else:

                                    movement = dir_task_norm 
                                    avoid_vector = core_sim.SimCore.avoid_obstacles(agent.position.tolist(), obs_array, movement.tolist())
                                    avoid_vector = np.array(avoid_vector)
                                    
                            #----------------IN TASK-------------------#
                            elif agent.state == 2: 
                                                                                                                
                                if current_task.type == "Int":
                                    
                                    dir_task = current_task.position - agent.position
                                    distance_task  = np.linalg.norm(dir_task) 
                                    if distance_task >= agent.engage_range:
                                        agent.state = 1
                                    
                                
                                #Just Started the Task
                                if agent.task_start == -1:                                            
                                    
                                    agent.task_start = self.time_steps
                                    agent.position = current_task.position                                    
                                    
                                else:
                                    
                                    #Check if TASK is CONCLUDED
                                    #print(f'Agent{agent.id} in Task {current_task.id} DoneSteps {self.time_steps - agent.task_start}')
                                    if (self.time_steps - agent.task_start) >= current_task.task_duration and \
                                        current_task.id != 0 and\
                                        current_task.type != "Hold" and\
                                        current_task.type != "Def" and\
                                        current_task.type != "Int" and\
                                        current_task.type != "Det" and\
                                        current_task.status != 2:
                                                                                                                   
                                        task = current_task

                                        agent.taskDone(task)
                                                                                
                                        task.doneReqs += agent.currentCap2Task
                                        task.currentReqs -= agent.currentCap2Task   
                                        
                                        task.removeAgentCap(agent)                                                                

                                        #print(f'Done: {task.doneReqs[task.typeIdx]} | {task.orgReqs[task.typeIdx]}')                                      
                                        if task.doneReqs[task.typeIdx] >= task.orgReqs[task.typeIdx]:                                                                                            

                                            if getattr(task, "kind", None) != "Escort":
                                                self.reached_tasks.add(task.id) 
                                            
                                            # Reward if just concluded the task
                                            if task.status != 2:                                            
                                                quality_reward += task.orgReqs[task.typeIdx] * 2 
                                                
                                                self.F_Reward += task.orgReqs[task.typeIdx] * self.final_rew_factor / self.reward_norm_factor
                                                if getattr(task, "kind", None) != "Escort":
                                                    self._wps_mark_window_outcome(task, success=True)
                                                task.status = 2
                                                if task.type == "Rec" and agent.type in ("R1", "R2"):
                                                    self._on_protected_rec_done(agent, success=True)
                                                if all(self._counts_for_mission_done(t) for t in self.tasks):
                                                    self.conclusion_time = self.time_steps
                                        else:
                                            quality_reward += agent.currentCap2Task[task.typeIdx]  
                                                                                                                           
                                    else:
                                        movement = agent.doTask(self.agent_directions[i], None, None, current_task.type)


                    #----------------RETURNING BASE (NO TASK)--------------------
                    if agent.state == 3:
                                                                
                        if np.linalg.norm(agent.position - self.bases[0]) < agent.max_speed + 5:
                            agent.state = 0
                        else:
                            movement = EnvUtils.norm_vector(self.bases[0]  - agent.position)
                            avoid_vector = core_sim.SimCore.avoid_obstacles(agent.position.tolist(), obs_array, movement.tolist())
                            avoid_vector = np.array(avoid_vector)
                                                                                                                        
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
            self.inject_dynamic_arrivals()
            if self.escort_enabled:
                self._sync_escorts()
            self._wps_update_sensing()
            self._wps_process_reveals()
            self._wps_expire_windows()
            self._wps_track_reserve()
            if self._pending_reset and any(
                len(a.tasks) > 0 and a.tasks[0].id != 0 for a in self.agents_obj if a.state != -1
            ):
                # Agents started responding after a dynamic reset.
                self._pending_reset = False
                                                                                                                                                                                                       
            rw = self.reward_weights
            # rewards for all agents are placed in the rewards dictionary to be returned
            self.rewards = {
                agent.name: (
                    rw["action"] * action_reward
                    + rw["distance"] * distance_reward
                    + rw["quality"] * quality_reward
                    + rw["s_quality"] * S_quality_reward
                    + rw["time"] * self.n_tasks * time_reward
                    + rw["alloc"] * alloc_reward
                    + rw["time_penaulty"] * time_penaulty
                    + rw["step"] * self.step_reward
                )
                / self.reward_norm_factor
                / self.max_time_steps
                for agent in self.agents_obj
            }
                                                                                      
            # Mission complete ignores persistent Det/Hold/Escort bookkeeping tasks.
            all_done = len(self.tasks) > 0 and all(self._counts_for_mission_done(t) for t in self.tasks)
            timed_out = (self.time_steps >= self.max_time_steps) and (self.max_time_steps > 0)
            done = timed_out or (self.early_terminate and all_done)

            if all_done and self.conclusion_time > self.max_time_steps:
                self.conclusion_time = self.time_steps
                                  
            # True episode success vs time-limit truncation
            self.terminations = {agent.name: (self.early_terminate and all_done and not timed_out) for agent in self.agents_obj}
            env_truncation = timed_out
            self.truncations = {agent.name: env_truncation for agent in self.agents_obj}     
            
            self.infos = {agent.name: {} for agent in self.agents_obj}            
            self.infos['selected'] = self.agent_selection
            self.infos['events'] = done_events            

            self._generate_observations()                                               

            if done:
                metrics =  self.calculate_metrics() 
                self.infos['metrics'] = metrics                
                self.rewards = {agent.name :  self.F_Reward for agent in self.agents_obj} #Rand -28                
                
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
                
        if total_distance > 0:
            F_distance = 1 / total_distance * self.max_coord
        else:
            F_distance = 0

        reward_weigths = [0.0, 1.0, 0.0]#[0.25, 0.6, 0.15]
        
        #self.F_Reward = reward_weigths[0] * F_distance/0.06 + reward_weigths[1] * F_quality/0.9 + reward_weigths[2] * F_Time / 1.4

        self.F_Reward = self.F_Reward 

        Losses = len([agent for agent in self.agents_obj if agent.state == -1])
        Kills = len([threat for threat in self.threats if threat.status == 2])


        # if self.conclusion_time == (self.max_time_steps + 1):
        #     #print("Fail to Conclude")
        #     self.F_Reward = 0
        #     F_Time = 0

        #print( f'Q:{F_quality}|D:{F_distance}|T:{F_Time}|F:{self.F_Reward}|Rem:{self.allocation_table.count(-1)}' )

        s_wps = self.compute_s_wps()
        escort_cov = float(
            self.escort_covered_steps / max(self.escort_required_steps, 1)
        )
        s_esc = (
            float(s_wps)
            + 20.0 * float(self.protected_rec_completed)
            - 30.0 * float(self.recon_losses)
            + 20.0 * escort_cov
        )

        return { 
            "F_time": F_Time ,
            "F_distance": F_distance ,
            "F_quality": F_quality,
            "F_Reward": self.F_Reward,
            "S_WPS": float(s_wps),
            "S_ESC": float(s_esc),
            "Losses": Losses,
            "Kills": Kills,
            "makespan": float(self.conclusion_time),
            "total_distance": float(self.total_distance),
            "n_reallocations": int(self.n_reallocations),
            "n_task_switches": int(getattr(self, "n_task_switches", 0)),
            "n_arrivals": int(self.n_arrivals),
            "n_tasks_final": int(len(self.tasks)),
            "n_reached": int(len(self.reached_tasks)),
            "n_missed_windows": int(self.n_missed_windows),
            "n_on_time": int(self.n_on_time),
            "n_windowed_tasks": int(self.n_windowed_tasks),
            "on_time_rate": float(self.n_on_time / max(self.n_on_time + self.n_missed_windows, 1)),
            "reserve_idle_fraction": float(
                self._idle_reserve_steps / max(self.time_steps * max(len(self.agents_obj), 1), 1)
            ),
            "escort_coverage_rate": escort_cov,
            "protected_rec_completed": int(self.protected_rec_completed),
            "recon_losses": int(self.recon_losses),
            "escort_losses": int(self.escort_losses),
            "threats_intercepted": int(self.threats_intercepted),
            "mutual_support_engagements": int(self.mutual_support_engagements),
            "protection_breaches": int(self.protection_breaches),
            "escort_requests": int(self.escort_requests),
            "escort_completed": int(self.escort_completed),
            "escort_failed": int(self.escort_failed),
        }

    def compute_s_wps(self) -> float:
        """Primary WPS score: on-time completions minus misses minus travel / rematch cost."""
        on_w = float(getattr(self, "on_time_bonus", 12.0) or 12.0)
        miss_w = float(getattr(self, "miss_penalty", 30.0) or 30.0)
        # Prefer fixed paper scales from WPS_hard for cross-case comparability
        on_w = 12.0
        miss_w = 30.0
        dist_term = 0.01 * float(self.total_distance) / max(float(self.max_coord), 1.0)
        rematch = float(getattr(self, "reassign_penalty", 0.0) or 0.0) * float(
            getattr(self, "n_task_switches", 0)
        )
        return (
            on_w * float(self.n_on_time)
            - miss_w * float(self.n_missed_windows)
            - dist_term
            - rematch
        )

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
    
            if  obstacles is not None:
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
    
    def releaseAllTasks(self, for_task_type):

        available_agents = set()        
        for agent in self.agents_obj:
            
            if agent.currentCap2Task[for_task_type] > 0:
                                                
                if agent.state != -1:                
                    
                    if len(agent.tasks) > 0:
                        agent.re_eval = True
                        agent.last_task = agent.tasks[0]
                    
                    agent.desallocateAll()
                    available_agents.add(agent.typeIdx)

        for task in self.tasks:
            
            if task.status != 2 and task.typeIdx == for_task_type :
                              
                cum_cap = 0
                for agentType in available_agents:
                    cum_cap += self.sceneData.UavCapTableByIdx[agentType][task.typeIdx]                
                
                if cum_cap == 0:                   
                    task.status = 2 #No capability to DO it
                    
                    # Adicionar o alvo para a lista alcançados por não ser mais possível
                    # E não prejudicar algoritmos que alocam errado
                    if task.id not in self.reached_tasks:                        
                        self.reached_tasks.add(task.id) 
                        task.status = 2    

                        if len(self.reached_tasks) == self.n_tasks:
                            self.conclusion_time = self.time_steps

                else:
                    task.status = 0                        
                    self.allocation_table[task.id] = set()                            
                 
        # self.tasks_current_quality = {i: -1 for i in range(self.n_tasks)}  
    
    def get_live_agents(self):

        return [agent for agent in self.agents_obj if agent.state != -1]  
       

####---------------------Dynamic Conditions ----------------------------------###

    def _register_dynamic_task(self, task):
        """Apply hard window + delayed/local reveal for a newly created pop-up task."""
        if self.hard_windows and getattr(task, "hard_deadline", None) is None:
            task.hard_deadline = self.time_steps + self.window_length
            task.task_window = (self.time_steps, task.hard_deadline)
            self.n_windowed_tasks += 1
        # Delayed global reveal, then local sensing can discover earlier for nearby agents
        if self.threat_delay > 0 or self.sense_radius > 0:
            reveal_t = self.time_steps + max(self.threat_delay, 0)
            self.pending_reveals.append((reveal_t, task.id))
        else:
            # Instant global knowledge (legacy)
            for name in self.agent_known_tasks:
                self.agent_known_tasks[name].add(task.id)

    def _wps_update_sensing(self):
        if self.sense_radius <= 0:
            return
        for agent in self.agents_obj:
            if agent.state == -1:
                continue
            known = self.agent_known_tasks.setdefault(agent.name, set())
            for task in self.tasks:
                if task.id == 0 or task.status == 2:
                    continue
                if task.id in known:
                    continue
                # Only sense dynamic / pending tasks (not already globally known)
                if getattr(task, "created_at", 0) <= 0 and getattr(task, "hard_deadline", None) is None:
                    continue
                dist = float(np.linalg.norm(agent.position - task.position))
                if dist <= self.sense_radius:
                    known.add(task.id)

    def _wps_process_reveals(self):
        if not self.pending_reveals:
            return
        # Asymmetric sensing: only local radius discovers tasks (no team-wide reveal).
        if not self.share_knowledge:
            self.pending_reveals = [
                (reveal_t, tid) for reveal_t, tid in self.pending_reveals if self.time_steps < reveal_t
            ]
            return
        remaining = []
        for reveal_t, tid in self.pending_reveals:
            if self.time_steps >= reveal_t:
                for name in self.agent_known_tasks:
                    self.agent_known_tasks[name].add(tid)
            else:
                remaining.append((reveal_t, tid))
        self.pending_reveals = remaining

    def _wps_mark_window_outcome(self, task, success: bool):
        """Count on-time vs missed once per hard-windowed task."""
        if getattr(task, "hard_deadline", None) is None:
            return
        if getattr(task, "_wps_outcome_counted", False):
            return
        task._wps_outcome_counted = True
        if success and self.time_steps <= task.hard_deadline:
            self.n_on_time += 1
            self.F_Reward += self.on_time_bonus
        else:
            self.n_missed_windows += 1
            self.F_Reward -= self.miss_penalty

    def _wps_expire_windows(self):
        if not self.hard_windows:
            return
        for task in self.tasks:
            deadline = getattr(task, "hard_deadline", None)
            if deadline is None or task.status == 2 or task.id == 0:
                continue
            if self.time_steps > deadline:
                task.status = 2
                task.final_quality = 0.0
                self._wps_mark_window_outcome(task, success=False)
                if task.id not in self.reached_tasks:
                    self.reached_tasks.add(task.id)
                # Free agents still assigned to this expired task
                for agent in self.agents_obj:
                    if agent.tasks and agent.tasks[0].id == task.id:
                        agent.desallocateAll()

    def _wps_track_reserve(self):
        live = [a for a in self.agents_obj if a.state != -1]
        if not live:
            return
        idle = sum(1 for a in live if (not a.tasks) or a.tasks[0].id == 0)
        self._idle_reserve_steps += idle

    def known_tasks_for(self, agent_name=None):
        """Tasks visible to one agent, or union of team knowledge if agent_name is None."""
        if agent_name is not None:
            ids = self.agent_known_tasks.get(agent_name, set())
            return [t for t in self.tasks if t.id in ids or t.id == 0]
        # Team union (shared local knowledge for decentralized planners)
        known = set()
        for s in self.agent_known_tasks.values():
            known |= s
        if not self.sense_radius and not self.threat_delay:
            return list(self.tasks)
        return [t for t in self.tasks if t.id in known or t.id == 0]

    def agent_visibility_map(self):
        """Per-agent known task id sets for visibility-masked assignment."""
        if not self.sense_radius and not self.threat_delay:
            return None
        return {name: set(ids) for name, ids in self.agent_known_tasks.items()}

    def generate_threat(self):
        
        for group in self.threats_groups:
            
            if len(group) > 0 and self.time_steps > 40 and self.time_steps % 10 == 0:
                
                if self.rndAgentGen.random() < self.threat_generation_probability:
                    n_spawn = 1
                    if self.burst_mode:
                        n_spawn = min(self.burst_size, len(group))
                    for bi in range(n_spawn):
                        if not group:
                            break
                        threat = group.pop(0)
                        if self.dual_region_bursts:
                            # Alternate left/right fronts within a burst
                            mid = self.area_width * 0.5
                            wide = max(self.threat_wide, 40.0)
                            if (self._burst_region_toggle + bi) % 2 == 0:
                                x = float(self.rndAgentGen.uniform(wide, mid - wide))
                            else:
                                x = float(self.rndAgentGen.uniform(mid + wide, self.area_width - wide))
                            threat.position = np.array([x, float(threat.position[1])])
                        threat.target_agent = self.get_closest_agent(threat.position)
                        threat.mission_target_agent = threat.target_agent

                        relative_task_id = self._alloc_task_id()

                        relative_task = self.TaskFromThreat(relative_task_id, threat)
                        
                        self.tasks.append(relative_task)
                        self._ensure_allocation_bucket(relative_task_id)
                        threat.relative_task = relative_task

                        self.threats.append(threat)

                        threat.relative_detect_task.currentReqs[5] -= 1.0
                        self._register_dynamic_task(relative_task)
                        self.event_list.append(["New_Threat", relative_task.id])
                        self.event_list.append(["Reset_Allocation", relative_task.typeIdx])
                        self._pending_reset = True
                    if self.dual_region_bursts and n_spawn > 0:
                        self._burst_region_toggle = (self._burst_region_toggle + 1) % 2
                   

    def inject_dynamic_arrivals(self):
        """Online Rec/Att task arrivals for dynamic TA experiments."""
        if self.arrival_rate <= 0 or self.time_steps < 5:
            return
        if self.rndTgtGen.random() >= self.arrival_rate:
            return
        if len(self.tasks) >= self.max_tasks - 1:
            return

        task_type = self.rndTgtGen.choice(["Att", "Rec"])
        task_id = self._alloc_task_id()
        selected_mission = self.rndMissionGen.choice(self.mission_areas) if self.mission_areas else None
        if self.dual_region_bursts:
            mid = self.area_width * 0.5
            wide = 40.0
            if self.rndTgtGen.random() < 0.5:
                x = float(self.rndTgtGen.uniform(wide, mid - wide))
            else:
                x = float(self.rndTgtGen.uniform(mid + wide, self.area_width - wide))
            y = float(self.rndTgtGen.uniform(self.area_height * 0.2, self.area_height * 0.8))
            position = np.array([x, y])
        else:
            position = self.random_position(
                self.rndTgtGen, obstacles=self.obstacles, contact_line=True, mission_area=selected_mission
            )
        from .DroneEnvComponents import Task

        new_task = Task(
            task_id,
            position,
            task_type,
            {task_type: 1.0},
            (self.time_steps, self.max_time_steps),
            self.sceneData,
            self.max_time_steps,
        )
        new_task.created_at = self.time_steps
        self.tasks.append(new_task)
        self._ensure_allocation_bucket(task_id)
        self.n_arrivals += 1
        self._register_dynamic_task(new_task)
        self.event_list.append(["New_Threat", task_id])  # treated as online arrival for replan hooks
        self.event_list.append(["Reset_Allocation", new_task.typeIdx])
        self._pending_reset = True

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
        if False:
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
        
        for threat in [threat for threat in self.threats if threat.status != 2]:
            
            if threat.status == 0 or threat.target_agent is None:
                direction = np.array([0,-1])                
                threat.position = threat.position + threat.max_speed * direction
            else:
                if self.escort_enabled:
                    self._retarget_threat_via_escort(threat)
                direction = EnvUtils.norm_vector(threat.target_agent.position  - threat.position)
                threat.position = threat.position + threat.max_speed * direction

                if np.linalg.norm(np.array(threat.target_agent.position) - np.array(threat.position)) < threat.engage_range:
                    self.handle_threat_engagement(threat)
            threat.relative_task.position = threat.position

            if threat.position[1] <= 0:
                threat.relative_task.status = 2
                self._wps_mark_window_outcome(threat.relative_task, success=False)

    def _escort_fighters_near(self, protected_agent, radius=None):
        """Fighters currently on the protected agent's escort task and within radius."""
        if protected_agent is None:
            return []
        escort = self._escort_by_recon.get(protected_agent.name)
        if escort is None or escort.status == 2:
            return []
        r = float(self.escort_radius if radius is None else radius)
        nearby = []
        for agent in self.agents_obj:
            if agent.state == -1 or agent.type not in self.escort_agent_types:
                continue
            if not agent.tasks or agent.tasks[0].id != escort.id:
                continue
            dist = float(np.linalg.norm(agent.position - protected_agent.position))
            if dist <= r:
                nearby.append((dist, agent))
        nearby.sort(key=lambda x: x[0])
        return [a for _, a in nearby]

    def _retarget_threat_via_escort(self, threat: Threat):
        mission = threat.mission_target_agent or threat.target_agent
        if mission is None or mission.state == -1:
            return
        if mission.type not in ("R1", "R2"):
            return
        escorts = self._escort_fighters_near(mission, self.escort_intercept_radius)
        if not escorts:
            threat.target_agent = mission
            threat.intercepting_agent = None
            return
        primary = escorts[0]
        threat.target_agent = primary
        threat.intercepting_agent = primary

    def handle_threat_engagement(self, threat: Threat):
        defenders = []
        primary = threat.target_agent
        mission = threat.mission_target_agent or primary

        if self.escort_enabled and mission is not None and mission.type in ("R1", "R2"):
            defenders = self._escort_fighters_near(mission, self.mutual_support_radius)
            if defenders:
                primary = defenders[0]
                threat.target_agent = primary
                threat.intercepting_agent = primary

        if primary is None:
            return

        if len(defenders) >= 2:
            self.mutual_support_engagements += 1
            att_sum = sum(float(a.currentCap2Task[2]) for a in defenders)
            def_sum = sum(float(a.currentCap2Task[3]) for a in defenders)
            eng_sum = sum(float(a.engage_range) for a in defenders) / len(defenders)
            attDiff = att_sum / max(threat.attack, 1e-6)
            defDiff = def_sum / max(threat.defence, 1e-6)
            engageDiff = eng_sum / max(threat.engage_range, 1e-6)
        else:
            attDiff = primary.currentCap2Task[2] / max(threat.attack, 1e-6)
            defDiff = primary.currentCap2Task[3] / max(threat.defence, 1e-6)
            engageDiff = primary.engage_range / max(threat.engage_range, 1e-6)

        avg_diff = (attDiff + defDiff + engageDiff) / 3                
        neutralize_prob = avg_diff / (avg_diff + 1)  

       
        rnd = self.rndAgentGen.random()
                       
        if rnd < neutralize_prob:        
            
            threat.status = 2
            threat.relative_task.status = 2
            self._wps_mark_window_outcome(threat.relative_task, success=True)
            self.threats_intercepted += 1
            
            primary.attackCap -= 1
            
            if primary.attackCap <= 0:
                primary.currentCap2Task[3] = 0
            
            if primary.tasks and primary.tasks[0] == threat.relative_task:
                primary.taskDone(threat.relative_task)
            
            self.step_reward += 1.0

        else:            
                                           
            threat.attackCap -= 1  

            primary.attackCap -= 1
            
            if primary.attackCap <= 0:
                primary.currentCap2Task[3] = 0 
                was_recon = primary.type in ("R1", "R2")
                was_escort = primary.type in self.escort_agent_types
                primary.outOfService()
                if was_recon:
                    self.recon_losses += 1
                    self.protection_breaches += 1
                    self._retire_escort_for(primary, failed=True)
                elif was_escort:
                    self.escort_losses += 1
                self.step_reward -= 1.0                             
            
            
            if threat.attackCap <= 0:
                threat.status = 0
                threat.relative_task.status = 2
                self._wps_mark_window_outcome(threat.relative_task, success=False)
            else:
                threat.target_agent = self.get_closest_agent(threat.position)
                threat.mission_target_agent = threat.target_agent
            
    
    def TaskFromThreat(self, task_id, threat: Threat):

        new_task = Task( task_id, threat.position, 
                        "Int", 
                        {"Int": 2.0, "Att" : threat.defence * 2, "Def" : threat.attack * 2}, 
                        (self.time_steps, self.max_time_steps), 
                        self.sceneData, 
                        self.max_time_steps,  
                        is_active=True,
                        relative_threat = threat )
        new_task.created_at = self.time_steps
        # Strong threats (T1) need two fighters; T2 stays single-slot.
        if getattr(threat, "threat_type", None) == "T1":
            new_task.required_agents = 2
            new_task.eligible_agent_types = set(self.escort_agent_types)
        return new_task

    def _counts_for_mission_done(self, task) -> bool:
        """Persistent bookkeeping tasks do not block episode completion."""
        if task.id == 0:
            return True
        if getattr(task, "kind", None) == "Escort":
            return True
        if task.type in ("Det", "Hold"):
            return True
        return task.status == 2

    def _create_escort_for(self, recon_agent, rec_task):
        if not self.escort_enabled or recon_agent is None:
            return None
        if recon_agent.name in self._escort_by_recon:
            return self._escort_by_recon[recon_agent.name]

        escort = Task(
            self._alloc_task_id(),
            np.array(recon_agent.position, dtype=float),
            "Def",
            {"Def": float(self.escort_requirement)},
            (self.time_steps, self.max_time_steps),
            self.sceneData,
            self.max_time_steps,
        )
        escort.kind = "Escort"
        escort.protected_agent = recon_agent
        escort.protected_task = rec_task
        escort.eligible_agent_types = set(self.escort_agent_types)
        escort.required_agents = max(2, int(np.ceil(self.escort_requirement)))
        escort.created_at = self.time_steps
        self.tasks.append(escort)
        self._ensure_allocation_bucket(escort.id)
        self._register_dynamic_task(escort)
        self._escort_by_recon[recon_agent.name] = escort
        self.escort_requests += 1
        self.event_list.append(["Escort_Created", escort.id])
        self.event_list.append(["Reset_Allocation", escort.typeIdx])
        self._pending_reset = True
        return escort

    def _release_escort_agents(self, escort_task):
        if escort_task is None:
            return
        escort_id = int(escort_task.id)
        for agent in list(self.agents_obj):
            if agent.state == -1:
                continue
            held = [t for t in list(agent.tasks) if t.id == escort_id]
            for task in held:
                agent.desAllocate(task)
            if held:
                # Clear idle hold so RTB can start on the next idle check
                if not agent.tasks or agent.tasks[0].id == 0:
                    agent.tasks = [self.task_idle]
                    agent.state = 0
                    agent.commit_until = 0
                    agent.next_free_time = self.time_steps
                    agent.next_free_position = agent.position

    def _retire_escort(self, escort_task, failed: bool = False):
        if escort_task is None or escort_task.status == 2:
            return
        self._release_escort_agents(escort_task)
        escort_task.status = 2
        recon = escort_task.protected_agent
        if recon is not None:
            self._escort_by_recon.pop(getattr(recon, "name", None), None)
        if failed:
            self.escort_failed += 1
        else:
            self.escort_completed += 1
        self.event_list.append(["Escort_Retired", escort_task.id])

    def _retire_escort_for(self, recon_agent, failed: bool = False):
        if recon_agent is None:
            return
        escort = self._escort_by_recon.get(recon_agent.name)
        if escort is not None:
            self._retire_escort(escort, failed=failed)

    def _on_protected_rec_done(self, recon_agent, success: bool = True):
        if success:
            self.protected_rec_completed += 1
        self._retire_escort_for(recon_agent, failed=not success)

    def _sync_escorts(self):
        """Update escort positions, coverage metrics, and retire stale escorts."""
        # Create escorts for recon already on Rec (e.g. after reset/replan)
        for agent in self.agents_obj:
            if agent.state == -1 or agent.type not in ("R1", "R2"):
                continue
            if not agent.tasks or agent.tasks[0].id == 0:
                continue
            cur = agent.tasks[0]
            if cur.type == "Rec" and cur.status != 2 and agent.name not in self._escort_by_recon:
                self._create_escort_for(agent, cur)

        # Retire escorts whose protected Rec ended or agent is down / idle
        for name, escort in list(self._escort_by_recon.items()):
            recon = escort.protected_agent
            rec_task = escort.protected_task
            dead = recon is None or recon.state == -1
            idle = recon is not None and (
                not recon.tasks or recon.tasks[0].id == 0 or recon.state in (0, 3)
            )
            rec_done = rec_task is not None and rec_task.status == 2
            wrong_task = (
                recon is not None
                and recon.tasks
                and recon.tasks[0].id != 0
                and (rec_task is None or recon.tasks[0].id != rec_task.id)
            )
            if dead or idle or rec_done or wrong_task:
                self._retire_escort(escort, failed=dead)
                continue

            # Follow protected UAV (do not snap assigned escorts onto the target)
            escort.position = np.array(recon.position, dtype=float)
            self.escort_required_steps += 1
            nearby = self._escort_fighters_near(recon, self.escort_radius)
            if nearby:
                self.escort_covered_steps += 1

    def compute_s_esc(self) -> float:
        escort_cov = float(
            self.escort_covered_steps / max(self.escort_required_steps, 1)
        )
        return (
            float(self.compute_s_wps())
            + 20.0 * float(self.protected_rec_completed)
            - 30.0 * float(self.recon_losses)
            + 20.0 * escort_cov
        )
          


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

