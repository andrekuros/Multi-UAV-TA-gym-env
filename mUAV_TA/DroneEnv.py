import gym
from gym import spaces
from gym.spaces import Dict, Discrete, MultiDiscrete, Box
import numpy as np
import random
import copy
import sys

import matplotlib.pyplot as plt
import seaborn as sns
#from pygame_screen_recorder import pygame_screen_recorder as pgr

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.utils import agent_selector

from .DroneEnvComponents import Drone, Task, Obstacle
from .MultiDroneEnvData import SceneData 
from .MultiDroneEnvUtils import DroneEnvOptions, DroneEnvUtils 

import pygame

MAX_INT = sys.maxsize


def env(config = None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    #internal_render_mode = config.render_mode if config.render_mode != "ansi" else "human"
    #env = raw_env(internal_render_mode, action_mode,  render_speed, max_time_steps, n_agents, n_tasks)
    env = MultiDroneEnv(config)
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
    env = MultiDroneEnv(config)
    env = parallel_to_aec(env)
    return env

class MultiDroneEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multi_drone_env_v0"}
    
    def __init__(self, config=None ):
        
        super(MultiDroneEnv, self).__init__()        
        
        if config == None:
            self.config = DroneEnvOptions()
        else:
            self.config = config              
            
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
        self.conclusion_time = self.max_time_steps

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
        self.max_agents = 10

        self.agent_selection = 'agent0'
        
        
        self.agents_obj = None
        self.possible_agents = ["agent" + str(i) for i in range(self.n_agents)]          
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agents = self.possible_agents

        self.agent_selector = agent_selector(self.possible_agents)
        self.current_agent = self.agent_selector.next()
                
        self.max_tasks = 30
        self.n_tasks = sum(self.config.tasks.values())
        self.tasks_config = self.config.tasks               
        self.tasks = None
        self.last_tasks_info = None
        
        #Dynamic Conditions        
        self.hidden_obstacles = self.config.hidden_obstacles        
        
        self.num_obstacles = self.config.num_obstacles        
        self.obstacles = None
        
        self.fail_rate = self.config.fail_rate         
        
        self.tasks_current_quality = None        
        self.allocation_table = None
                                               
        #Table with the quality of each drone for each task
        self.quality_table = None
                
        self.time_steps = 0
        self.total_distance = 0
        self.drone_distances = None
        self.drone_directions = None 

        self.F_Reward = 0 

        self.event_list = []      
                
        self.previous_drones_positions = None
        self.previous_drones_positions = None
                        
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
            "agent_type": Box(low=0, high=1, shape=(2,), dtype=np.float32),  # One-hot encoded agent types  # Assuming 2 possible types
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

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    def get_task_info(self, agent: Drone):

        task_values = [ {
                
                "id": task.task_id,
                "position": task.position / self.max_coord,
                "type": task.typeIdx,
                "status": task.status,                
                "quality": task.final_quality                
                }  for task in self.tasks
            ]
            
        

        # for task in self.tasks:
        #     if task.status != 0:
        #         continue
        #     else:
        #         #print("taks", task.task_id)
        #         distance = self.euclidean_distance(agent.next_free_position, task.position)  # Compute the distance
                
        #         #task_values.extend(self._one_hot(task.typeIdx, 2))

        #         task_values.extend([
        #             distance / self.max_coord,  # Normalize the distance
        #             agent.fit2Task[task.typeIdx],
        #             #task.position[0] / self.max_coord,
        #             #task.position[1] / self.max_coord,

        #             1 if task.status == 0 else 0,                

        #             ])
                
                
        
        # Pad the task_values array to match the maximum number of tasks
        #task_values.extend([-1] * (self._observation_spaces["agent0"]["tasks_info"].shape[0] - len(task_values)))

        #task_values = np.array(task_values, dtype=np.float32)
        
        #print(task_values)
        return task_values

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

                ])
            
        #Pad with -1 if the number of agents is smaller than the max_agents
        agents_values.extend([-1] * (self._observation_spaces["agent0"]["agents_info"].shape[0] - len(agents_values)))
        
        agents_values = np.array(agents_values, dtype=np.float32)  

        return agents_values
                

    def observation_space(self, agent):
           return self._observation_spaces[agent]  
    
    def _generate_observations(self):
                
        #self.last_tasks_obs = []
        #task_values.extend([-1] * (self.observation_space["agent0"]["tasks_info"].shape[0]))
        agents_info = self.get_agents_info()

        self.observations = {
            agent.name : {
                #Add Own Data for relative features
                "agent_id": agent.drone_id,
                "agent_position": agent.position / self.max_coord,
                "agent_state": self._one_hot(agent.state, 5),
                "agent_type": self._one_hot(agent.typeIdx, 6),
                "next_free_time": [agent.next_free_time / self.max_time_steps],
                "position_after_last_task": agent.next_free_position / self.max_coord,                
                #"agent_relay_area": agent.relay_area,

                #Complete data for tasks and agents
                "tasks_info": self.get_task_info(agent),   
                "agents_info": agents_info,              
                
            }
            for agent in self.agents_obj
        }

        self.last_tasks_info = [task.task_id for task in self.tasks if task.status == 0]            


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

    def reset(self, seed = 0 , return_info = True, options={"options":1}):
        
        self._seed = seed
        self.rndAgentGen = random.Random(seed)
        self.rndObsGen = random.Random(self.rndAgentGen.randint(0, MAX_INT))
        self.rndTgtGen = random.Random(self.rndAgentGen.randint(0, MAX_INT))        
        #print("Seed:",self._seed)
        #print (self.rndAgentGen.random(),self.rndObsGen.random(),self.rndTgtGen.random())
                                
        if self.agents_obj != None:
           self.agents_obj.clear() 
        if self.tasks != None:
            self.tasks.clear() 
        if self.obstacles != None:
            self.obstacles.clear()
        
        self.F_Reward = 0 

        self.agents = self.possible_agents                             
                        
        #-------------------  Define Obstacles  -------------------#
        self.obstacles = []
        for _ in range(self.num_obstacles):
            size = self.rndObsGen.randint(30, 100)
            position = self.random_position(self.rndObsGen, obstacles = self.obstacles, own_range = size , contact_line = True)            
            self.obstacles.append(Obstacle(position, size))

                                                    
        #-------------------  Init Agents  -------------------#
        self.agents_obj = []
        n_previous = 0
        for agent_type, n_agents in self.agents_config.items():
            n_previous = len(self.agents_obj)
            self.agents_obj += [Drone(n_previous + i, self.agents[i + n_previous], 
                                      self.random_position(self.rndAgentGen, obstacles = self.obstacles) if self.random_init_pos else self.bases[0],
                                      agent_type, self.sceneData) for i in range(n_agents)]
            #self.agents_obj += [Drone(n_previous + i, self.agents[i + n_previous], self.bases[0], agent_type, self.sceneData) for i in range(n_agents)]
            
                        
        
        #-------------------  Define Fail Condition  -------------------#
        for agent in self.agents_obj:
            if self.rndAgentGen.random() < self.fail_rate * agent.fail_multiplier:                
                agent.fail_event = self.rndAgentGen.randint(1, 1000 if self.max_time_steps == -1 else self.max_time_steps )
                #print (f'Agent({agent.drone_id}): {agent.fail_event}')
               
        
        #-------------------  Define Tasks  -------------------#
        self.tasks = []
        for task_type, n_tasks in self.tasks_config.items(): 
            len_tasks = len(self.tasks)
            self.tasks += [Task(i + len_tasks, self.random_position(self.rndTgtGen, obstacles = self.obstacles, contact_line = True), 
                                task_type, (20, self.max_time_steps) , self.sceneData) for i in range(n_tasks)]
                    
        
        #Tasks that each drone is doing        
        #         
        self.allocation_table = [-1 for _ in range(self.n_tasks)]
        
        self.tasks_current_quality = {i: [] for i in range(self.n_tasks)}  
        
        #Pre-Calculate the quality of each drone for each task
        self.quality_table =  self.calculate_capacities_table(self.agents_obj, self.tasks, self.sceneData)        
        
        self.previous_drones_positions = None        
                
        self.time_steps = 0        
        self.drone_distances = np.zeros(len(self.agents_obj))
        self.reached_tasks.clear()
        self.total_distance = 0
        self.drone_directions = np.zeros(self.n_agents)
                              
        self.all_agent_positions = np.vstack([drone.position for drone in self.agents_obj])
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
                   "drones" : copy.deepcopy(self.agents),
                   "tasks" : copy.deepcopy(self.tasks),
                   "quality_table" : copy.deepcopy(self.quality_table) ,
                   "events" : []
                   }

#------------------ STEP FUNTION --------------------------#    
    def step(self, actions):
                      
        action_reward = 0
        distance_reward = 0
        quality_reward = 0
        time_reward = 0

        done_events = []                

        self.agent_selection = self.agent_selector.next()
        
        while self.agents_obj[self.agent_name_mapping[self.agent_selection]].state == -1:
            self.agent_selection = self.agent_selector.next()            

        self.current_agent = self.agent_selection
        
        #print(".", end="")
        if self.action_mode == "TaskAssign":
                      
            # Incrementar o contador de tempo
            self.time_steps += 1
            self.previous_drones_positions = np.copy([drone.position for drone in self.agents_obj])                                          
           
            if len(self.event_list) != 0:
                
                event = self.event_list.pop()
                
                if event == "Reset_Allocation":
                    self.releaseAllTasks()
                    #print("Allocation_RESET")
                    done_events.append(event)
                                
            
            #print(actions,self.time_steps)

            #if not isinstance(actions,dict) and actions != None:   
            #    actions = {"agent0" : actions}
            
            
            if isinstance(actions,dict):               
                                
                for drone_name, obs_task_ids in actions.items():                                    
                        
                        drone_index = self.agent_name_mapping[drone_name]                       
                        
                        if self.agents_obj[drone_index].state == -1:
                            continue

                        if not isinstance(obs_task_ids, list):                                                                            
                                                        
                            #case of stay idle action
                            #if obs_task_ids == self.max_tasks:
                            #    action_reward += 1.0#5
                                #print("ok", end=".")
                            #    continue
                            
                            obs_task_ids = [obs_task_ids]
                        
                        for obs_task_id in obs_task_ids:
                                                                                                                                            
                            if obs_task_id < len(self.last_tasks_info):
                                obs_task_id = self.last_tasks_info[obs_task_id]
                            else:                            
                                action_reward += 0#-10                                
                                                        
                            task_id = obs_task_id

                            if task_id >= self.n_tasks:
                                continue                                                               
                                
                            if task_id != None and self.allocation_table.count(-1) != 0:                
                                                        
                                #print(task_id,self.time_steps)
                                task = self.tasks[task_id]
                                
                                if task.status == 0:
                                    #print(drone_name, task, self.reached_tasks, self.agents_obj[drone_index].tasks)                                            
                                    aloccation = self.agents_obj[drone_index].allocate(task, self.time_steps, self.max_time_steps)
                                    
                                    if aloccation > 0:                                    
                                        #print(f'{drone_name} -> {task_id} | Remianing: {self.allocation_table.count(-1)-1}')                                    
                                        self.tasks_current_quality[task_id] = self.quality_table[drone_index][task_id]                              
                                        #task_info[drone_index] = [task, 1]#, self.quality_table[drone_index][task], self.tasks[task].position]
                                        self.allocation_table[task_id] = drone_index
                                        task.status = 1
                                        
                                        action_reward += 5.0 #* self.agents_obj[drone_index].fit2Task[task.typeIdx]#
                                        quality_reward += self.agents_obj[drone_index].fit2Task[task.typeIdx]   # noqa: E501
                                        distance_reward += self.calculate_drone_expected_reward(self.agents_obj[drone_index])                                                  
                                        
                                        if not -1 in self.allocation_table:
                                            
                                            time_reward = max(self.agents_obj, key=lambda obj: obj.next_free_time).next_free_time / 200
                                        
                                        if self.agents_obj[drone_index].state != 1 and self.agents_obj[drone_index].state != -1: 
                                            self.agents_obj[drone_index].state = 1
                                
                                else:
                                    action_reward += -5
                                        #print("ASDASFD")                                             
                                #else:
                                    #print("old_task")
            #print("Drone_tasts:", self.agents_obj[0].tasks)
        
            # Verificar se os drones alcançaram seus alvos
            reached_tasks_this_step = set()    

            if self.hidden_obstacles:            
                self.detect_segments()
            
            # Calcular as novas posições dos drones 
            for i, drone in enumerate(self.agents_obj):
                                                         
                #Check if drone is operational
                if drone.state == -1: 
                    continue
                
                #print(drone.fail_event , self.time_steps)                
                if drone.fail_event == self.time_steps:
                    drone.state = -1
                    drone.tasks = []
                    drone.desallocateAll(self.time_steps)
                    self.event_list.append("Reset_Allocation")                    
                    #print("Fail:" , drone.drone_id, drone.fail_event )
                    #print(self.agents_obj[drone.drone_id].tasks)
                    continue
                
                movement = np.array([0,0])
                avoid_vector = np.array([0,0])
                
                #------------------- HAS TASK ------------------
                if len(drone.tasks) > 0:    
                                                    
                     current_task = self.tasks[drone.tasks[0]]

                     # Calcular condições para task
                     dir_task = current_task.position - drone.position
                     distance_task  = np.linalg.norm(dir_task)                                          
                     dir_task_norm = dir_task / distance_task                                          
                     #hdg_task = np.degrees(np.arctan2(dir_task_norm[1], dir_task_norm[0]))
                                       
                     #----------------IDLE-----------------
                     if drone.state == 0: 
                        if len(drone.tasks) == 0:
                                if self.allocation_table.count(-1) == 0:
                                    drone.state = 3                                                                                                   
                     
                     #----------------NAVIGATING-----------------
                     elif drone.state == 1: 
                     
                         # Definir uma distância limite para considerar que o alvo foi alcançado
                         if distance_task < drone.max_speed:                      
                            
                            drone.state = 2
                            drone.task_start = self.time_steps  

                         movement = dir_task_norm 
                         avoid_vector = drone.avoid_obstacles(drone, self.obstacles, movement, self.sceneData)                                                  
                            
                     #----------------IN TASK-------------------
                     elif drone.state == 2: 
                                                                        
                        #Just Started the Task
                        if drone.task_start == -1:                                            
                            
                            drone.task_start = self.time_steps
                            drone.position = current_task.position
                            
                        else:
                            
                            #Check if Task is Concluded
                            if (self.time_steps - drone.task_start) >= 0:#current_task.task_duration:
                            
                                task_id = drone.tasks.pop(0)    
                                
                                drone.tasks_done.append(task_id)                           
                                self.tasks[task_id].final_quality = self.quality_table[drone.drone_id][task_id] 
                        
                                #task_info[i] = [ task_id, 2 , self.quality_table[i][task_id]] #,  self.tasks[task_id].position ] 

                                # Adicionar o alvo alcançado ao conjunto de alvos alcançados
                                if task_id not in self.reached_tasks:
                                    reached_tasks_this_step.add(task_id)
                                    self.reached_tasks.add(task_id) 
                                    self.tasks[task_id].status = 2    

                                    if len(self.reached_tasks) == self.n_tasks:
                                        self.conclusion_time = self.time_steps
                                
                                #print("----------TASK_DONE:------------" , task_id,self.tasks[task_id].final_quality)                                
                                
                                if len(drone.tasks) == 0:
                                    if self.allocation_table.count(-1) == 0:
                                        drone.state = 3                                   
                                    else:
                                        drone.state = 0                                    
                                else:
                                    drone.state = 1
                                    
                        movement = drone.doTask(self.drone_directions[i], dir_task_norm, distance_task, current_task.type)
                                                                             
                #----------------RETURNING BASE (NO TASK)--------------------
                if drone.state == 3:
                                                              
                     if np.linalg.norm(drone.position - self.bases[0]) < drone.max_speed + 5:
                         drone.state = 0
                     else:
                         movement = DroneEnvUtils.norm_vector(self.bases[0]  - drone.position)
                         avoid_vector = drone.avoid_obstacles(drone, self.obstacles, movement, self.sceneData)     
                                                                                                                      
                movement = DroneEnvUtils.norm_vector(movement + avoid_vector) * drone.max_speed
                
                #print(movement + avoid_vector, movement , avoid_vector)
                
                # Atualizar a posição do drone                   
                drone.position = drone.position + movement 
                                                                  
                #Limitar Drone à area de atuação                         
                drone.position = np.clip(drone.position, 0, [self.area_width, self.area_height])  
         
            self.all_agent_positions = np.vstack([drone.position for drone in self.agents_obj])
                        
            # Armazenar a direção de cada drone            
            self.drone_directions = [ direction for direction in (self.all_agent_positions - self.previous_drones_positions)]
                                                
            # Calcular a distância percorrida pelos drones em cada etapa
            dists = np.linalg.norm(self.all_agent_positions - self.previous_drones_positions, axis=1)           
            
            self.drone_distances += dists    
            self.total_distance += np.sum(dists)
                                                                                              
            time_penaulty = - (self.n_tasks - len(self.reached_tasks))/self.n_tasks * (self.time_steps / self.max_time_steps)
            alloc_reward = - self.allocation_table.count(-1)            
                                                
            #print(self.allocation_table)
            #print(alloc_reward)
                                                  
            # rewards for all agents are placed in the rewards dictionary to be returned
            self.rewards = {agent.name :  0.0 * action_reward  +   #Rand +50
                                          2.5 * distance_reward +  #Rand -4
                                          6.0 * quality_reward +   #Rand +6
                                          0.0 * self.n_tasks * time_reward +      #Rand -9
                                          0.0 * alloc_reward  +
                                          1.0 * time_penaulty for agent in self.agents_obj} #Rand -28 
            
            #[ self._cumulative_rewards[agent] = self.rewards[agent] for agent in self.possible_agents]
                                       
            #Definir a condição de término (todos os alvos alcançados)
            done = (len(self.reached_tasks) == self.n_tasks or ((self.time_steps >= self.max_time_steps) and (self.max_time_steps > 0)))
            
            #Only for speed up traning without Dynamic conditions
            #done = done or (self.allocation_table.count(-1) == 0 and self.time_steps >= 50)
                                  
            self.terminations = { agent.name : (done or (agent.state < 0 )) for agent in self.agents_obj}
            
            #env_truncation = done#(self.time_steps >= self.max_time_steps) if self.max_time_steps > 0 else done             
            env_truncation = (self.time_steps >= self.max_time_steps) if self.max_time_steps > 0 else done             
            
            self.truncations = {agent.name: env_truncation for agent in self.agents_obj}     
            
            self.infos = {agent.name: {} for agent in self.agents_obj}            
            self.infos['selected'] = self.agent_selection
            self.infos['events'] = done_events
                                                           
            self._generate_observations()
                        
            #print("Setp1:", self.current_agent, self.agent_selector._current_agent)
            #self.current_agent = self.agent_selector.next()
            
            #print(self.current_agent, end=".")

            if done:
                metrics =  self.calculate_metrics() 
                self.infos['metrics'] = metrics
                #if not -1 in self.allocation_table:
                    #print("", end=".")
                #print(self.F_Reward )

                self.rewards = {agent.name :  self.F_Reward * 30 for agent in self.agents_obj} #Rand -28
                
                return self.observations, self.rewards, self.terminations, self.truncations, self.infos #metrics
            else:
                return self.observations, self.rewards, self.terminations, self.truncations, self.infos
                                        
    
    def observe(self, agent):
   
        # Get the observation for the given agent   
        observation = self.observations[agent]
        observation['agent_id'] = agent
        return observation
 
    def calculate_drone_expected_reward(self, agent):
        
        #total_distance = 0
        
        #if len(tasks) == 0:
        #    return np.linalg.norm(uav_position - base_position) / self.max_coord
        
        # Calculate distance between current position and first task
        #total_distance += np.linalg.norm(uav_position - self.tasks[0].position) 

        # Calculate distance between last task and base position
        if len(agent.tasks) >= 2:
            total_distance = np.linalg.norm(agent.next_free_position - self.tasks[agent.tasks[-2]].position)
        else:
            total_distance = np.linalg.norm(agent.next_free_position - agent.position)
        
        #print(total_distance)
    
        # Calculate reward (you can adjust the penalty factor as needed)
        reward = -1.0 * total_distance / self.max_coord
    
        return reward    

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
                for drone in self.agents_obj:
                    drone_to_mid_point = mid_point - np.array(drone.position)
                    distance = np.linalg.norm(drone_to_mid_point)
    
                    if distance <= detection_distance:
                        detected = True
                        break  # Não é necessário verificar outros drones para este segmento
    
                if detected:
                    obstacle.detected_segments.append((start_angle, stop_angle))
                        
    def random_position(self,  rndGen, obstacles = None, min_distance = 20, own_range = 3, contact_line = False):
          
        if contact_line:
            limit_line = self.sceneData.ContactLine
        else:
            limit_line = 0
            
        tries = 0
        while tries < 100:            
            
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
                
    def calculate_capacities_table(self, drones, tasks, SceneData):
        
        # Cria uma matriz de capacidade inicializada com zeros
        capacity_matrix = [[0 for _ in tasks] for _ in drones]
        
        # Preenche a matriz de capacidade com os valores de adequação de sensores
        for i, drone in enumerate(drones):
            for j, task in enumerate(tasks):
                # Para cada combinação de drone e alvo, encontra o valor máximo de adequação
                max_capacity = 0
                #for sensor_type in drone.sensors: 
                    #print(sensor_type, " -- ", task.task_type)
                agent_capacity = drone.fit2Task[task.typeIdx]
                    
                if agent_capacity > max_capacity:
                    max_capacity = agent_capacity
                
                # Armazena o valor máximo de adequação na matriz de capacidade
                capacity_matrix[i][j] = max_capacity
        
        return capacity_matrix                
   
    def unallocated_tasks(self):                
        allocated_tasks = set(task for drone in self.agents_obj for task in drone.tasks)
        all_tasks = set(range(self.n_tasks))
        return list(all_tasks.difference(allocated_tasks | self.reached_tasks))
    
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
                    if task.task_id not in self.reached_tasks:                        
                        self.reached_tasks.add(task.task_id) 
                        self.tasks[task.task_id].status = 2    

                        if len(self.reached_tasks) == self.n_tasks:
                            self.conclusion_time = self.time_steps

                else:
                    task.status = 0                        
                    self.allocation_table[task.task_id] = -1          
                   



                          
        self.tasks_current_quality = {i: -1 for i in range(self.n_tasks)}  
    
    def get_live_agents(self):

        return [agent for agent in self.agents_obj if agent.state != -1]     


####----------------------Metrics Calculation-----------------------------------
    def calculate_load_balancing_std(self):
        load_balancing_std = np.std(self.drone_distances)
        return load_balancing_std

    def calculate_load_balancing(self):
        min_distance = np.min(self.drone_distances)
        max_distance = np.max(self.drone_distances)
        load_balancing = max_distance - min_distance
        return load_balancing
   
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
        self.F_Reward = 0.25 * F_distance + 0.6 * F_quality + 0.15 * F_Time

        return { 
            "F_time": F_Time ,
            "F_distance": F_distance ,
            #"load_balancing": load_balancing,
            "F_load": 1 / load_balancing_std,
            "F_quality": F_quality,
            "F_Reward": self.F_Reward 
        }

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
        ax.set_title(f'Task Allocation: ({n_agents} drones, {n_tasks} tasks)')
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
            pygame.display.set_caption('Multi Drone Task Allocation')
        
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
                        
        # Desenhar drones
        for i,drone in enumerate(self.agents_obj):
                        
            pygame.draw.circle(comm_surface, (40, 40, 40), (int(drone.position[0]), int(drone.position[1])), drone.relay_area)
            
            if show_lines:
                if len(drone.tasks) > 0: 
                    # Desenhar linha entre o drone e seu alvo atual            
                    pygame.draw.line(agents_surface, (210, 210, 210), drone.position, self.tasks[drone.tasks[0]].position, 1)
            
                # Desenhar linha entre o alvo atual e o próximo (mais clara)
                if len(drone.tasks) > 1:                
                    pygame.draw.line(agents_surface, (100, 100, 100), self.tasks[drone.tasks[0]].position, self.tasks[drone.tasks[1]].position, 1)
            
            #pygame.draw.circle(self.screen, (0, 0, 255), (int(drone.position[0]), int(drone.position[1])), 7)
            #self.draw_rotated_x(self.screen, int(drone.position[0]), int(drone.position[1]), 10, self.drone_directions[i])
            if drone.state != 0:
                self.draw_rotated_triangle(agents_surface, int(drone.position[0]), int(drone.position[1]), 20, np.degrees(np.arctan2(self.drone_directions[i][1],self.drone_directions[i][0])) , drone.type, drone.state)            
            else:
                self.draw_rotated_triangle(agents_surface, int(drone.position[0]), int(drone.position[1]), 20, -90 , drone.type, drone.state)            
            
                            
               
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
                                               

        # Desenhar alvos
        for i, task in enumerate(self.tasks):
            if task.type == "Rec":
                color = (0, 255, 0) if i in self.reached_tasks else (40, 80, 40)           
                pygame.draw.circle(agents_surface, color, (int(task.position[0]), int(task.position[1])), 8)
            else:
                color = (200, 0, 0) if i in self.reached_tasks else (80, 40, 40)           
                pygame.draw.circle(agents_surface, color, (int(task.position[0]), int(task.position[1])), 10) 
            
            # Renderizar o número do alvo e desenhá-lo no centro do círculo
            task_number_text = font.render(str(i), True, (30, 30, 30))  # Renderizar o texto (preto)
            text_rect = task_number_text.get_rect(center=(int(task.position[0]), int(task.position[1])))  # Centralizar o texto no círculo
            agents_surface.blit(task_number_text, text_rect)
                                  
        
        
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

    def draw_detected_circle_segments(self, surface, color, obstacle, width=1):
        
        radius = obstacle.size
        center = obstacle.position
        
        for start_angle, stop_angle in obstacle.detected_segments:
            rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            pygame.draw.arc(surface, color, rect, start_angle, stop_angle, width)

    


#-----------------------------------------------------------------------------#