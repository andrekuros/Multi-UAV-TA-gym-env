import gym
from gym import spaces
from gym.spaces import Dict, Discrete, MultiDiscrete
import pygame
import numpy as np
import random
import copy
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from pygame_screen_recorder import pygame_screen_recorder as pgr

import functools
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.utils import agent_selector

from DroneEnvComponents import Drone, Target, Obstacle
import Utils as ut


# Tamanho da tela
SCREEN_WIDTH = 850
SCREEN_HEIGHT = 750
N_TARGETS_TYPES = 4
MAX_INT = sys.maxsize

def env(render_mode=None, action_mode="TaskAssign",  render_speed=-1, max_time_steps=-1, num_drones=5, num_targets=10, num_obstacles=5):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    #env = raw_env(internal_render_mode, action_mode,  render_speed, max_time_steps, num_drones, num_targets)
    env = MultiDroneEnv(render_mode=render_mode, action_mode=action_mode,  render_speed=render_speed, max_time_steps=max_time_steps, num_drones=num_drones, num_targets=num_targets, num_obstacles=num_obstacles)
    # This wrapper is only for environments which print results to the terminal
    #if render_mode == "ansi":
    #    env = wrappers.CaptureStdoutWrapper(env)
    
    # this wrapper helps error handling for discrete action spaces
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    #env = wrappers.OrderEnforcingWrapper(env)
    
    return env


def raw_env(render_mode=None, action_mode="TaskAssign",  render_speed=-1, max_time_steps=-1, num_drones=5, num_targets=10, num_obstacles=5):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = MultiDroneEnv(render_mode, action_mode,  render_speed, max_time_steps, num_drones, num_targets)
    env = parallel_to_aec(env)
    return env

class MultiDroneEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multi_drone_env_v0"}
    
    def __init__(self, 
                 render_mode = None, 
                 action_mode="TaskAssign",  
                 render_speed=-1, 
                 max_time_steps=-1, 
                 num_drones=5, 
                 num_targets=10,
                 num_obstacles=5):
        
        super(MultiDroneEnv, self).__init__()
        
        self.seed = 0
        self.rndGen = random.Random(self.seed)
        self.max_time_steps = max_time_steps
        self.render_speed = render_speed        
        self.render_enabled = render_speed != -1 #Render is activated if speed != -1
        

        self.clock = pygame.time.Clock()
        self.action_mode = action_mode
        self.reached_targets = set()
        
        # Define Agents
        self.NUM_DRONES = num_drones        
        self.drones = None
        self.possible_agents = [i for i in range(num_drones)]
        
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
                
        self.NUM_TARGETS = num_targets  
        self.targets = None                 
                
        self.NUM_OBSTACLES = num_obstacles
        
        self.obstacles = None
                        
        
        self.tasks_current_quality = None
        self.drone_tasks = None

        self.sensors_table = np.array([[1.0, 0.0, 0.3, 0.5],
                                       [0.0, 0.0, 1.0, 0.0],
                                       [0.2, 0.0, 0.0, 1.0],
                                       [0.0, 1.0, 0.0, 0.3]])
                                       
        #Table with the quality of each drone for each target
        self.quality_table = None
                
        self.time_steps = 0
        self.total_distance = 0
        self.drone_distances = None
        self.drone_directions = None
        
        self.max_speed = 5.0
                
        self.previous_drones_positions = None
        self.previous_drones_positions = None
                        
        # Inicializar o Pygame
        self.screen = None      
        self.recrdr = None
        
        if self.render_enabled:
            pygame.init()
            # Configurar o MovieWriter
            #self.recrdr = pgr("Test_Tessi.gif") # init recorder object
              

        #if action_mode == "TaskAssign":
            # A ação agora deve ser uma tupla de alocação de tarefas para cada drone
            #self.action_space = spaces.Tuple([spaces.MultiDiscrete([len(self.targets)] * 2) for _ in range(self.NUM_DRONES)])
        #    self.action_space = Discrete(self.NUM_DRONES)
        
        #elif action_mode == "DroneControl":
            # Ação: ângulo de movimento (0 a 360 graus)
        #    self.action_space = spaces.Box(low=0, high=360, shape=(self.NUM_DRONES,), dtype=np.float32)
            
        #else:
        #    print("\nUnknown Action Mode")     
        #    exit(-1)
            
        # Observação: coordenadas (x, y) dos drones e alvos
        #self.observation_space = spaces.Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT),
        #                                    shape=(2 * (self.NUM_DRONES + self.NUM_TARGETS),), dtype=np.float32)
  
    
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):        
        return Dict({"agent_position" : spaces.Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT), shape=(2,), dtype=np.float32),
                    "target_status": MultiDiscrete([self.NUM_TARGETS, 4])})#, spaces.Box(low=0,high=1.0, shape=()), spaces.Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT), shape=(2,))])})
                     

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.NUM_TARGETS)
    
    
    def reset(self, seed=0, return_info = True, options={"options":1}):
        
        self.rndAgentGen = random.Random(seed)
        self.rndObsGen = random.Random(self.rndAgentGen.randint(0, MAX_INT))
        self.rndTgtGen = random.Random(self.rndAgentGen.randint(0, MAX_INT))
        
        self.seed = seed
                                
        if self.drones != None:
           self.drones.clear() 
        if self.targets != None:
            self.targets.clear() 
        if self.obstacles != None:
            self.obstacles.clear()
        
        self.agents = self.possible_agents[:]                        
                        
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            size = self.rndGen.randint(30, 100)
            position = self.random_position(self.rndObsGen, obstacles = self.obstacles, own_range = size )
            
            self.obstacles.append(Obstacle(position, size))
        
        
        listSensors = [self.random_list(1,2,3) for i in range(self.NUM_DRONES)]
        
        if False:
            types = [0,1,2,3]
            anyChange = True        
            while anyChange:            
                
                anyChange = False
                
                for sType in types:
                    if not any(sType in sublist for sublist in listSensors):
                        #print("Não criou", sType)
                        listSensors[self.rndAgentGen.randint(0, len(listSensors)-1)][0] = sType
                        anyChange = True
                    
                
        #self.drones = [Drone(i, self.random_position(self.rndAgentGen, obstacles = self.obstacles), sensors=listSensors[i]) for i in range(self.NUM_DRONES)]
        self.drones = [Drone(i, np.array([0,0]), sensors=listSensors[i]) for i in range(self.NUM_DRONES)]
        
        self.targets = [Target(i, self.random_position(self.rndTgtGen, obstacles = self.obstacles), target_type=self.rndGen.randint(0, N_TARGETS_TYPES-1)) for i in range(self.NUM_TARGETS)]
                
        
        #Tasks that each drone is doing
        self.drone_tasks = {i: [] for i in range(self.NUM_DRONES)}  
        
        self.tasks_current_quality = {i: [] for i in range(self.NUM_TARGETS)}  
        
        #Pre-Calculate the quality of each drone for each target
        self.quality_table =  self.calculate_capacities_table(self.drones, self.targets,self.sensors_table)        
        
        self.previous_drones_positions = None        
                
        self.time_steps = 0        
        self.drone_distances = np.zeros(len(self.drones))
        self.reached_targets.clear()
        self.total_distance = 0
        self.drone_directions = np.zeros(self.NUM_DRONES)
                              
        self.all_drone_positions = np.vstack([drone.position for drone in self.drones])
        self.target_position = np.vstack([target.position for target in self.targets])
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        #if self.action_mode == "DroneControl":
        #    observation = np.concatenate((self.target_position, self.all_drone_positions))
        
        #if self.action_mode == "TaskAssign":
        #    observation = {agent: None for agent in self.agents}
        
        observations = {self.agents[i]: { "agent_position" : self.all_drone_positions[i],
                        "target_status"  : [-1,-1,-1]} for i in range(len(self.agents))}
        
                      
        return observations
    
    def get_initial_state(self):
       # return a dictionary with the initial state information
       return  { 
                   "drones" : copy.deepcopy(self.drones) ,
                   "targets" : copy.deepcopy(self.targets),
                   "quality_table" : copy.deepcopy(self.quality_table) }

    def step(self, actions):
                      
        if self.action_mode == "TaskAssign":
                      
            # Incrementar o contador de tempo
            self.time_steps += 1

            # Armazenar as posições atuais dos drones antes de atualizá-los
            self.previous_drones_positions = np.copy([drone.position for drone in self.drones])
                                          
            # Iterar sobre as chaves e valores da ação       
            target_info = [False for i in range(len(self.agents))]
                        
            if isinstance(actions,dict):               
                for drone_index, task in actions.items():                                    
                        
                        if task != None:                
                                                    
                        # Adicione a tarefa à lista de tarefas do drone
                            #self.drone_tasks[drone_index] = self.drone_tasks[drone_index] + task_allocation
                            self.drones[drone_index].tasks.append(task)                                                                                    
                            self.tasks_current_quality[task] = self.quality_table[drone_index][task]                              
                            target_info[drone_index] = [task, 1]#, self.quality_table[drone_index][task], self.targets[task].position]
                                      
        
            # Verificar se os drones alcançaram seus alvos
            reached_targets_this_step = set()    

            self.detect_segments()
            
            # Calcular as novas posições dos drones voando para o alvo
            for i, drone in enumerate(self.drones):
                            
               if len(drone.tasks) > 0:    
                                                    
                     current_target = self.targets[drone.tasks[0]].position  

                     # Calcular a direção em relação ao alvo
                     direction = current_target - drone.position
                     distance = np.linalg.norm(direction)
                                       
                     # Definir uma distância limite para considerar que o alvo foi alcançado
                     if distance < 3:  
                    
                        target_id = drone.tasks.pop(0)    
                        reached_targets_this_step.add(target_id)
                        drone.tasks_done.append(target_id)
                       
                        self.targets[target_id].final_quality = self.quality_table[i][target_id] 
                        
                        target_info[i] = [ target_id, 2 , self.quality_table[i][target_id]] #,  self.targets[target_id].position ] 
                        
                        # Adicionar o alvo alcançado ao conjunto de alvos alcançados
                        self.reached_targets.add(target_id)          
                    
                     else:
                         
                         # Normalizar a direção e multiplicar pela ação
                         direction_normalized = direction / distance
                         
                         
                    
                         avoid_vector = drone.avoid_obstacles(drone, self.obstacles, current_target)     
                         
                         movement = ut.norm_vector(direction_normalized + avoid_vector) * self.max_speed
                         
                         #print(movement + avoid_vector, movement , avoid_vector)
                         # Atualizar a posição do drone                   
                         drone.position = drone.position + movement 
                         
                         
                         
                         #Limitar Drone à area de atuação                         
                         drone.position = np.clip(drone.position, 0, [SCREEN_WIDTH, SCREEN_HEIGHT])                                   
                            
            
            self.all_drone_positions = np.vstack([drone.position for drone in self.drones])
            
            # Armazenar a direção de cada drone
            self.drone_directions = [np.degrees(np.arctan2(direction_normalized[1], direction_normalized[0])) for direction_normalized in (self.all_drone_positions - self.previous_drones_positions)]
            
            # Calcular a distância percorrida pelos drones em cada etapa
            dists = np.linalg.norm(self.all_drone_positions - self.previous_drones_positions, axis=1)           
            
            self.drone_distances += dists    
            self.total_distance += np.sum(dists)
                                                                                              
            # rewards for all agents are placed in the rewards dictionary to be returned
            rewards = {k: len(reached_targets_this_step) for k in self.agents}  
            
            # Definir a condição de término (todos os alvos alcançados)
            done = len(self.reached_targets) == self.NUM_TARGETS or (self.time_steps >= self.max_time_steps and self.max_time_steps > 0)            
            
            terminations = {agent: done for agent in self.agents}
            
            env_truncation = (self.time_steps >= self.max_time_steps) if self.max_time_steps > 0 else False             
            
            truncations = {agent: env_truncation for agent in self.agents}            
                                               
            # current observation is just the other player's most recent action
            observations = {
                self.agents[i]: { "agent_position" : self.all_drone_positions[i],
                                  "target_status"  : target_info[i]}
                                  for i in range(len(self.agents))
                }            
            
            if done:
                metrics =  self.calculate_metrics()
                
                return observations, rewards, terminations, truncations, metrics
            else:
                return observations, rewards, terminations, truncations, {}
                                        
    
    
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
                for drone in self.drones:
                    drone_to_mid_point = mid_point - np.array(drone.position)
                    distance = np.linalg.norm(drone_to_mid_point)
    
                    if distance <= detection_distance:
                        detected = True
                        break  # Não é necessário verificar outros drones para este segmento
    
                if detected:
                    obstacle.detected_segments.append((start_angle, stop_angle))

    
    def _calculate_reward(self):
        # Calcular a distância entre cada drone e seu alvo mais próximo
        min_distances = [min(np.linalg.norm(drone.position - target.position) for target in self.targets) for drone in self.drones]
        return -np.sum(min_distances)

    def _get_observation(self,mode="DroneControl"):
        
        return np.concatenate((self.all_drone_positions, self.all_drone_positions))
        
        
    def _check_done(self):
        # Verificar se todos os drones estão próximos de um alvo
        done = all(any(np.linalg.norm(drone - target) < 10 for target in self.targets) for drone in self.drones)
        return done
    
    def random_position(self,  rndGen, obstacles = None, min_distance = 20, own_range = 3):
                       
        tries = 0
        while tries < 100:            
            
            x = self.rndGen.uniform(own_range + min_distance, SCREEN_WIDTH - own_range - min_distance)
            y = self.rndGen.uniform(own_range + min_distance, SCREEN_HEIGHT - own_range - min_distance)
            
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
        
        
        
      
    def random_list(self,n_min, n_max, t_max):
        # Determina a quantidade de elementos aleatórios a serem gerados
        length = self.rndGen.randint(n_min, n_max)
        
        # Gera a lista de elementos aleatórios
        random_list = [self.rndGen.randint(0, t_max) for _ in range(length)]
        
        return random_list
      
    def calculate_capacities_table(self, drones, targets, sensor_matrix):
        
        # Cria uma matriz de capacidade inicializada com zeros
        capacity_matrix = [[0 for _ in targets] for _ in drones]
        
        # Preenche a matriz de capacidade com os valores de adequação de sensores
        for i, drone in enumerate(drones):
            for j, target in enumerate(targets):
                # Para cada combinação de drone e alvo, encontra o valor máximo de adequação
                max_capacity = 0
                for sensor_type in drone.sensors: 
                    #print(sensor_type, " -- ", target.target_type)
                    sensor_capacity = self.sensors_table[sensor_type][target.target_type]
                    
                    if sensor_capacity > max_capacity:
                        max_capacity = sensor_capacity
                
                # Armazena o valor máximo de adequação na matriz de capacidade
                capacity_matrix[i][j] = max_capacity
        
        return capacity_matrix                
   
    def unallocated_tasks(self):                
        allocated_tasks = set(task for drone in self.drones for task in drone.tasks)
        all_tasks = set(range(self.NUM_TARGETS))
        return list(all_tasks.difference(allocated_tasks | self.reached_targets))


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
        
        total_time = self.time_steps
        total_distance = self.total_distance
        #load_balancing = self.calculate_load_balancing()
        load_balancing_std = self.calculate_load_balancing_std()
        
        total_quality = np.mean([0 if t.final_quality == -1 else t.final_quality  for t in self.targets])        

        #for 15 drones / 50 ,targets
        #total_distance = 18245.25,  total_time = 397.5 , load_balancing_std = 371.21        
        return {
            "total_time": total_time / 397.5 ,
            "total_distance": total_distance /  18245.25 ,
            #"load_balancing": load_balancing,
            "load_balancing_std": load_balancing_std / 371.21 ,
            "total_quality": total_quality
        }

    def plot_metrics(self, df, num_drones, num_tasks):
        # Group data by algorithm and calculate means and standard deviations
        grouped = df.groupby('Algorithm')
        means = grouped.mean()
        std_devs = grouped.std()
        
        
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
        ax.set_title(f'Task Allocation: ({num_drones} drones, {num_tasks} tasks)')
        ax.set_xticks(np.arange(num_metrics) * group_spacing + (bar_width * (num_algorithms - 1) / 2))
        ax.set_xticklabels(list(df.columns)[:-1])
        ax.legend()
        ax.set_ylim(0, 2.0)
    
        plt.tight_layout()
        plt.show()
    
    def plot_convergence(self,df, num_drones, num_tasks,algorithm):
        
        cumulative_means = df.expanding().mean()
    
        palette = sns.color_palette("Set1",n_colors=len(df)-1)
              
        fig, ax = plt.subplots()
        auxDf = cumulative_means.reset_index()
        
        for i, metric in enumerate(auxDf.columns[1:]):
            ax.plot(auxDf[metric], label=metric, color = palette[i])
    
        ax.set_xlabel('Número de simulações')
        ax.set_ylabel('Média acumulada das métricas')
        ax.set_title(f'Convergência das Métricas {algorithm} : ({num_drones} drones, {num_tasks} tarefas)')
        ax.legend()
        ax.set_ylim(0, 1.5)
            
        plt.tight_layout()
        plt.show()

#####--------------- Rendering in PyGame -------------------###################

    def draw_rotated_triangle(self, surface, x, y, size, angle):
        mod_size = size / 2
        
        angle_rad = np.radians(angle)
    
        dx1 = mod_size  * np.cos(angle_rad - np.pi / 0.7)
        dy1 = mod_size  * np.sin(angle_rad - np.pi / 0.7)
    
        dx2 = mod_size * np.cos(angle_rad)
        dy2 = mod_size * np.sin(angle_rad)
        
        dx3 = mod_size  * np.cos(angle_rad + np.pi / 0.7)
        dy3 = mod_size  * np.sin(angle_rad + np.pi / 0.7)
           
        points = [(x + dx1, y + dy1), (x + dx2, y + dy2), (x + dx3, y + dy3)]
        pygame.draw.polygon(surface, (0, 0, 255), points )


        
    
    def draw_rotated_x(self,surface, x, y, size, angle):
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
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Drone Task Allocation')
        
        # Desenhar fundo
        self.screen.fill((0, 0, 0))
        
        # Desenhar drones
        for i,drone in enumerate(self.drones):
            
            if show_lines:
                if len(drone.tasks) > 0: 
                    # Desenhar linha entre o drone e seu alvo atual            
                    pygame.draw.line(self.screen, (210, 210, 210), drone.position, self.targets[drone.tasks[0]].position, 1)
            
                # Desenhar linha entre o alvo atual e o próximo (mais clara)
                if len(drone.tasks) > 1:                
                    pygame.draw.line(self.screen, (100, 100, 100), self.targets[drone.tasks[0]].position, self.targets[drone.tasks[1]].position, 1)
            
            #pygame.draw.circle(self.screen, (0, 0, 255), (int(drone.position[0]), int(drone.position[1])), 7)
            #self.draw_rotated_x(self.screen, int(drone.position[0]), int(drone.position[1]), 10, self.drone_directions[i])
            self.draw_rotated_triangle(self.screen, int(drone.position[0]), int(drone.position[1]), 20, self.drone_directions[i])
            
            
        
        font = pygame.font.Font(None, 18)
        
        if True:
            # Desenhar obstáculos
            for obstacle in self.obstacles:
                obstacle_color = (100, 30, 30)  # Cor vermelha clara
                pygame.draw.circle(self.screen, obstacle_color, (int(obstacle.position[0]), int(obstacle.position[1])), obstacle.size)
                
                self.draw_detected_circle_segments(self.screen, obstacle_color,obstacle)
            
                # Renderizar o texto "No Fly Zone" e desenhá-lo no centro do círculo
                obstacle_text = font.render("NFZ", True, (0, 0, 0))  # Renderizar o texto (preto)
                text_rect = obstacle_text.get_rect(center=(int(obstacle.position[0]), int(obstacle.position[1])))  # Centralizar o texto no círculo
                self.screen.blit(obstacle_text, text_rect)        
            
        #else:
            

            

            # # Desenhar obstáculos
            # for obstacle in self.obstacles:
            #     obstacle_color = (255, 150, 150)  # Cor vermelha clara
                
            #     # Dividir a circunferência do obstáculo em 12 partes
            #     num_segments = 12
            #     angle_step = 2 * np.pi / num_segments
        
            #     for i in range(num_segments):
            #         angle1 = i * angle_step
            #         angle2 = (i + 1) * angle_step
                    
            #         # Calcular as posições dos pontos na circunferência
            #         x1 = int(obstacle.position[0] + obstacle.size * np.cos(angle1))
            #         y1 = int(obstacle.position[1] + obstacle.size * np.sin(angle1))
            #         x2 = int(obstacle.position[0] + obstacle.size * np.cos(angle2))
            #         y2 = int(obstacle.position[1] + obstacle.size * np.sin(angle2))
                    
            #         # Verificar se algum drone está a 30 metros ou menos desta parte do obstáculo
            #         draw_segment = False
            #         for drone in self.drones:
            #             distance_to_segment = min(np.linalg.norm(drone.position - np.array([x1, y1])),
            #                                       np.linalg.norm(drone.position - np.array([x2, y2])))
            #             if distance_to_segment <= 30:
            #                 draw_segment = True
            #                 break
        
            #         # Desenhar o segmento da circunferência se estiver dentro da faixa de visão de algum drone
            #         if draw_segment:
            #             pygame.draw.line(self.screen, obstacle_color, (x1, y1), (x2, y2), 2) 
               

        # Desenhar alvos
        for i, target in enumerate(self.targets):
            color = (0, 255, 0) if i in self.reached_targets else (255, 0, 0)           
            pygame.draw.circle(self.screen, color, (int(target.position[0]), int(target.position[1])), 8)
            
            # Renderizar o número do alvo e desenhá-lo no centro do círculo
            target_number_text = font.render(str(i), True, (30, 30, 30))  # Renderizar o texto (preto)
            text_rect = target_number_text.get_rect(center=(int(target.position[0]), int(target.position[1])))  # Centralizar o texto no círculo
            self.screen.blit(target_number_text, text_rect)
                          

        
        pygame.display.flip()
        
        # Salvar a imagem atual do jogo
        if self.recrdr != None:
            self.recrdr.click(self.screen) # save frame as png to _temp_/ folder
        
        # Limitar a taxa de quadros
        self.clock.tick(self.render_speed * 60)
        
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