import gym
from gym import spaces
from gym.spaces import Dict, Discrete
import pygame
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from DroneEnvComponents import Drone, Target

# Tamanho da tela
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


class DroneEnv(gym.Env):
    
    def __init__(self, action_mode="TaskAssign",  render_enabled=True, max_time_steps=-1, num_drones=5, num_targets=10):
        super(DroneEnv, self).__init__()
        
        self.seed = 0
        self.max_time_steps = max_time_steps
        self.render_enabled = render_enabled         

        self.clock = pygame.time.Clock()
        self.action_mode = action_mode
        self.reached_targets = set()
        
        self.NUM_DRONES = num_drones
        self.NUM_TARGETS = num_targets  
        
        self.drones = None
        self.targets = None                
        
        self.drone_tasks = {i: [] for i in range(self.NUM_DRONES)}
                
        self.time_steps = 0
        self.total_distance = 0
        self.drone_distances = None
        self.drone_directions = None
        
        self.max_speed = 5.0
                
        self.previous_drones_positions = None
        self.previous_drones_positions = None
                        
        # Inicializar o Pygame
        self.screen = None      
        if render_enabled:
            pygame.init()
              

        if action_mode == "TaskAssign":
            # A ação agora deve ser uma tupla de alocação de tarefas para cada drone
            #self.action_space = spaces.Tuple([spaces.MultiDiscrete([len(self.targets)] * 2) for _ in range(self.NUM_DRONES)])
            self.action_space = Dict({i: Discrete(self.NUM_TARGETS + 1) for i in range(self.NUM_DRONES)})
        
        elif action_mode == "DroneControl":
            # Ação: ângulo de movimento (0 a 360 graus)
            self.action_space = spaces.Box(low=0, high=360, shape=(self.NUM_DRONES,), dtype=np.float32)
            
        else:
            print("\nUnknown Action Mode")     
            exit(-1)
            
        # Observação: coordenadas (x, y) dos drones e alvos
        self.observation_space = spaces.Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT),
                                            shape=(2 * (self.NUM_DRONES + self.NUM_TARGETS),), dtype=np.float32)

    
    def reset(self, seed=0):
        
        random.seed(seed)
        self.seed = seed
                        
        
        if self.drones != None:
           self.drones.clear() 
        if self.targets != None:
            self.targets.clear() 
        
        self.drones = [Drone(i, self.random_position()) for i in range(self.NUM_DRONES)]
        self.targets = [Target(i, self.random_position()) for i in range(self.NUM_TARGETS)]
               
        self.previous_drones_positions = None
        self.previous_drones_positions = None
                
        self.time_steps = 0        
        self.drone_distances = np.zeros(len(self.drones))
        self.reached_targets.clear()
        self.total_distance = 0
        self.drone_directions = np.zeros(self.NUM_DRONES)
        self.drone_tasks = {i: [] for i in range(self.NUM_DRONES)}
        
        # Atribuir tarefas aos drones
        #if self.tasks_selection == "Random":
        #    self.assign_tasks()
        
        # Concatenar as posições dos drones e alvos em um único array
        self.all_drone_positions = np.vstack([drone.position for drone in self.drones])
        observation = np.concatenate((self.all_drone_positions, self.all_drone_positions))
        
        return observation

    def step(self, action):
               
        if self.action_mode == "DroneControl":
            # Mover os drones com base na ação
            for i in range(self.NUM_DRONES):
        
                drone_velocity = action[i]["velocity"]
                movement = np.array(drone_velocity)
        
                self.drones[i] = self.drones[i] + movement
        
                # Limitar a posição dos drones dentro da tela
                self.drones[i] = np.clip(self.drones[i], 0, [SCREEN_WIDTH, SCREEN_HEIGHT])
        
            # Calcular a recompensa
            reward = self._calculate_reward()
        
            # Verificar se o episódio terminou
            done = self._check_done()

            return self._get_observation(), reward, done, {}

        
        if self.action_mode == "TaskAssign":
            
            # Incrementar o contador de tempo
            self.time_steps += 1

            # Armazenar as posições atuais dos drones antes de atualizá-los
            self.previous_drones_positions = np.copy([drone.position for drone in self.drones])
                                          
            # Iterar sobre as chaves e valores da ação       

            if isinstance(action,dict):               
                for drone_index, task_allocation in action.items():                                    
                        
                                        
                        # Adicione a tarefa à lista de tarefas do drone
                        #self.drone_tasks[drone_index] = self.drone_tasks[drone_index] + task_allocation
                        self.drones[drone_index].tasks = self.drones[drone_index].tasks + task_allocation                                         
        
            # Calcular as novas posições dos drones voando para o alvo
            for i, drone in enumerate(self.drones):
                            
               if len(drone.tasks) > 0:                                                        
                   current_target = self.targets[drone.tasks[0]].position                                
               else:
                   continue
                                    
               # Calcular a direção em relação ao alvo
               direction = current_target - drone.position
               distance = np.linalg.norm(direction)
                                 
               if distance > 1:
                   # Normalizar a direção e multiplicar pela ação
                   direction_normalized = direction / distance
                   movement = direction_normalized * self.max_speed

                   # Atualizar a posição do drone                   
                   self.drones[i].position = self.drones[i].position + movement
            
            
            self.all_drone_positions = np.vstack([drone.position for drone in self.drones])
            
            # Armazenar a direção de cada drone
            self.drone_directions = [np.degrees(np.arctan2(direction_normalized[1], direction_normalized[0])) for direction_normalized in (self.all_drone_positions - self.previous_drones_positions)]
            
            # Calcular a distância percorrida pelos drones em cada etapa
            dists = np.linalg.norm(self.all_drone_positions - self.previous_drones_positions, axis=1)           
            
            self.drone_distances += dists    
            self.total_distance += np.sum(dists)
    
            # Verificar se os drones alcançaram seus alvos
            reached_targets_this_step = set()
            
            for i, drone in enumerate(self.drones):
                
                if len(drone.tasks) > 0:
                    
                    current_target = self.targets[drone.tasks[0]].position                                        
                    distance = np.linalg.norm(drone.position - current_target)
    
                    if distance < 3:  # Definir uma distância limite para considerar que o alvo foi alcançado
                        
                        target_id = drone.tasks.pop(0)    
                        reached_targets_this_step.add(target_id)
                        drone.tasks_done.append(target_id)
    
                        # Adicionar o alvo alcançado ao conjunto de alvos alcançados
                        self.reached_targets.add(target_id)
                                                
            # Calcular a recompensa com base nos alvos alcançados
            reward = len(reached_targets_this_step)
                        
            # Definir a condição de término (todos os alvos alcançados)
            done = len(self.reached_targets) == self.NUM_TARGETS or (self.time_steps >= self.max_time_steps and self.max_time_steps > 0)            
            
            # Atualizar a observação :::: (TODO: errado )
            observation = np.concatenate((self.all_drone_positions, self.all_drone_positions))
    
            if done:
                metrics =  self.calculate_metrics()
                return observation, reward, done, metrics
            else:
                return observation, reward, done, {}
                                        
    
    def _calculate_reward(self):
        # Calcular a distância entre cada drone e seu alvo mais próximo
        min_distances = [min(np.linalg.norm(drone.position - target.position) for target in self.targets) for drone in self.drones]
        return -np.sum(min_distances)

    def _check_done(self):
        # Verificar se todos os drones estão próximos de um alvo
        #done = all(any(np.linalg.norm(drone - target) < 10 for target in self.targets) for drone in self.drones)
        return done
    
    def random_position(self):        
        x = random.uniform(0, SCREEN_WIDTH)
        y = random.uniform(0, SCREEN_HEIGHT)       
        return np.array([x, y])

    def generate_specific_task_actions(self, specific_tasks):
        task_actions = {}
        for drone_index, tasks in specific_tasks.items():
            task_actions[drone_index] = tasks

        return task_actions    

    def generate_random_tasks_all(self):
        
        task_list = list(range(self.NUM_TARGETS))
        random.shuffle(task_list)

        # Calcular o número mínimo de tarefas por drone e o número de drones que receberão uma tarefa extra
        min_tasks_per_drone, extra_tasks = divmod(self.NUM_TARGETS, self.NUM_DRONES)

        # Gerar ações de tarefas para os drones
        task_actions = {}
        task_idx = 0
        for i in range(self.NUM_DRONES):
            num_tasks = min_tasks_per_drone + (1 if i < extra_tasks else 0)
            task_actions[i] = task_list[task_idx:task_idx + num_tasks]
            task_idx += num_tasks

        return task_actions

    def unallocated_tasks(self):                
        allocated_tasks = set(task for drone in self.drones for task in drone.tasks)
        all_tasks = set(range(self.NUM_TARGETS))
        return list(all_tasks.difference(allocated_tasks | self.reached_targets))
       
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

        return {
            "total_time": total_time / 383,
            "total_distance": total_distance / 7320,
            #"load_balancing": load_balancing,
            "load_balancing_std": load_balancing_std / 314
        }
    
    
    def plot_metrics(self, df, num_drones, num_tasks, algorithm):
        # Calcular médias e desvios padrão
        df.total_time =  df.total_time 
        df.total_distance =  df.total_distance 
        #df.load_balancing = df.load_balancing
        df.load_balancing_std =  df.load_balancing_std 
        
        
        means = df.mean()
        std_devs = df.std()        
        
        # Criar gráfico de barras com barras de erro
        fig, ax = plt.subplots()
        index = np.arange(len(means))
        bar_width = 0.7
        opacity = 0.8
    
        plt.bar(index, means, bar_width, alpha=opacity, color='b', label='Média', yerr=std_devs, capsize=5)
    
        plt.xlabel('Métricas')
        plt.ylabel('Valores')
        plt.title(f'Task Allocation {algorithm} :  ({num_drones} drones, {num_tasks} tarefas)')
        plt.xticks(index, df.columns)
        plt.legend()
        ax.set_ylim(0, 1.5)
    
        plt.tight_layout()
        plt.show()

    def plot_convergence(self,df, num_drones, num_tasks,algorithm):
        cumulative_means = df.expanding().mean()
    
        fig, ax = plt.subplots()
        for metric in cumulative_means.columns:
            ax.plot(cumulative_means[metric], label=metric)
    
        ax.set_xlabel('Número de simulações')
        ax.set_ylabel('Média acumulada das métricas')
        ax.set_title(f'Convergência das Métricas {algorithm} : ({num_drones} drones, {num_tasks} tarefas)')
        ax.legend()
        ax.set_ylim(0, 1.5)
        
    
        plt.tight_layout()
        plt.show()

    def draw_rotated_x(self,surface, x, y, size, angle):
        half_size = size / 2
        angle_rad = np.radians(angle)
    
        dx1 = half_size * np.cos(angle_rad - np.pi / 4)
        dy1 = half_size * np.sin(angle_rad - np.pi / 4)
    
        dx2 = half_size * np.cos(angle_rad + np.pi / 4)
        dy2 = half_size * np.sin(angle_rad + np.pi / 4)
    
        pygame.draw.line(surface, (0, 0, 255), (x - dx1, y - dy1), (x + dx1, y + dy1), 2)
        pygame.draw.line(surface, (0, 0, 255), (x - dx2, y - dy2), (x + dx2, y + dy2), 2)


    
    def render(self):
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Drone Task Allocation')
        
        # Desenhar fundo
        self.screen.fill((255, 255, 255))
        
        # Desenhar drones
        for i,drone in enumerate(self.drones):
            #pygame.draw.circle(self.screen, (0, 0, 255), (int(drone[0]), int(drone[1])), 5)
            self.draw_rotated_x(self.screen, int(drone.position[0]), int(drone.position[1]), 10, self.drone_directions[i])
        
        font = pygame.font.Font(None, 18)

        # Desenhar alvos
        for i, target in enumerate(self.targets):
            color = (0, 255, 0) if i in self.reached_targets else (255, 0, 0)           
            pygame.draw.circle(self.screen, color, (int(target.position[0]), int(target.position[1])), 8)
            
            # Renderizar o número do alvo e desenhá-lo no centro do círculo
            target_number_text = font.render(str(i), True, (30, 30, 30))  # Renderizar o texto (preto)
            text_rect = target_number_text.get_rect(center=(int(target.position[0]), int(target.position[1])))  # Centralizar o texto no círculo
            self.screen.blit(target_number_text, text_rect)
            
        
        pygame.display.flip()
        
        # Limitar a taxa de quadros
        self.clock.tick(120)
        
        # Verificar se a janela está fechada
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True
   

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    
    env = DroneEnv(action_mode= "TaskAssign", num_drones=5, num_targets = 20, render_enabled=False)
    env.reset(seed=0)
    
    totalMetrics = []
    # Testar o ambiente com ações calculadas
    for episode in range(100):
        
        observation = env.reset( seed = episode)
        done = False

        actions = env.generate_random_tasks_all() 
        #actions = {0: [2, 3, 4, 5, 10, 13, 15], 1: [9], 2: [1, 6, 8, 14, 19, 17, 18, 11, 16], 3: [], 4: [0, 7, 12]}
        
        env.step(actions)
        
        print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")       
        while not done:
            
            
            # Definir uma ação para mover os drones em direção aos alvos
            action = None#np.full(env.NUM_DRONES, 5)

            observation, reward, done, info = env.step(action)
            
            if done:
                totalMetrics.append(info)
                
                #print("Time steps for the current round:", info["time_steps"])
                #print("Total distance traveled by drones in the current round:", info["total_distance"])

    env.close()
    
    metricsDf = pd.DataFrame(totalMetrics)
    
    # Chamar a função de plotagem
    env.plot_metrics(metricsDf, len(env.drones), len(env.targets), "Random")
    env.plot_convergence(metricsDf, len(env.drones), len(env.targets),"Random")

