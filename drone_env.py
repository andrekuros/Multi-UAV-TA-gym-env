import gym
from gym import spaces
import pygame
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Tamanho da tela
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Configurações dos drones e alvos
NUM_DRONES = 10
NUM_TARGETS = 20

class DroneEnv(gym.Env):
    
    def __init__(self, render_enabled=True):
        super(DroneEnv, self).__init__()

        self.render_enabled = render_enabled        

        self.clock = pygame.time.Clock()
        self.mode = "TaskAssign"
        self.reached_targets = set()
        
        self.drones = None
        self.targets = None
        self.drone_tasks = self.assign_tasks()

        self.time_steps = 0
        self.total_distance = 0
        self.drone_distances = None
        self.drone_directions = None
        
        self.previous_drones_positions = np.copy(self.drones)
        
        
        
        # Inicializar o Pygame
        pygame.init()

        self.screen = None        

        # Ação: ângulo de movimento (0 a 360 graus)
        self.action_space = spaces.Box(low=0, high=360, shape=(NUM_DRONES,), dtype=np.float32)

        # Observação: coordenadas (x, y) dos drones e alvos
        self.observation_space = spaces.Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT),
                                            shape=(2 * (NUM_DRONES + NUM_TARGETS),), dtype=np.float32)

    def reset(self):

        
        self.time_steps = 0
        self.drones = np.array([self.random_position() for _ in range(NUM_DRONES)])
        self.targets = np.array([self.random_position() for _ in range(NUM_TARGETS)])
        self.drone_distances = np.zeros(len(self.drones))
        self.reached_targets.clear()
        self.total_distance = 0
        self.drone_directions = np.zeros(NUM_DRONES)
        
        # Atribuir tarefas aos drones
        self.assign_tasks()
        

        return np.concatenate((self.drones.flatten(), self.targets.flatten()))


    def step(self, action):
        
        
        if self.mode == "RL":
            # Mover os drones com base na ação
            for i in range(NUM_DRONES):
                angle = np.radians(action[i])
                self.drones[i] = self.drones[i] + np.array([np.cos(angle), np.sin(angle)])
    
                # Limitar a posição dos drones dentro da tela
                self.drones[i] = np.clip(self.drones[i], 0, [SCREEN_WIDTH, SCREEN_HEIGHT])
    
            # Calcular a recompensa
            reward = self._calculate_reward()
    
            # Verificar se o episódio terminou
            done = self._check_done()
    
            return self._get_observation(), reward, done, {}
        
        if self.mode == "TaskAssign":
            
            # Incrementar o contador de tempo
            self.time_steps += 1

            # Armazenar as posições atuais dos drones antes de atualizá-los
            self.previous_drones_positions = np.copy(self.drones)
        
            # Calcular as novas posições dos drones
            for i, drone in enumerate(self.drones):
                if i in self.drone_tasks:
                    current_target = self.get_current_target(i)
    
                    # Calcular a direção em relação ao alvo
                    direction = current_target - drone
                    distance = np.linalg.norm(direction)
    
                    if distance > 1:
                        # Normalizar a direção e multiplicar pela ação
                        direction_normalized = direction / distance
                        movement = direction_normalized * action[i]
    
                        # Atualizar a posição do drone
                        self.drones[i] += movement
                
            
            # Armazenar a direção de cada drone
            self.drone_directions = [np.degrees(np.arctan2(direction_normalized[1], direction_normalized[0])) for direction_normalized in (self.drones - self.previous_drones_positions)]
            
            # Calcular a distância percorrida pelos drones em cada etapa
            distance_per_step = np.sum(np.linalg.norm(self.drones - self.previous_drones_positions, axis=1))
            self.drone_distances += np.linalg.norm(self.drones - self.previous_drones_positions, axis=1)

            self.total_distance += distance_per_step
    
            # Verificar se os drones alcançaram seus alvos
            reached_targets = set()
            for i, drone in enumerate(self.drones):
                if i in self.drone_tasks:
                    current_target = self.get_current_target(i)
                    distance = np.linalg.norm(drone - current_target)
    
                    if distance < 10:  # Definir uma distância limite para considerar que o alvo foi alcançado
                        target_id = self.drone_tasks[i].pop(0)    
                        reached_targets.add(target_id)
    
                        # Adicionar o alvo alcançado ao conjunto de alvos alcançados
                        self.reached_targets.add(target_id)
                    
                        # Verificar se a lista de tarefas do drone está vazia
                        if not self.drone_tasks[i]:
                            del self.drone_tasks[i]
    
            # Calcular a recompensa com base nos alvos alcançados
            reward = len(reached_targets)
    
            # Definir a condição de término (todos os alvos alcançados)
            done = not self.drone_tasks
    
            
            # Atualizar a observação
            observation = np.concatenate((self.drones.flatten(), self.targets.flatten()))
    
            if done:
                metrics =  self.calculate_metrics()
                return observation, reward, done, metrics
            else:
                return observation, reward, done, {}
            
        
                    

    def _get_observation(self):
        return np.concatenate([drone.ravel() for drone in self.drones] + [target.ravel() for target in self.targets])

    def _calculate_reward(self):
        # Calcular a distância entre cada drone e seu alvo mais próximo
        min_distances = [min(np.linalg.norm(drone - target) for target in self.targets) for drone in self.drones]
        return -np.sum(min_distances)

    def _check_done(self):
        # Verificar se todos os drones estão próximos de um alvo
        done = all(any(np.linalg.norm(drone - target) < 10 for target in self.targets) for drone in self.drones)
        return done
    
    def random_position(self):
        x = random.uniform(0, SCREEN_WIDTH)
        y = random.uniform(0, SCREEN_HEIGHT)
        return np.array([x, y])

    
    def assign_tasks(self):
        task_list = list(range(NUM_TARGETS))
        random.shuffle(task_list)

        num_tasks_per_drone = len(task_list) // NUM_DRONES
        self.drone_tasks = {i: task_list[i*num_tasks_per_drone:(i+1)*num_tasks_per_drone] for i in range(NUM_DRONES)}

    def get_current_target(self, drone_id):
        target_id = self.drone_tasks[drone_id][0]
        return self.targets[target_id]

    
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
            "total_time": total_time,
            "total_distance": total_distance,
            #"load_balancing": load_balancing,
            "load_balancing_std": load_balancing_std
        }
    
    
    def plot_metrics(self, df, num_drones, num_tasks):
        # Calcular médias e desvios padrão
        df.total_time = df.total_time /400
        df.total_distance = df.total_distance /10000
        #df.load_balancing = df.load_balancing /1000
        df.load_balancing_std = df.load_balancing_std / 500
        
        
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
        plt.title(f'Task Allocation ({num_drones} drones, {num_tasks} tarefas)')
        plt.xticks(index, df.columns)
        plt.legend()
    
        plt.tight_layout()
        plt.show()

    def plot_convergence(self,df, num_drones, num_tasks):
        cumulative_means = df.expanding().mean()
    
        fig, ax = plt.subplots()
        for metric in cumulative_means.columns:
            ax.plot(cumulative_means[metric], label=metric)
    
        ax.set_xlabel('Número de simulações')
        ax.set_ylabel('Média acumulada das métricas')
        ax.set_title(f'Convergência das Métricas ({num_drones} drones, {num_tasks} tarefas)')
        ax.legend()
    
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
            self.draw_rotated_x(self.screen, int(drone[0]), int(drone[1]), 10, self.drone_directions[i])
        
        # Desenhar alvos
        for i, target in enumerate(self.targets):
            color = (0, 255, 0) if i in self.reached_targets else (255, 0, 0)
            pygame.draw.circle(self.screen, color, (int(target[0]), int(target[1])), 5)
        
        pygame.display.flip()
        
        # Limitar a taxa de quadros
        self.clock.tick(60)
        
        # Verificar se a janela está fechada
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True
   

    def close(self):
        pygame.quit()





if __name__ == "__main__":
    env = DroneEnv(False)
    
    totalMetrics = []
    # Testar o ambiente com ações calculadas
    for episode in range(20):
        observation = env.reset()
        done = False

        while not done:
            
            if env.render_enabled:            
                env.render()
            
            # Definir uma ação para mover os drones em direção aos alvos
            action = np.full(NUM_DRONES, 5)

            observation, reward, done, info = env.step(action)
            
            if done:
                totalMetrics.append(info)
                #print(info)
                #print("Time steps for the current round:", info["time_steps"])
                #print("Total distance traveled by drones in the current round:", info["total_distance"])

    env.close()
    
    metricsDf = pd.DataFrame(totalMetrics)
    
    # Chamar a função de plotagem
    env.plot_metrics(metricsDf, len(env.drones), len(env.targets))
    #env.plot_convergence(metricsDf, len(env.drones), len(env.targets))

