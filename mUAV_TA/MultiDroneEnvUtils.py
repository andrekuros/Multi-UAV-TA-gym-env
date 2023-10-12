import random
import numpy as np
from typing import List

class agentEnvOptions:
    def __init__(self, 
                 render_mode = 'human',                  
                 render_speed=-1,
                 simulation_frame_rate = 0.01, 
                 action_mode="TaskAssign",
                 simulator_module = "Internal", 
                 max_time_steps=150, 
                 agents= {"F1" : 4, "F2" : 2, "R1" : 6},                 
                 tasks= { "Att" : 4 , "Rec" : 16, "Hold" : 4},
                 multiple_tasks_per_agent = False,
                 multiple_agents_per_task = True,
                 random_init_pos=False,
                 num_obstacles=0,
                 hidden_obstacles = False,
                 fail_rate = 0.0,
                 info = "No Info"):
        
        self.render_mode = render_mode 
        self.render_speed = render_speed
        self.simulation_frame_rate = simulation_frame_rate
        self.action_mode=action_mode 
        self.simulator_module = simulator_module        
        self.max_time_steps = max_time_steps                
        self.random_init_pos = random_init_pos
        self.agents = agents
        self.tasks = tasks        
        self.multiple_tasks_per_agent = multiple_tasks_per_agent
        self.multiple_agents_per_task = multiple_agents_per_task
        self.num_obstacles = num_obstacles
        self.hidden_obstacles = hidden_obstacles
        self.fail_rate = fail_rate
        self.info = info

class EnvUtils:
    
    def generate_random_tasks_all(agents, tasks, seed = 0):
        
        """
        This function generates a random task assignment for each agent in the environment.
        
        Args:
        agents (list): A list of agents in the environment.
        tasks (list): A list of tasks in the environment.
        
        Returns:
        task_actions (dict): A dictionary containing task assignments for each agent, with the agent index as the key
                            and the list of assigned tasks as the value.
        """
        rndGen = random.Random(seed)
        task_list = [task.task_id for task in tasks]#list(range(len(tasks)))
        rndGen.shuffle(task_list)

        # Calculate the minimum number of tasks per agent and the number of agents that will receive an extra task
        min_tasks_per_agent, extra_tasks = divmod(len(tasks), len(agents))

        # Generate task actions for the agents
        task_actions = {}
        task_idx = 0
        
        for i in range(len(agents)):
            n_tasks = min_tasks_per_agent + (1 if i < extra_tasks else 0)
            task_actions[f'agent{i}'] = task_list[task_idx:task_idx + n_tasks]
            task_idx += n_tasks

        return task_actions


    def get_perpendicular_normalized_vector(direction):
        normal_vector = np.array([-direction[1], direction[0]])
        normalized_vector = normal_vector / np.linalg.norm(normal_vector)
        return normalized_vector


    def are_vectors_aligned_with_margin(a, b, margin_degrees):
        # Normalizar os vetores
        a_normalized = a / np.linalg.norm(a)
        b_normalized = b / np.linalg.norm(b)

        # Calcular o produto escalar
        dot_product = np.dot(a_normalized, b_normalized)

        # Calcular o ângulo entre os vetores (em graus)
        angle_degrees = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

        # Verificar se o ângulo está dentro da margem especificada
        return np.abs(angle_degrees) < margin_degrees or np.abs(angle_degrees - 180) < margin_degrees

    def rotate_vector_90_degrees_clockwise(vector):
        return np.array([vector[1], -vector[0]])

    def rotate_vector_90_degrees_counterclockwise(vector):
        return np.array([-vector[1], vector[0]])


    def norm_vector(vector):
        # Calcular a magnitude do vetor
        magnitude = np.linalg.norm(vector)

        if magnitude == 0:
            return np.array([0,0])
        # Dividir cada componente do vetor pela magnitude
        normalized_vector = vector / magnitude

        return normalized_vector
    
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    
    def desallocateAll(agents: list, env):
        
        for agent in agents:
                        
            #Remove the agent caps from the task
            for task in agent.tasks:                
                if agent.desAllocate(task):
                    env.allocation_table[task.id].remove(agent.id)
                    
            agent.tasks = [env.task_idle]
            agent.status = 0
            agent.next_free_time = env.time_steps
            agent.next_free_position = agent.position

class ACMIExporter:
    def __init__(self):
        self.acmi_data = []
        self.header = 'FileType=text/acmi/tacview\nFileVersion=2.2\n'

    def add_drone_state(self, timestep, agent):
        """
        Add drone state for a given timestep.

        :param timestep: Current simulation timestep
        :param drone_id: ID of the drone
        :param position: Position of the drone as "x,y,z"
        :param orientation: Orientation of the drone as "pitch,yaw,roll". Default is "0,0,0".
        """
        # Using the official TacView ACMI text format for object creation and updates
        self.acmi_data.append(f'#{timestep}')
        self.acmi_data.append(f'{agent.id},type=Air+FixedWing')
        self.acmi_data.append(f'{agent.id},T={agent.position[0]}|{agent.position[1]}|5000.0, name="agent{agent.id}"')        

    def export_to_acmi(self, filepath):
        """
        Export the accumulated drone states to an .acmi file.

        :param filepath: Path where the .acmi file should be saved.
        """
        acmi_content = self.header
        for entry in self.acmi_data:
            acmi_content += entry + "\n"
        
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(acmi_content)


