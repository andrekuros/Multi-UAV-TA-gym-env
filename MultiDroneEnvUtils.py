import random
import numpy as np

class DroneEnvOptions:
    def __init__(self, 
                 render_mode = 'human',                  
                 render_speed=-1,
                 simulation_frame_rate = 0.2, 
                 action_mode="TaskAssign",
                 simulator_module = "Internal", 
                 max_time_steps=300, 
                 agents= {"F1" : 2, "F2" : 2, "R1" : 3, "R2" : 3},                 
                 tasks= { "Att" : 8 , "Rec" : 22},
                 random_init_pos=True,
                 num_obstacles=0,
                 hidden_obstacles = False,
                 fail_rate = 0.0):
        
        self.render_mode = render_mode 
        self.render_speed = render_speed
        self.simulation_frame_rate = simulation_frame_rate
        self.action_mode=action_mode 
        self.simulator_module = simulator_module        
        self.max_time_steps = max_time_steps                
        self.random_init_pos = random_init_pos
        self.agents = agents
        self.tasks = tasks        
        self.num_obstacles = num_obstacles
        self.hidden_obstacles = hidden_obstacles
        self.fail_rate = fail_rate

def generate_random_tasks_all(drones, tasks, seed = 0):
    
    """
    This function generates a random task assignment for each drone in the environment.
    
    Args:
    drones (list): A list of drones in the environment.
    tasks (list): A list of tasks in the environment.
    
    Returns:
    task_actions (dict): A dictionary containing task assignments for each drone, with the drone index as the key
                         and the list of assigned tasks as the value.
    """
    rndGen = random.Random(seed)
    task_list = [task.task_id for task in tasks]#list(range(len(tasks)))
    rndGen.shuffle(task_list)

    # Calculate the minimum number of tasks per drone and the number of drones that will receive an extra task
    min_tasks_per_drone, extra_tasks = divmod(len(tasks), len(drones))

    # Generate task actions for the drones
    task_actions = {}
    task_idx = 0
    
    for i in range(len(drones)):
        n_tasks = min_tasks_per_drone + (1 if i < extra_tasks else 0)
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


