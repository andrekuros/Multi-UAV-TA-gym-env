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
                 agents= {"F1" : 0, "F2" : 0, "R1" : 1, "R2" : 1},                 
                 tasks= { "Att" : 0 , "Rec" : 2, "Hold" : 0},
                 multiple_tasks_per_agent = False,
                 multiple_agents_per_task = True,
                 random_init_pos=False,
                 num_obstacles=0,
                 hidden_obstacles = False,
                 fail_rate = 0.0,
                 threats_list = [("T1", 4), ("T2" , 2)],
                 fixed_seed = -1,
                 info = "No Info",
                 # RL experiment toggles (see RL_EXPERIMENT_PLAN.md)
                 early_terminate=False,
                 capability_mask=False,
                 saturate_mask=False,
                 reward_weights=None,
                 # Dynamic TA (publication roadmap)
                 arrival_rate=0.0,
                 include_time_windows=False,
                 dynamic_idle_penalty=0.0,
                 # WPS (Windowed Pop-up Strike) — harder dynamic claim
                 sense_radius=0.0,
                 threat_delay=0,
                 hard_windows=False,
                 window_length=30,
                 burst_mode=False,
                 burst_size=3,
                 miss_penalty=25.0,
                 on_time_bonus=10.0,
                 dual_region_bursts=False,
                 share_knowledge=True,
                 commit_horizon=0,
                 reassign_penalty=0.0,
                 # Escort / coalition protection
                 escort_enabled=False,
                 escort_radius=70.0,
                 escort_requirement=1.2,
                 escort_intercept_radius=100.0,
                 mutual_support_radius=80.0,
                 escort_agent_types=("F1", "F2")):
        
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
        self.threats_list = threats_list
        self.fixed_seed = fixed_seed
        self.info = info
        self.early_terminate = early_terminate
        self.capability_mask = capability_mask
        self.saturate_mask = saturate_mask
        # Defaults match current production weights; override for ablations.
        self.reward_weights = reward_weights or {
            "action": 0.0,
            "distance": 1.0,
            "quality": 1.0,
            "s_quality": 1.0,
            "time": 0.0,
            "alloc": 0.0,
            "time_penaulty": 0.0,
            "step": 0.0,
        }
        self.arrival_rate = arrival_rate
        self.include_time_windows = include_time_windows
        self.dynamic_idle_penalty = dynamic_idle_penalty
        self.sense_radius = sense_radius
        self.threat_delay = threat_delay
        self.hard_windows = hard_windows
        self.window_length = window_length
        self.burst_mode = burst_mode
        self.burst_size = burst_size
        self.miss_penalty = miss_penalty
        self.on_time_bonus = on_time_bonus
        self.dual_region_bursts = dual_region_bursts
        self.share_knowledge = share_knowledge
        self.commit_horizon = int(commit_horizon or 0)
        self.reassign_penalty = float(reassign_penalty or 0.0)
        self.escort_enabled = bool(escort_enabled)
        self.escort_radius = float(escort_radius or 70.0)
        self.escort_requirement = float(escort_requirement or 1.2)
        self.escort_intercept_radius = float(escort_intercept_radius or 100.0)
        self.mutual_support_radius = float(mutual_support_radius or 80.0)
        self.escort_agent_types = tuple(escort_agent_types or ("F1", "F2"))

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
            for task in list(agent.tasks):
                if agent.desAllocate(task):
                    tid = getattr(task, "id", None)
                    if tid is None or tid < 0 or tid >= len(env.allocation_table):
                        continue
                    bucket = env.allocation_table[tid]
                    if isinstance(bucket, set) and agent.name in bucket:
                        bucket.remove(agent.name)
                    elif isinstance(bucket, (list, set)) and agent.name in bucket:
                        try:
                            bucket.remove(agent.name)
                        except (ValueError, KeyError):
                            pass
                    
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


