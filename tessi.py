import numpy as np
from DroneEnv import MultiDroneEnv
import pandas as pd

class TessiAgent:
    def __init__(self, num_drones, n_tasks, tessi_model = 1):
        self.num_drones = num_drones
        self.n_tasks = n_tasks
        self.tessi_model = tessi_model

    def allocate_tasks(self, drone_states, task_states):
        action = {}
                
        selected = set()
        #print(task_states[0].task_id)
        for drone in drone_states:
            min_distance = float('inf')
            chosen_task = None
                        
            for task in task_states:
                
                if task.task_id in selected:
                    continue
                
                distance = np.linalg.norm(drone.position - task.position) 

                if distance < min_distance:
                    min_distance = distance
                    chosen_task = task.task_id
                    

            if chosen_task != None:
                
                if self.tessi_model == 1:                
                    action[drone.drone_id] = chosen_task
                    selected.add(chosen_task)
                #elif self.tessi_model == 2:
                #    if chosen_task not in allocation:
                #        action[drone.id] = chosen_task

        return action
   
    



