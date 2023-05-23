import numpy as np
from DroneEnv import MultiDroneEnv
import pandas as pd

class TessiAgent:
    def __init__(self, num_drones, n_tasks, max_dist, tessi_model = 1):
        self.num_drones = num_drones
        self.n_tasks = n_tasks
        self.tessi_model = tessi_model
        self.max_dist = max_dist

    def allocate_tasks(self, drone_states, task_states):
        action = {}        
                
        selected = set()
        #print(task_states[0].task_id)
        
        for drone in drone_states:
                        
            min_rew = float('inf')
            chosen_task = None
                        
            for task in task_states:
                
                if task.task_id in selected:
                    continue
                                
                distance = np.linalg.norm(drone.next_free_position - task.position) / self.max_dist
                quality = drone.fit2Task[task.typeIdx]
                
                if -6.0 * distance  + 4.0 * quality < min_rew:
                    min_rew = distance * quality
                    chosen_task = task.task_id
                    

            if chosen_task != None:
                
                if self.tessi_model == 1:                
                    action[drone.name] = chosen_task
                    selected.add(chosen_task)
                #elif self.tessi_model == 2:
                #    if chosen_task not in allocation:
                #        action[drone.id] = chosen_task

        return action
   
    



