import numpy as np
from DroneEnv import MultiDroneEnv
import pandas as pd

class TessiAgent:
    def __init__(self, num_drones, num_targets, tessi_model = 1):
        self.num_drones = num_drones
        self.num_targets = num_targets
        self.tessi_model = tessi_model

    def allocate_tasks(self, drone_states, target_states):
        action = {}
                
        selected = set()
        #print(target_states[0].target_id)
        for drone in drone_states:
            min_distance = float('inf')
            chosen_target = None
                        
            for target in target_states:
                
                if target.target_id in selected:
                    continue
                
                distance = np.linalg.norm(drone.position - target.position) 

                if distance < min_distance:
                    min_distance = distance
                    chosen_target = target.target_id
                    

            if chosen_target != None:
                
                if self.tessi_model == 1:                
                    action[drone.drone_id] = chosen_target
                    selected.add(chosen_target)
                #elif self.tessi_model == 2:
                #    if chosen_target not in allocation:
                #        action[drone.id] = chosen_target

        return action
   
    



