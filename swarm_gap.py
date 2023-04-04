import numpy as np
import random

class SwarmGap:
    def __init__(self, drones, targets, quality_table, exchange_interval=100, seed=0):
        
        self.seed = seed
        
        self.drones = drones
        self.targets = targets
        self.quality_table = quality_table
        
        self.num_agents = len(drones)
        self.num_targets = len(targets) 
                
        self.unallocated_tasks = [tgt.target_id for tgt in targets]
        self.drones_out = []
        
        self.exchange_interval = exchange_interval
                
        self.task_assignment = {i: [] for i in range(self.num_agents)}
        
        self.token_exchange_list = np.random.permutation(np.arange(0, self.num_agents))

   
    def process_token(self):
        
        action = None
        
        if len(self.token_exchange_list) == 0:            
            self.token_exchange_list = np.random.permutation(np.arange(0, self.num_agents))
        
        drone_id = self.token_exchange_list[0]        
        drone = self.drones[drone_id]
                 
        
        if len(self.unallocated_tasks) == 0:
            return
        
        distances = np.linalg.norm(np.array([self.targets[i].position for i in self.unallocated_tasks]) - drone.position, axis=1)       
        
        Qs = np.array([self.quality_table[drone_id][task] for task in self.unallocated_tasks])
       
        max_dist = np.max(distances)       
        max_Q = np.max(Qs)
        
        if max_Q == 0.0:
            self.drones[drone_id].has_capability = False
            self.drones_out.append(drone_id)
            #print("Drone_out", drone_id)            
        else:
        
            alpha = 0.5
            st = 0.6
                       
            capability = (max_dist - distances) / max_dist * alpha + (1 - (max_Q  - Qs) / max_Q) * (1 - alpha) 
            #print( distances)
            # Normalize os valores para que eles somem 1
            tendencies = pow(st,2) / (pow(st,2) + np.square(capability))
            
            chosen_index = - 1
                        
            
            for i,t in enumerate(tendencies):        
                #chosen_index = np.random.choice(np.arange(len(capability)), p=tendencies)
                randN = random.uniform(0,1)
                if t < randN:
                    chosen_index = i
                    break
                #else:
                 #   print("rejected", t, " vs ", randN)
            
            #Rest only one drone
            if len(self.drones_out) == len(self.drones) - 1:
                chosen_index = np.argmin(tendencies)
                
            
            if chosen_index >= 0:
                taskSelected = self.unallocated_tasks[chosen_index]        
                action = {}
                action[drone_id] = taskSelected
                self.unallocated_tasks.remove(taskSelected)
            
            
                
                    
          
                   
        #"Send" token to Next in order        
        self.token_exchange_list = np.delete(self.token_exchange_list,0)
        
        if len(self.token_exchange_list) == 0:
            
            availables = []
            for drone in self.drones:
                if drone.has_capability:
                    availables.append(drone.drone_id)
            
            random.shuffle(availables)
            #self.token_exchange_list = np.random.permutation(np.arange(0, self.num_agents))
            #print("Restart Token", availables)
        
        return action

