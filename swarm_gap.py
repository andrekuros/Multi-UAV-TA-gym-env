import numpy as np
import random

class SwarmGap:
    def __init__(self, drones, tasks, exchange_interval=100, seed=0):
        
        self.seed = seed
        
        self.drones = drones
        self.tasks = tasks        
        
        self.n_agents = len(drones)
        self.n_tasks = len(tasks) 
                
        #self.unallocated_tasks = [tgt.task_id for tgt in tasks]
        self.drones_out = []
        
        self.exchange_interval = exchange_interval
                
        self.task_assignment = {i: [] for i in range(self.n_agents)}
        
        self.token_exchange_list = np.random.permutation(np.arange(0, self.n_agents))

   
    def process_token(self, drones, tasks):
                
        self.drones = drones
        action = None
        
        if len(self.token_exchange_list) == 0:            
            self.token_exchange_list = np.random.permutation(np.arange(0, self.n_agents))
           
                
        drone_id = self.token_exchange_list[0]        
        drone = self.drones[drone_id]
                                       
        if drone.state != -1:
            
            distances = np.linalg.norm(np.array([task.position for task in tasks]) - drone.next_free_position, axis=1)       
            
            Qs = np.array([self.drones[drone_id].fit2Task[task.typeIdx] for task in tasks])
        
            max_dist = np.max(distances)       
            max_Q = np.max(Qs)
            
            if max_Q == 0.0:
                self.drones[drone_id].has_capability = False
                self.drones_out.append(drone_id)
                #print("Drone_out", drone_id)            
            else:
            
                #Higer Alpha priorize Distance    
                alpha = 0.6
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
                    taskSelected = tasks[chosen_index]        
                    action = {}
                    action["agent" + str(drone_id)] = taskSelected.task_id
                    #self.unallocated_tasks.remove(taskSelected)
                  
        #"Send" token to Next in order        
        self.token_exchange_list = np.delete(self.token_exchange_list,0)
        
        if len(self.token_exchange_list) == 0:
            
            availables = []
            for drone in self.drones:
                if drone.has_capability and drone.state != -1:
                    availables.append(drone.drone_id)
            
            random.shuffle(availables)
            #self.token_exchange_list = np.random.permutation(np.arange(0, self.n_agents))
            #print("Restart Token", availables)
        
        return action

