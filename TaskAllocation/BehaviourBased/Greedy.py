import numpy as np


class GreedyAgent:
    def __init__(self, greedy_model = 1):
        self.greedy_model = greedy_model
        
    
    
    def allocate_tasks(self, drone_states, task_states):
        action = []        
                
        selected = set()
        #print(task_states[0].task_id)
        min_dist = float('+inf')
        chosen_task = None
        drone_name = None
        
        for drone in drone_states:
            
            for task in task_states:
                
                if task.id in selected:
                    continue
                                
                distance = np.linalg.norm(drone.next_free_position - task.position) 
                # quality = drone.fit2Task[task.typeIdx]
                
                if  distance < min_dist:
                    min_dist =  distance 
                    chosen_task = task
                    drone_name = drone.name
                    

        if chosen_task is not None:
            
                        
            action.append((drone_name,chosen_task))
           
            #elif self.tessi_model == 2:
            #    if chosen_task not in allocation:
            #        action[drone.id] = chosen_task

        return action
   
    



