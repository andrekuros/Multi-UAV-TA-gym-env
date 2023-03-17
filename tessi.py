import numpy as np
from DroneEnv import DroneEnv
import pandas as pd

class TessiAgent:
    def __init__(self, num_drones, num_targets):
        self.num_drones = num_drones
        self.num_targets = num_targets

    def allocate_tasks(self, drone_states, target_states):
        allocation = []
        
        #print(target_states[0].target_id)
        for drone in drone_states:
            min_distance = float('inf')
            chosen_target = None
                        
            for target in target_states:
                distance = np.linalg.norm(drone.position - target.position) 

                if distance < min_distance:
                    min_distance = distance
                    chosen_target = target.target_id
                    

            if chosen_target != None:
                allocation.append(chosen_target)

        return allocation
   
    
totalMetrics = []

env = DroneEnv(action_mode= "TaskAssign", num_drones=5, num_targets = 20, render_enabled=False)
env.reset()
agent = TessiAgent(num_drones=5, num_targets=20)


for episode in range(100):
    observations = env.reset(seed = episode)

    done = False
    
    print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")   
    while not done:
        
        #print([env.targets[i].target_id for i in env.unallocated_tasks()])
        task_allocation = agent.allocate_tasks(env.drones, [env.targets[i] for i in env.unallocated_tasks()] )
                
        actions = None
        
        if env.time_steps % 1 == 0:
            # Convert task_allocation to actions
            actions = {}
            for drone_id, target_id in enumerate(task_allocation):
                actions[drone_id] = [target_id]
                               
        observation, reward, done, info = env.step(actions)

        if done:
            totalMetrics.append(info)


        if env.render_enabled:            
            env.render()
       
env.close()

metricsDf = pd.DataFrame(totalMetrics)

# Chamar a função de plotagem
env.plot_metrics(metricsDf, len(env.drones), len(env.targets), "SWARM-GAP")
env.plot_convergence(metricsDf, len(env.drones), len(env.targets), "SWARM-GAP")





