import numpy as np
import random
from DroneEnv import DroneEnv
import pandas as pd


class SwarmGap:
    def __init__(self, env, exchange_interval=100, seed=0):
        
        self.seed = seed
        self.env = env
        self.num_agents = env.NUM_DRONES
        self.num_tasks = env.NUM_TARGETS       
        self.exchange_interval = exchange_interval
                
        self.task_assignment = {i: [] for i in range(self.num_agents)}
        
        self.token_exchange_list = np.random.permutation(np.arange(0, self.num_agents))

   
    def process_token(self):
        
        if len(self.token_exchange_list) == 0:            
            self.token_exchange_list = np.random.permutation(np.arange(0, self.num_agents))
        
        drone_id = self.token_exchange_list[0]
        
        drone = env.drones[drone_id]
        pendingTasks = env.unallocated_tasks()
        
        if len(pendingTasks) == 0:
            return
        
        distances = np.linalg.norm(np.array([env.targets[task].position for task in pendingTasks]) - drone.position, axis=1)       
        max_cap = np.max(1/distances)       
        capability = (1/distances) / max_cap
        
        # Normalize os valores para que eles somem 1
        probabilities = capability / capability.sum()

        # Escolha um item aleatoriamente com base nas probabilidades
        chosen_index = np.random.choice(np.arange(len(capability)), p=probabilities)
        
        taskSelected = pendingTasks[chosen_index]        
        action = {}
        action[drone_id] = [taskSelected]
               
        #"Send" token to Next in order        
        self.token_exchange_list = np.delete(self.token_exchange_list,0)
        
        return action

   
# Exemplo de uso
env = DroneEnv(action_mode= "TaskAssign", num_drones=5, num_targets = 20, render_enabled=False)
env.reset()

swarm_gap = SwarmGap(env, exchange_interval=8)

totalMetrics = []
# Testar o ambiente com ações calculadas
for episode in range(50):
    
   
    observation = env.reset( seed = episode)
    done = False
      
    #actions = env.generate_random_tasks_all() 
    #env.step(actions)
    
    print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")      
    while not done :
            
       if env.render_enabled:
           env.render()
           
       actions = None
       
       if env.time_steps % swarm_gap.exchange_interval == 0:
           
           actions = swarm_gap.process_token()                                                  

       observation, reward, done, info = env.step(actions)
       
       if done:
           totalMetrics.append(info)           

env.close()

metricsDf = pd.DataFrame(totalMetrics)

# Chamar a função de plotagem
env.plot_metrics(metricsDf, len(env.drones), len(env.targets), "SWARM-GAP")
env.plot_convergence(metricsDf, len(env.drones), len(env.targets), "SWARM-GAP")
