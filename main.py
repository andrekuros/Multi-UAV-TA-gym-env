from DroneEnv import DroneEnv
from genetic_algorithm import genetic_algorithm
from swarm_gap import swarm_gap
import pandas as pd

import numpy as np
import random

#--------------- Evaluate GA--------------------------------

#recorde 2.67
# Configurar os hiperparâmetros do algoritmo genético
population_size = 30
num_generations = 20
crossover_prob = 0.7
mutation_prob = 0.2


# Criar o ambiente
env = DroneEnv(render_enabled=False,  action_mode= "TaskAssign")
env.reset( seed = 0)

# Executar o algoritmo genético e obter o melhor cromossomo
best_chromosome = genetic_algorithm(
    env,
    population_size,
    num_generations,
    crossover_prob,
    mutation_prob
)



#%%%
#------------ Evaluate Swarm-GAP--------------------------------------------
from DroneEnv import DroneEnv

from swarm_gap import SwarmGap
from tessi import TessiAgent
import pandas as pd
import numpy as np
import seaborn as sns
import random


algorithms = ["Random","Swarm-GAP", "Tessi"]

num_drones=5
num_targets = 20

#env = DroneEnv(mode="Actions", render_enabled=True)
env = DroneEnv(action_mode= "TaskAssign", num_drones=num_drones, num_targets=num_targets, render_enabled=False)

totalMetrics = []


# Testar o ambiente com ações calculadas pelo SWARM-GAP
for algorithm in algorithms:
    
    print("\nStarting Algorithm:", algorithm)
    
   
    
    
    for episode in range(100):
        
        observation = env.reset(seed=episode)
        done = False
         
        if algorithm == "Random":
            agent = True        
        
        if algorithm == "Tessi":
            agent = TessiAgent(num_drones=num_drones, num_targets=num_targets)
        
        if algorithm == "Swarm-GAP":
            agent = SwarmGap(env, exchange_interval=8)
    
        print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")   
        
        while not done:
                            
            actions = None
            
            
            if algorithm == "Random":
                
                if agent:
                    actions = env.generate_random_tasks_all()
                    agent = False
                
            
            elif algorithm == "Swarm-GAP":
                
                if env.time_steps % swarm_gap.exchange_interval == 0:                    
                    actions = swarm_gap.process_token()    
            
            elif algorithm == "Tessi":            
                
                task_allocation = agent.allocate_tasks(env.drones, [env.targets[i] for i in env.unallocated_tasks()] )
                                                        
                if env.time_steps % 1 == 0:
                    # Convert task_allocation to actions
                    actions = {}
                    for drone_id, target_id in enumerate(task_allocation):
                        actions[drone_id] = [target_id]
            
        
            observation, reward, done, info = env.step(actions)
            
            if env.render_enabled:
                env.render()
    
            if done:
                info["Algorithm"] = algorithm
                totalMetrics.append(info)
    
    env.close()

metricsDf = pd.DataFrame(totalMetrics)
# Chamar a função de plotagem


#%%%
env.plot_metrics(metricsDf, len(env.drones), len(env.targets))
for algorithm in algorithms:
    env.plot_convergence(metricsDf[metricsDf.Algorithm == algorithm], len(env.drones), len(env.targets), algorithm)

