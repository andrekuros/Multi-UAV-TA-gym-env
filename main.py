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

from swarm_gap import swarm_gap
from swarm_gap import swarm_gap_task_assign
import pandas as pd

import numpy as np
import random


#env = DroneEnv(mode="Actions", render_enabled=True)
env = DroneEnv(mode="TaskAssign", render_enabled=True)

totalMetrics = []

exangeTime = 100

# Testar o ambiente com ações calculadas pelo SWARM-GAP
for episode in range(3):
    
    observation = env.reset(change_scene=True, seed=episode)
    done = False

    while not done:

        if env.render_enabled:
            env.render()
            
        if env.time_steps % exangeTime:
            
            actions = send_token(env)
        
        
        #print (actions)

        observation, reward, done, info = env.step(actions)

        if done:
            totalMetrics.append(info)

env.close()

metricsDf = pd.DataFrame(totalMetrics)

# Chamar a função de plotagem
env.plot_metrics(metricsDf, len(env.drones), len(env.targets))
#env.plot_convergence(metricsDf, len(env.drones), len(env.targets))

