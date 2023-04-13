from DroneEnv import MultiDroneEnv
from DroneEnv import env
from swarm_gap import SwarmGap
from tessi import TessiAgent
import pandas as pd
import argparse
import json
import time
import copy
import concurrent.futures
import cProfile

from gym import spaces
from godot_rl.core.godot_env import GodotEnv
from godot_rl.core.utils import lod_to_dol

import MultiDroneEnvUtils as utils


algorithms = ["Random","Tessi1"] #"Swarm-GAP"

episodes = 3

config = utils.DroneEnvOptions(  
    
    render_speed = 5,
    max_time_steps = 5000,
    action_mode= "TaskAssign",
    agents = {"R1" : 10 ,"C1" :2 } ,
    tasks = { "Rec" : 20 }   ,
    num_obstacles = 5        ,
    hidden_obstacles = False )


simEnv = "PyGame"

if simEnv == "PyGame":
    worldModel = MultiDroneEnv(config)
    env = env
    
#elif simEnv == "Godot":
#    env = GodotEnv()
if True:

    from pettingzoo.test import parallel_api_test
    from pettingzoo.test import parallel_seed_test

    parallel_api_test(worldModel, num_cycles=1000)
    #parallel_seed_test(env, num_cycles=100, test_kept_state=True)
    exit

totalMetrics = []

for algorithm in algorithms:
    
    start_time = time.time()
    print("\nStarting Algorithm:", algorithm)
        
    for episode in range(episodes):
        
        observation  = worldModel.reset(seed=0)#episode)        
        info         = worldModel.get_initial_state()
        
        drones = info["drones"]
        tasks = info["tasks"]
        quality_table =  info["quality_table"]
        
        done = {0 : False}
        truncations = {0 : False}
                        
        if algorithm == "Random":            
            planned_actions = utils.generate_random_tasks_all(drones, tasks, seed = episode) 
            #print(planned_actions)
        
        if algorithm == "Tessi1":
            agent = TessiAgent(num_drones=worldModel.n_agents, n_tasks=worldModel.n_tasks, tessi_model = 1)
        
        if algorithm == "Tessi2":
            agent = TessiAgent(num_drones=worldModel.n_agents, n_tasks=worldModel.n_tasks, tessi_model = 2)
        
        if algorithm == "Swarm-GAP":
            agent = SwarmGap(drones, tasks, quality_table, exchange_interval=1)
    
        print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")   
        
        while not all(done.values()) and not all(truncations.values()):
                            
            actions = None
                        
            if algorithm == "Random":
                            
                if len(planned_actions) > 0:
                    
                    actions = {}                     
                    toDelete = [] 
                     
                    for i, tasks in planned_actions.items():                                             
                        
                        if len(tasks) > 0:
                            actions[i] = planned_actions[i].pop()                      
                        else:
                            toDelete.append(i)
                
                    for i in toDelete: 
                        del planned_actions[i] 
                            
            elif algorithm == "Swarm-GAP":
                
                if worldModel.time_steps % agent.exchange_interval == 0:                    
                    actions = agent.process_token()    
            
            elif algorithm == "Tessi1" or algorithm == "Tessi2":            
                
                
                                                        
                if worldModel.time_steps % 10 == 0:
                    # Convert task_allocation to actions
                    actions = agent.allocate_tasks(worldModel.drones, [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] )
                                    
            observation, reward, done, truncations, info = worldModel.step(actions)
                        
            #for agent in range(env.NUM_DRONES):                
            #    print(observation[agent])
                #if not all(obs["task_status"], -1):   
                #    print(observation)
            
            if worldModel.render_enabled:
                worldModel.render()
            
            #print(done)
            if all(done.values()):
            #if done:
                info["Algorithm"] = algorithm
                totalMetrics.append(info)
                            
            if all(truncations.values()):                
                print("\nMax Steps Reached:", worldModel.time_steps )
                
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("\nExecution time ", algorithm, execution_time, "seconds")
    
    worldModel.close()



metricsDf = pd.DataFrame(totalMetrics)
# Chamar a função de plotagem

worldModel.plot_metrics(metricsDf, len(worldModel.drones), len(worldModel.tasks))
#for algorithm in algorithms:
#    worldModel.plot_convergence(metricsDf[metricsDf.Algorithm == algorithm], len(worldModel.drones), len(worldModel.tasks), algorithm)


#%%%





