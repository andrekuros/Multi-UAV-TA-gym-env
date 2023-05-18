#%%
import numpy as np
from DroneEnv import MultiDroneEnv
from DroneEnv import env
from swarm_gap import SwarmGap
from tessi import TessiAgent
import pandas as pd
#import argparse
#import json
import time
#import copy
#import concurrent.futures
#import cProfile

#from gym import spaces
#from godot_rl.core.godot_env import GodotEnv
#from godot_rl.core.utils import lod_to_dol

import MultiDroneEnvUtils as utils



algorithms = ["Random"]
algorithms += ["Tessi1"] #"Swarm-GAP" Tessi1

episodes = 300

config = utils.DroneEnvOptions(  
    
    render_speed = -1,
    max_time_steps = 300,
    action_mode= "TaskAssign",
    agents = {"F1" : 1, "R1" : 1 } ,
    tasks = { "Rec" : 6 , "Att" : 4 }   ,
    num_obstacles = 0        ,
    hidden_obstacles = False,
    fail_rate = 0.01 )

config=None

simEnv = "PyGame"

if simEnv == "PyGame":
    worldModel = MultiDroneEnv(config)
    env = env
    
#elif simEnv == "Godot":
#    env = GodotEnv()
if False:

    from pettingzoo.test import parallel_api_test
    from pettingzoo.test import parallel_seed_test

    parallel_api_test(worldModel, num_cycles=1000)
    #parallel_seed_test(env, num_cycles=100, test_kept_state=True)
   

totalMetrics = []
total_reward = {}

for algorithm in algorithms:
    
    start_time = time.time()
    print("\nStarting Algorithm:", algorithm)
    
    total_reward[algorithm] = []    
    
    for episode in range(episodes):
                
        observation  = worldModel.reset(seed=episode)         
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
        
        if algorithm == "Swarm-GAP":
            agent = SwarmGap(drones, tasks, quality_table, exchange_interval = 1)
    
        print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")   
        
        episodo_reward = 0
        while not all(done.values()) and not all(truncations.values()):
                            
            actions = None
                        
            if algorithm == "Random":
                            
                if worldModel.time_steps % 1 == 0 and worldModel.time_steps >= 2:
                    #print(planned_actions, worldModel.time_steps)
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
                        
                    #print("plan->:",actions) 
                            
            elif algorithm == "Swarm-GAP":
                
                if worldModel.time_steps % agent.exchange_interval:                    
                    actions = agent.process_token()    
            
            elif algorithm == "Tessi1":            
                                                                                                        
                if worldModel.time_steps % 1 == 0 :
                    # Convert task_allocation to actions                    
                    un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                                        
                    actions = agent.allocate_tasks(worldModel.agents_obj, un_taks_obj )
                    #actions = agent.allocate_tasks(worldModel.agents_obj, [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] )                    
                    
            #if actions != {} and actions != None:
            #       print(actions)
            observation, reward, done, truncations, info = worldModel.step(actions)
            #print({'agent_id': worldModel.agent_selector.next()})            
            episodo_reward += sum(reward.values())/worldModel.n_agents
            
            if worldModel.render_enabled:
                worldModel.render()
            
            #print(done)
            if all(done.values()):
            #if done:
                metrics = info['metrics']
                metrics["Algorithm"] = algorithm
                totalMetrics.append(metrics)
                #print("Done Rew:", reward["agent0"])
                                            
            if all(truncations.values()):                
                #print("\nMax Steps Reached:", worldModel.time_steps )
                metrics = info['metrics']
                metrics["Algorithm"] = algorithm
                totalMetrics.append(metrics)

        total_reward[algorithm].append(episodo_reward) 
        #print(worldModel.time_steps)#print("Trunc Rew:", total_reward)
                    
    end_time = time.time()
    execution_time = end_time - start_time
    print("\nExecution time ", algorithm, execution_time, "seconds")
    
    worldModel.close()

import matplotlib.pyplot as plt

for alg in algorithms:
    print(f'Rew({alg}): {np.mean(total_reward[alg])}')
    print(f'Rew({alg}): Max: {max(total_reward[alg])}, Min: {min(total_reward[alg])}')
    #plt.hist(total_reward[alg], bins=100)
    #plt.show()

#%%%
metricsDf = pd.DataFrame(totalMetrics)
worldModel.plot_metrics(metricsDf, len(worldModel.agents), worldModel.n_tasks)

for algorithm in algorithms:
    worldModel.plot_convergence(metricsDf[metricsDf.Algorithm == algorithm], len(worldModel.agents), len(worldModel.tasks), algorithm)

print(metricsDf.mean())








