#%%
import os
import random
import numpy as np
from DroneEnv import MultiDroneEnv
from DroneEnv import env
from swarm_gap import SwarmGap
from tessi import TessiAgent
from CBBA import CBBA
import pandas as pd
import matplotlib.pyplot as plt
import time
from Tianshou_Policy import _get_model
import torch
from tianshou.data import Batch

#from gym import spaces
#from godot_rl.core.godot_env import GodotEnv
#from godot_rl.core.utils import lod_to_dol

import MultiDroneEnvUtils as utils

algorithms = []
algorithms += ['Random']
algorithms += ["Greedy"]
algorithms += ["Swarm-GAP"]
algorithms += ["CBBA"]
algorithms +=  ["TBTA"]
#algorithms +=  ["TBTA2"]

print(algorithms)
episodes = 20

config = utils.DroneEnvOptions(     
    render_speed = -1,
    max_time_steps = 1000,
    action_mode= "TaskAssign",
    agents= {"F1" : 4,"R1" : 6},                 
    tasks= { "Att" : 6 , "Rec" : 14},
    random_init_pos = False,
    num_obstacles = 0,
    hidden_obstacles = False,
    fail_rate = 0.0 )

#config=None

simEnv = "PyGame"

if simEnv == "PyGame":
    worldModel = MultiDroneEnv(config)
    #env = env
    
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
                
        observation, info  = worldModel.reset(seed=episode)         
        info         = worldModel.get_initial_state()
        
        drones = info["drones"]
        tasks = info["tasks"]
        quality_table =  info["quality_table"]
        
        done = {0 : False}
        truncations = {0 : False}
                        
        if algorithm == "Random":            
            planned_actions = utils.generate_random_tasks_all(drones, tasks, seed = episode) 
            single_random_alloc = True
            #print(planned_actions)
        
        if algorithm == "Greedy":
            agent = TessiAgent(num_drones=worldModel.n_agents, n_tasks=worldModel.n_tasks, max_dist=worldModel.max_coord, tessi_model = 1)               
        
        if algorithm == "Swarm-GAP":
            agent = SwarmGap(worldModel.agents_obj, worldModel.tasks, exchange_interval = 1)
        
        if algorithm == "CBBA":
            agent = CBBA(worldModel.agents_obj, worldModel.tasks, worldModel.max_coord)
        
        if algorithm == "TBTA" or algorithm == "TBTA2":
            # load policy as in your original code
            
            if algorithm == "TBTA":
                load_policy_name = 'policy_CustomNetReducedEval_TBTA_02_max30agents_timeRew_DroneEncodings_OWN_olyOwn_noAgemts.pth'            
            if algorithm == "TBTA2": 
                load_policy_name = 'policy_CustomNetReducedEval_TBTA_01_max30agents_timeRew_DroneEncodings.pth'            
            load_policy_path = os.path.join("dqn_Custom", load_policy_name)                    
            agent = _get_model(worldModel)
            saved_state = torch.load(load_policy_path )           
            agent.load_state_dict(saved_state)
            agent.eval()
            agent.set_eps(0.0)
    
        print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")   
        
        episode_reward = 0
        while not all(done.values()) and not all(truncations.values()):
                            
            actions = None
                        
            if algorithm == "Random":
                            
                if worldModel.time_steps % 1 == 0 and worldModel.time_steps >= 0:
                    
                    #if info['events'] == ["Reset_Allocation"]:
                        #print("New TAsks Alloc")                        
                    if True:
                        un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                        if un_taks_obj != []: 
                            
                            task = random.choice(un_taks_obj)
                            agent = worldModel.agent_selection
                            actions = {agent : task.task_id}
                    
                    else:    
                    
                    
                        un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()]                                         
                            
                        if un_taks_obj != []:
                            planned_actions = utils.generate_random_tasks_all(worldModel.get_live_agents() , un_taks_obj, seed = episode) 

                        if len(planned_actions) > 0:
                            
                            actions = {}                     
                            toDelete = [] 
                                                                                    
                            for agent_id, tasks in planned_actions.items():                                             
                                
                                if len(tasks) > 0:
                                    actions[agent_id] = planned_actions[agent_id].pop()
                                    #if single_random_alloc:                                    
                                    #break                      
                                else:
                                    toDelete.append(agent_id)
                        
                            for i in toDelete: 
                                del planned_actions[i]
                            #print(actions)                         
                            
            elif algorithm == "Swarm-GAP":
                
                if worldModel.time_steps % agent.exchange_interval == 0:  
                    
                    un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()]                   
                    if un_taks_obj != []:
                        actions = agent.process_token(worldModel.agents_obj, un_taks_obj)    
            
            elif algorithm == "Greedy":            
                                                                                                        
                if worldModel.time_steps % 1 == 0 :
                    
                    un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()]                                                             
                    
                    if un_taks_obj != []:
                        actions = agent.allocate_tasks(worldModel.get_live_agents(), un_taks_obj )
                    
                    
            elif algorithm == "CBBA":
                if worldModel.time_steps % 1 == 0 :
                    
                    un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                    
                    if un_taks_obj != []:
                        actions = agent.allocate_tasks( worldModel.get_live_agents(), un_taks_obj )                                
                        
            elif algorithm == "TBTA" or algorithm == "TBTA2":
                
                un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                if un_taks_obj != []:
                    agent_id = "agent" + str(worldModel.agent_selector._current_agent)                
                    obs_batch = Batch(obs=observation[agent_id], info=[{}])               
                    action = agent(obs_batch).act
                    actions = {agent_id : action[0]}
            
            elif algorithm == "CTBTA":
                
                un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                if un_taks_obj != []:
                    agent_id = "agent" + str(worldModel.agent_selector._current_agent)                
                    obs_batch = Batch(obs=observation[agent_id], info=[{}])               
                    action = agent(obs_batch).act
                    actions = {agent_id : action[0]}
            
            #if actions != {} and actions != None:
            #       print(actions)
            
            observation, reward, done, truncations, info = worldModel.step(actions)
            #print({'agent_id': worldModel.agent_selector.next()})            
            episode_reward += sum(reward.values())/worldModel.n_agents
            
            if worldModel.render_enabled:
                worldModel.render()
                        
            if all(done.values()):            
                metrics = info['metrics']
                metrics['Reward'] = episode_reward
                metrics["Algorithm"] = algorithm
                
                totalMetrics.append(metrics)
                                            
            if all(truncations.values()):                                
                metrics = info['metrics']
                metrics['Reward'] = episode_reward
                metrics["Algorithm"] = algorithm
                totalMetrics.append(metrics)

        total_reward[algorithm].append(episode_reward)         
                    
    end_time = time.time()
    execution_time = end_time - start_time
    print("\nExecution time ", algorithm, execution_time, "seconds")
    
    worldModel.close()

for alg in algorithms:
    print(f'Rew({alg}): {np.mean(total_reward[alg])}')
    print(f'Rew({alg}): Max: {max(total_reward[alg])}, Min: {min(total_reward[alg])}')
    #plt.hist(total_reward[alg], bins=100)
    #plt.show()


metricsDf = pd.DataFrame(totalMetrics)
#worldModel.plot_metrics(metricsDf, len(worldModel.agents), worldModel.n_tasks)
import seaborn as sns

#for algorithm in algorithms:
#    worldModel.plot_convergence(metricsDf[metricsDf.Algorithm == algorithm], len(worldModel.agents), len(worldModel.tasks), algorithm)

df = metricsDf#[metricsDf['Algorithm'] != 'Greedy']
#print(metricsDf.mean())
grouped = df.groupby('Algorithm', sort=False)
means = grouped.mean()
std_devs = grouped.std()

std_devs = std_devs / means.loc['Random']
means = means / means.loc['Random']        

# Calculate the number of algorithms and metrics
num_algorithms = len(grouped)
num_metrics = len(df.columns) - 1

palette = sns.color_palette("Set1",n_colors=num_algorithms)
        
# Create a single plot
fig, ax = plt.subplots(figsize=(10, 5))

# Define the bar width and the spacing between groups of bars
bar_width = 0.7 / num_algorithms
group_spacing = 1.2

max_val = 0
# Create a bar chart for each algorithm
for i, (algo, data) in enumerate(grouped):
    index = np.arange(num_metrics) * group_spacing + i * bar_width       
    ax.bar(index, means.loc[algo], bar_width, alpha=0.8, label=algo, yerr=std_devs.loc[algo], capsize=5, color=palette[i])
    if  max(means.loc[algo]) > max_val:
         max_val = max(means.loc[algo])

ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title(f'Task Allocation: ({6} drones, {10} tasks)')
ax.set_xticks(np.arange(num_metrics) * group_spacing + (bar_width * (num_algorithms - 1) / 2))
ax.set_xticklabels(list(df.columns)[:-1])
ax.legend()
ax.set_ylim(0, max_val*1.1)

plt.tight_layout()
plt.show()







