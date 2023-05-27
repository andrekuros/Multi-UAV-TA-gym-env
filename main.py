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
import torch.nn.functional as F

#from gym import spaces
#from godot_rl.core.godot_env import GodotEnv
#from godot_rl.core.utils import lod_to_dol

def softmax_stable(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

import MultiDroneEnvUtils as utils

algorithms = []
algorithms += ['Random']
#algorithms += ['Random2']
algorithms += ["Greedy"]
algorithms += ["Swarm-GAP"]
#algorithms += ["CBBA"]
algorithms +=  ["TBTA"]
algorithms +=  ["TBTA2"]
#algorithms +=  ["CTBTA"]

print(algorithms)
episodes = 30

config = utils.DroneEnvOptions(     
    render_speed = -1,
    max_time_steps = 500,
    action_mode= "TaskAssign",
    agents= {"F1" : 4,"R1" : 6},                 
    tasks= { "Att" : 4 , "Rec" : 16},
    random_init_pos = False,
    num_obstacles = 0,
    hidden_obstacles = False,
    fail_rate = 0.0 )

#config=None

simEnv = "PyGame"
#worldModel = MultiDroneEnv(config)
#if simEnv == "PyGame":
#    worldModel = MultiDroneEnv(config)
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
                
        # config = utils.DroneEnvOptions(     
        #             render_speed = -1,
        #             max_time_steps = 1000,
        #             action_mode= "TaskAssign",
        #             agents= {"F1":2 ,"F2":2, "R1":4, "R2":4 },
        #             tasks= { "Att" : 4 , "Rec" : 16},
        #             random_init_pos = True,
        #             num_obstacles = 0,
        #             hidden_obstacles = False,
        #             fail_rate = 0.0 )
        
        worldModel = MultiDroneEnv(config)

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
            rndGen = random.Random(episode*2)
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
                load_policy_name = 'policy_CustomNetReducedEval_TBTA_03_pre_process_New_REW_step500.pth'
                load_policy_path = os.path.join("dqn_Custom", load_policy_name)                    
                agent = _get_model(model="CustomNetReduced", env=worldModel)           
                

            if algorithm == "TBTA2": 
                load_policy_name = 'policy_CustomNetMultiHeadEval_TBTA_03_pre_process_New_REW_step500.pth'            
                load_policy_path = os.path.join("dqn_Custom", load_policy_name)                    
                agent = _get_model(model="CustomNetMultiHead", env=worldModel)
            
            saved_state = torch.load(load_policy_path )           
            agent.load_state_dict(saved_state)
            agent.eval()
            agent.set_eps(0.0)
    
        if algorithm == "CTBTA":
            
            agent = CBBA(worldModel.agents_obj, worldModel.tasks, worldModel.max_coord)
            load_policy_name = 'policy_CustomNetMultiHeadEval_TBTA_03_pre_process_New_REW_step500.pth'            
            load_policy_path = os.path.join("dqn_Custom", load_policy_name)                    
            agent2 = _get_model(model="CustomNetMultiHead", env=worldModel)
            
            saved_state = torch.load(load_policy_path )           
            agent2.load_state_dict(saved_state)
            agent2.eval()
            agent2.set_eps(0.0)
        
        print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")   
        

        episode_reward = 0
        while not all(done.values()) and not all(truncations.values()):
                            
            actions = None
                        
            if algorithm == "Random" or algorithm == "Random2":
                            
                if worldModel.time_steps % 1 == 0 and worldModel.time_steps >= 0:
                    
                    #if info['events'] == ["Reset_Allocation"]:
                        #print("New TAsks Alloc")                        
                    if algorithm == "Random":
                        un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                        
                        if un_taks_obj != []: 
                            
                            task = rndGen.choice(un_taks_obj)
                            #agent = rndGen.choice(worldModel.get_live_agents()).name
                            agent = worldModel.agent_selection


                            actions = {agent : task.task_id}
                    
                    elif algorithm == "Random2":

                        un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                        if un_taks_obj != []: 
                            
                            task = rndGen.choice(un_taks_obj)
                            agent = rndGen.choice(worldModel.get_live_agents()).name
                            #agent = worldModel.agent_selection

                            actions = {agent : task.task_id}

                    elif algorithm == "Random3":    
                    
                    
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
                        
                        drones = [worldModel.agents_obj[worldModel.agent_name_mapping[worldModel.agent_selection]]]
                        actions = agent.allocate_tasks(drones, un_taks_obj )
                    
                    
            elif algorithm == "CBBA":
                if worldModel.time_steps % 1 == 0 :
                    
                    un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                    
                    if un_taks_obj != []:
                        actions = agent.allocate_tasks( worldModel.get_live_agents(), un_taks_obj )                                
                        #print(actions)
                        
            elif algorithm == "TBTA" or algorithm == "TBTA2":
                
                un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                if un_taks_obj != []:
                    agent_id = "agent" + str(worldModel.agent_selector._current_agent)                
                    obs_batch = Batch(obs=observation[agent_id], info=[{}])               
                    action = agent(obs_batch).act
                    #print([task.type for task in un_taks_obj])
                    #print([agent.type for agent in worldModel.get_live_agents()], worldModel.time_steps)
                    action = np.argmax(action)
                    actions = {agent_id : action}
                    #print(actions)
            
            elif algorithm == "CTBTA":
                
                un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                
                if un_taks_obj != []:
                    Qs ={}
                    live_agents = worldModel.get_live_agents()
                    for uav in live_agents:                                       
                        obs_batch = Batch(obs=observation[uav.name], info=[{}]) 
                        #Qs[uav.name] =softmax_stable(agent2(obs_batch).act[0])
                        Qs[uav.name] =agent2(obs_batch).act[0]
                        #print(Qs)            
                    #print(Qs)
                    actions = agent.allocate_tasks(live_agents , un_taks_obj, Qs=Qs ) 
                    #print(actions)
                    
            
            #if actions != {} and actions != None:
            #       print(actions)
            
            observation, reward, done, truncations, info = worldModel.step(actions)
            #print({'agent_id': worldModel.agent_selector.next()})            
            episode_reward += sum(reward.values())/worldModel.n_agents
            
            if worldModel.render_enabled:
                worldModel.render()
                        
            if all(done.values()):            
                metrics = info['metrics']
                metrics['S_Reward'] = episode_reward
                metrics["Algorithm"] = algorithm
                
                totalMetrics.append(metrics)
                                            
            if all(truncations.values()):                                
                metrics = info['metrics']
                metrics['S_Reward'] = episode_reward
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

#%%%

metricsDf = pd.DataFrame(totalMetrics)
# Salvar o DataFrame em um arquivo CSV
metricsDf.to_csv('Resultados_Final_01_07ob_Trein100Est_Dyn07.csv', index=False)

#worldModel.plot_metrics(metricsDf, len(worldModel.agents), worldModel.n_tasks)
import seaborn as sns
import matplotlib as mpl

# Define o estilo seaborn como 'whitegrid'
sns.set_style("whitegrid")

df = metricsDf.drop(['F_load', 'F_Reward'], axis=1)#[metricsDf['Algorithm'] != 'Greedy']
#print(metricsDf.mean())
grouped = df.groupby('Algorithm', sort=False)
means = grouped.mean()
std_devs = grouped.std()

std_devs = std_devs / means.loc['Random']
means = means / means.loc['Random']        
std_devs
means

# Calculate the number of algorithms and metrics
num_algorithms = len(grouped)
num_metrics = len(df.columns) - 1

palette = sns.color_palette("Set1",n_colors=num_algorithms)
        
# Create a single plot
fig, ax = plt.subplots(figsize=(8, 6))

# Define the bar width and the spacing between groups of bars
bar_width = 0.6 / num_algorithms
group_spacing = 0.9

max_val = 0
# Create a bar chart for each algorithm
for i, (algo, data) in enumerate(grouped):
    index = np.arange(num_metrics) * group_spacing + i * bar_width       
    ax.bar(index, means.loc[algo], bar_width, alpha=0.95, label=algo, yerr=std_devs.loc[algo], capsize=6, color=palette[i], linewidth=0.0)
    if  max(means.loc[algo]) > max_val:
         max_val = max(means.loc[algo])

ax.set_xlabel('Metrics',fontsize=12)
ax.set_ylabel('Values',fontsize=12)
ax.set_title(f'Task Allocation: ({10} drones, {20} tasks)  |  Dynamic: fails(0.7)',fontsize=14)

ax.set_xticks(np.arange(num_metrics) * group_spacing + (bar_width * (num_algorithms - 1) / 2))
ax.set_xticklabels(list(df.columns)[:-1], fontsize=12)

ax.set_yticks(np.arange(0,max_val*1.35, 0.20))


ax.legend(loc='upper left', fontsize=13)
ax.set_ylim(0, max_val*1.35)

plt.tight_layout()
plt.show()

#%%
for algorithm in algorithms:
    worldModel.plot_convergence(metricsDf[metricsDf.Algorithm == algorithm]/ means.loc['Random'], len(worldModel.agents), len(worldModel.tasks), algorithm)





