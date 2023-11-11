#%%
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#Task Allocation Algorithms
from mUAV_TA.DroneEnv import MAX_INT, MultiUAVEnv
from mUAV_TA.DroneEnv import env
from TaskAllocation.BehaviourBased.swarm_gap import SwarmGap
from TaskAllocation.MarketBased.CBBA import CBBA
from TaskAllocation.BehaviourBased.Greedy import GreedyAgent

import subprocess


import mUAV_TA.MultiDroneEnvUtils as utils

#RL Models/Policies
from TaskAllocation.RL_Policies.Tianshou_Policy import _get_model
import torch
from tianshou.data import Batch
import torch.nn.functional as F

#from gym import spaces
#from godot_rl.core.godot_env import GodotEnv
#from godot_rl.core.utils import lod_to_dol

def softmax_stable(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

algorithms = []
# algorithms += ['Random']
# # # #algorithms += ['Random2']
# algorithms += ["Greedy"]
# algorithms += ["Swarm-GAP"]
# algorithms += ["CBBA"]
algorithms +=  ["TBTA"]
#algorithms +=  ["TBTA2"]
#algorithms +=  ["CTBTA"]

episodes = 10
fail_rate = 0.0
render_speed = -1
resolution_increase = 1
print(algorithms)

same_policy = False
   
scal_analysis = "None"
cases = []

if scal_analysis == "Tasks":
    
    for i in range(1,29):
        case = {'case' : i, 'F1':2, 'F2': 2, "R1" : 3, 'R2' : 3}
        case['Att'] = 2 + int(i / 5)
        case['Rec'] = i 
        cases.append(case)
elif scal_analysis == "Agents": 

    for i in range(1, 13):
        case = {'case' : i, 'F1':int(i/2), "R1" : i, 'F2': 0, 'R2' : 0}
        case['Att'] = 6
        case['Rec'] = 24
        cases.append(case)
else:
    cases =  [{'case' : 0, 'Hold': 4, 'Att': 4, 'Rec' : 16, 'F1':1, 'F2': 0, "R1" : 6 }]
    cases =  [{'case' : 0, 'Hold': 0, 'Att': 10, 'Rec' : 0,  'F1':0, 'F2': 1, "R1" : 0, "R2" : 0 }]

caseResults = []
totalMetrics = []
total_reward = {}
total_F_reward = {}
time_reward = {}
distance_reward = {}
quality_reward = {}
load_reward = {}
losses = {}
kills = {}

process_time = {}
process_times = {}

expName = f'UCF_1_ep{episodes}_fail{fail_rate}_scal_{scal_analysis}' 

caseResults = []
#time.sleep(1)


# Path to the PyQt script
pyqt_script_path = "PyQtServer.py"

print("Next")

# Run the PyQt script
if render_speed != -1:
    subprocess.Popen(["python", pyqt_script_path])

for c_idx,case in enumerate(cases):
    
    #if c_idx +1 != 28:
        #print(c_idx)
    #    continue

    print("\nStarting Case: ", case)
    
    for algorithm in algorithms:
        
        config = utils.agentEnvOptions(     
            render_speed = render_speed,
            simulation_frame_rate = 0.01 * resolution_increase, #Std 0.02
            max_time_steps = 150 * resolution_increase,
            action_mode= "TaskAssign",
            agents= {"F1" : case['F1'], "F2" : case['F2'], "R1" : case['R1'], "R2" : case['R2']},                 
            tasks= { "Att" : case['Att'], "Rec" : case['Rec'],  "Hold" : case['Hold']},
            random_init_pos = False,
            num_obstacles = 0,
            hidden_obstacles = False,
            multiple_tasks_per_agent = False,
            multiple_agents_per_task = True,
            fail_rate = fail_rate,
            threats_list = [],#[("T1", 4), ("T2" , 2)],
            fixed_seed = -1,
            info = algorithm  )                

        worldModel = MultiUAVEnv(config)       
        n_tasks = worldModel.n_tasks 
        n_agents = worldModel.n_agents
        print("\nStarting Algorithm:", algorithm)
        
        total_reward[algorithm] = [] 
        total_F_reward[algorithm] = []
        
        time_reward[algorithm] = [] 
        distance_reward[algorithm] = []
        quality_reward[algorithm] = [] 
        load_reward[algorithm] = [] 
        losses[algorithm] = [] 
        kills[algorithm] = [] 
        
        process_time[algorithm] = []  
        process_times[algorithm]  = []  
            
        for episode in range(episodes):
           
            episode_seed = episode#1622124873183273465            

            #print("Start_Episode", episode )
            rndGen = random.Random(episode_seed)
            
            observation, info  = worldModel.reset(seed=episode_seed)#episode+14)                 
            info         = worldModel.get_initial_state()
            
            agents = info["agents"]
            tasks = info["tasks"]
            quality_table =  info["quality_table"]
           
            done = {0 : False}
            truncations = {0 : False}
                            
            if algorithm == "Random":                            
                single_random_alloc = True

            if algorithm == "Greedy":                            
                policy = GreedyAgent()
                worldModel.multiple_tasks_per_agent = True                
                worldModel.multiple_agents_per_task = True
                                                                                    
            if algorithm == "Swarm-GAP":
                policy = SwarmGap(worldModel.agents_obj, worldModel.tasks, exchange_interval = 1)
                worldModel.multiple_tasks_per_agent = True                
                worldModel.multiple_agents_per_task = True
            
            if algorithm == "CBBA":
                policy = CBBA(worldModel.agents_obj, worldModel.tasks, worldModel.max_coord, seed = rndGen.randint(0, MAX_INT))
                worldModel.multiple_tasks_per_agent = True                
                worldModel.multiple_agents_per_task = True

            
            if algorithm == "TBTA" or algorithm == "TBTA2":
                # load policy as in your original code
                worldModel.multiple_tasks_per_agent = False                
                worldModel.multiple_agents_per_task = True
                
                if algorithm == "TBTA":
                    # load_policy_name = 'policy_CustomNetMultiHead_Eval_TBTA_4R120RT_02.pth'
                    load_policy_name = 'policy_CustomNetMultiHead_TBTA_NOV06_Emb128.pth'
                    load_policy_path = os.path.join("dqn_Custom", load_policy_name) 
                    # load_policy_name = 'PPO_CustomNetMultiHead_Eval_TBTA_OCT01.pth'
                    # load_policy_path = os.path.join("ppo_Custom", load_policy_name) 
                    
                    # load_policy_name2 = 'policy_CustomNetMultiHead_Eval_TBTA_OCT01F.pth'
                       
                    # load_policy_path2 = os.path.join("dqn_Custom", load_policy_name2)                
                    policy = _get_model(model="CustomNetMultiHead", env=worldModel, seed = episode_seed)   
                    # policyF = _get_model(model="CustomNetMultiHead", env=worldModel, seed = episode_seed)         
                    

                if algorithm == "TBTA2": 
                    load_policy_name = 'policy_CustomNetMultiHead_Eval_TBTA_02_simplified_UCF1.pth'            
                    load_policy_path = os.path.join("dqn_Custom", load_policy_name)                    
                    policy = _get_model(model="CustomNetMultiHead", env=worldModel,  seed = episode_seed)
                
                saved_state = torch.load(load_policy_path )           
                policy.load_state_dict(saved_state)
                policy.eval()
                policy.set_eps(0.0)
        
            if algorithm == "CTBTA":

                policy = CBBA(worldModel.agents_obj, worldModel.tasks, worldModel.max_coord, seed = rndGen.randint(0, MAX_INT))
                load_policy_name = 'policy_CustomNetMultiHead_Eval_TBTA_02_simplified_UCF_mask01_seed0All.pth'            
                load_policy_path = os.path.join("dqn_Custom", load_policy_name)                             
                policy2 = _get_model(model="CustomNetMultiHead", env=worldModel, seed = episode_seed)
    
                saved_state = torch.load(load_policy_path )           
                policy2.load_state_dict(saved_state)
                policy2.eval()
                policy2.set_eps(0.0)
            
            print ("."  if (episode+1)%10 != 0 else str(episode+1), end="")   
            
            episode_reward = 0
            episode_process_time = []           

            while not all(done.values()) and not all(truncations.values()):
                                
                actions = None
                            
                if algorithm == "Random" or algorithm == "Random2":
                                
                    if worldModel.time_steps % 1 == 0 and worldModel.time_steps >= 0:
                        
                        #if info['events'] == ["Reset_Allocation"]:
                            #print("New TAsks Alloc")                        
                        if algorithm == "Random":
                                                                                    
                            un_taks_obj = [task for task in worldModel.tasks if (task.status != 2 and task.type != "Hold")]
                            # print(" Len:" , len([(task.id, task.type) for task in un_taks_obj]))

                            if un_taks_obj != []: 
                                                                
                                start_time = time.time()
                               
                                task = rndGen.choice(un_taks_obj)      
                                
                                agent =  worldModel.agent_by_name[worldModel.current_agent]#rndGen.choice(worldModel.get_live_agents())                                                                
                                
                                if agent.tasks == [worldModel.task_idle]:
                                                                        
                                    if agent.state != 99:                                
                                        actions = {agent.name : worldModel.last_tasks_info.index(task)}                                                                
                                        #print(actions, task.type)
                                    #else:
                                    #   print(agent.state)

                                    end_time = time.time()
                                    episode_process_time.append(end_time - start_time)
                        
                        elif algorithm == "Random2":

                            un_taks_obj = [task for task in worldModel.tasks if (task.status != 2 and task.type != "Hold")]
                            #print(" Len:" , len([(task.id, task.type) for task in un_taks_obj]))

                            if un_taks_obj != []: 
                                
                                
                                start_time = time.time()
                               
                                task = rndGen.choice(un_taks_obj)                                
                                agent =  worldModel.agent_by_name[worldModel.current_agent]#rndGen.choice(worldModel.get_live_agents())                                                                
                                                                                                                                                                                                 
                                actions = {agent.name : worldModel.last_tasks_info.index(task)}                                                                
                                
                                end_time = time.time()
                                episode_process_time.append(end_time - start_time)
                        
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
                    
                    if worldModel.time_steps % policy.exchange_interval == 0:  
                        
                        #un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()]    
                        un_taks_obj = [task for task in worldModel.tasks 
                                       if task.id != 0 and
                                       task.allocatedReqs[task.typeIdx] < task.currentReqs[task.typeIdx] and 
                                       task.status != 2]               

                        #print([t.allocatedReqs for t in worldModel.tasks])
 
                        actions = {}
                        if un_taks_obj != []:
                            start_time = time.time()
                            
                            result_actions = policy.process_token(worldModel.agents_obj, un_taks_obj) 
                            if result_actions is not None:                            
                                for action in result_actions:
                                    actions[action[0]] = [worldModel.last_tasks_info.index(act) for act in action[1]]
                                



                            end_time = time.time()
                            episode_process_time.append(end_time - start_time)
                
                elif algorithm == "Greedy":            
                                                                                                            
                    if worldModel.time_steps % 1 == 0 :
                         
                        actions = {}
                        un_taks_obj = [task for task in worldModel.tasks 
                                       if task.id != 0 and
                                           task.allocatedReqs[task.typeIdx] < task.currentReqs[task.typeIdx] and 
                                           task.status != 2]                                                    
                        
                        if un_taks_obj != []:
                                                        
                            act = policy.allocate_tasks(worldModel.agents_obj, un_taks_obj )                                                        
                            actions[act[0][0]] = [worldModel.last_tasks_info.index(act[0][1])]
                            
               
                elif algorithm == "CBBA":
                    if worldModel.time_steps % 1 == 0 :
                        
                        #un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                        un_taks_obj = [task for task in worldModel.tasks 
                                       if task.id != 0 and
                                       task.allocatedReqs[task.typeIdx] < task.currentReqs[task.typeIdx] and
                                       task.status != 2]            
                        actions = {}
                        if len(worldModel.get_live_agents()) > 0:
                            
                            #print(worldModel.get_live_agents())
                            if un_taks_obj != []:
                                start_time = time.time()   
                                result_actions = policy.allocate_tasks( worldModel.get_live_agents(), un_taks_obj )                                                            
                                #print(result_actions)
                                if result_actions is not None:                            
                                    for action in result_actions:
                                        actions[action[0]] = [worldModel.last_tasks_info.index(act) for act in action[1]]
                                end_time = time.time()
                                episode_process_time.append(end_time - start_time)
                                # print(actions)
                                
                elif algorithm == "TBTA" or algorithm == "TBTA2":
                    
                    #un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                    un_taks_obj = worldModel.tasks 


                    if un_taks_obj != []:
                        
                        start_time = time.time() 
                        actions = {}  
                        parallelDecision = False

                        if parallelDecision:
                        
                            for agent in worldModel.agents_obj:
                                                                                
                                agent_id = f'{worldModel.agents_obj[worldModel.agent_selector._current_agent].name}'                                            
                                obs_batch = Batch(obs=[observation[agent_id]], info=[{}])                                               
                                action = policy(obs_batch).act
                                action = np.argmax(action)                                
                                actions[agent_id] = action                                                                
                                
                            end_time = time.time()
                            episode_process_time.append(end_time - start_time)
                            
                            
                        else:
                            agent_id = f'{worldModel.agents_obj[worldModel.agent_selector._current_agent].name}'                                            
                            obs_batch = Batch(obs=[observation[agent_id]], info=[{}])               
                            #print(obs_batch)
                            if worldModel.agents_obj[worldModel.agent_selector._current_agent].type != "AA":                            
                                action = policy(obs_batch).act
                                action = np.argmax(action)                            
                                actions[agent_id] = action                            
                                end_time = time.time()
                                episode_process_time.append(end_time - start_time)
                            
                            #print(actions)
                
                elif algorithm == "CTBTA":
                    
                    un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                    
                    best_task = -1
                    best_task_id = 0
                    best_uav = 'agent0'
                    
                    #print(worldModel.allocation_table)
                    if un_taks_obj != []:
                        Qs ={}
                        live_agents = worldModel.get_live_agents()
                        for uav in live_agents:                                       
                            obs_batch = Batch(obs=[observation[uav.name]], info=[{}]) 
                            #Qs[uav.name] =softmax_stable(agent2(obs_batch).act[0])
                            #print(obs_batch)
                            Qs[uav.name] = policy2(obs_batch).act[0]
                            idx_max = np.argmax(Qs[uav.name])
                            print(idx_max, len(un_taks_obj), Qs[uav.name] )                            
                            best_task_F = Qs[uav.name][idx_max]

                            if best_task_F > best_task:
                                best_task = best_task_F
                                best_task_id = observation[uav.name]["tasks_info"][idx_max]["id"]
                                best_uav = uav.name                                

                            #print(observation[uav.name]["tasks_info"])
                            #print(f'{uav.name} Max_tasK: {observation[uav.name]["tasks_info"][idx_max]["id"]}')                        
                   
                        #actions = policy.allocate_tasks(live_agents , un_taks_obj, Qs=Qs ) 
                        actions = {best_uav : best_task_id}
                        print(actions)
                        print(actions)

                #if actions != {} and actions != None:
                #       print(actions)
                
                observation, reward, done, truncations, info = worldModel.step(actions)
                                        
                episode_reward += sum(reward.values())/worldModel.n_agents
                #print("Step:",sum(reward.values())/worldModel.n_agents)
                #print("Total:",episode_reward)

                if worldModel.render_enabled:
                    worldModel.render()
                    
                if all(done.values()) or all(truncations.values()):            
                    metrics = info['metrics']
                    metrics['S_Reward'] = episode_reward
                    metrics["Algorithm"] = algorithm 
                    metrics['seed']= episode_seed
                    totalMetrics.append(metrics)

                    #worldModel.exporter.export_to_acmi("sim1.acmi")
                    #print(episode_reward)
                                                        
            total_reward[algorithm].append(episode_reward)
            total_F_reward[algorithm].append(totalMetrics[-1]['F_Reward'])
            time_reward[algorithm].append(totalMetrics[-1]['F_time'])
            distance_reward[algorithm].append(totalMetrics[-1]['F_distance'])
            quality_reward[algorithm].append(totalMetrics[-1]['F_quality'])
            losses[algorithm].append(totalMetrics[-1]['Losses'])
            kills[algorithm].append(totalMetrics[-1]['Kills'])

#           load_reward[algorithm].append(totalMetrics[-1]['F_load'])
                 
            process_time[algorithm].append(np.mean(episode_process_time))         
            process_times[algorithm].append(len(episode_process_time))          
                                
        print("\nExecution time ", algorithm, sum(process_time[algorithm]), "seconds")        
        worldModel.close()


    for alg in algorithms:
        print(f'Case {case["case"]} -> {alg}-> S_REW: {np.mean(total_reward[alg]):.2f} ( {min(total_reward[alg]):.2f} / {max(total_reward[alg]):.2f})')         
        print(f'Case {case["case"]} -> {alg}-> R_REW: {np.mean(total_F_reward[alg]):.2f} ( {min(total_F_reward[alg]):.2f} / {max(total_F_reward[alg]):.2f})') 
        #plt.hist(total_reward[alg], bins=100)
        #plt.show()
        caseResults.append({'case' : n_agents ,'algorithm' : alg, 'mean_S_reward' : total_reward[alg], 
                            'mean_process_time' : process_time[alg],
                            'process_runs' : process_times[alg],
                            'mean_R_reward' : total_F_reward[alg],
                            'time_reward' : time_reward[alg],
                            'distance_reward' : distance_reward[alg],
                            'quality_reward' : quality_reward[alg],
                            'load_reward' : load_reward[alg],
                            'losses' : losses[alg],
                            'kills'  : kills[alg],
                            'n_Tasks' : n_tasks,'n_Agents' : n_agents, 
                            'caseData' : case})


#print(caseResults)

casesDf = pd.DataFrame(caseResults)
casesDf.to_csv(f'Cases_{expName}.csv', index=False)

metricsDf = pd.DataFrame(totalMetrics)
metricsDf.to_csv(f'Resultados_{expName}.csv', index=False)
print(f'\nResults Saved to:\nResultados_{expName}.csv' )


#%%
#metricsDf.to_csv('Resultado_Final_Qualify_Principal01_CaseMegaDist.csv', index=False)
import pandas as pd
import seaborn as sns
import matplotlib as mpl

fail_rate = 0.0
metricsDf = pd.read_csv('Resultados_UCF_1_ep50_fail0.0_scal_None.csv')

#worldModel.plot_metrics(metricsDf, len(worldModel.agents), worldModel.n_tasks)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Define o estilo seaborn como 'whitegrid'
sns.set_style("whitegrid")

df = metricsDf#.drop(['F_load', 'S_Reward'], axis=1)#[metricsDf['Algorithm'] != 'Greedy']
#df = df[df.S_Reward > 100]
# df['S_Reward'] = df['S_Reward'] / 500
# df['F_Reward'] = df['F_Reward'] / 500
# df['F_time'] = df['F_time'] / 2
# df['F_distance'] = df['F_distance'] * 5
# df['F_load'] = df['F_load'] * 200

df = df[df.Algorithm != 'Greedy']

grouped = df.groupby('Algorithm', sort=False)
means = grouped.mean()
std_devs = grouped.std()

means_rnd = means.loc['Random']   
std_devs = std_devs / means_rnd
means = means / means_rnd 
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
    # ax.bar(index, means.loc[algo], bar_width, alpha=0.95, label=algo, capsize=6, color=palette[i], linewidth=0.0)
    if  max(means.loc[algo]) > max_val:
         max_val = max(means.loc[algo])

ax.set_xlabel('Metrics',fontsize=12)
ax.set_ylabel('Values',fontsize=12)
ax.set_title(f'Task Allocation: (10 agents, [2-30] tasks)  |  Dynamic: fails({fail_rate})',fontsize=14)

ax.set_xticks(np.arange(num_metrics) * group_spacing + (bar_width * (num_algorithms - 1) / 2))
ax.set_xticklabels(list(df.columns)[:-1], fontsize=12)

ax.set_yticks(np.arange(0,max_val*1.25, 0.20))


ax.legend(loc='upper left', fontsize=13)
ax.set_ylim(0, max_val*1.25)

plt.tight_layout()
plt.show()
#%%%
import seaborn as sns
import matplotlib.pyplot as plt

# Define seaborn style as 'whitegrid'
sns.set_style("whitegrid")

df = metricsDf
df = df[df.Algorithm != 'Greedy']

df = metricsDf.drop(['F_load', 'F_Reward'], axis=1)#[metricsDf['Algorithm'] != 'Greedy']
# Normalize the data by the 'Random' algorithm
random_means = df[df['Algorithm'] == 'Random'].mean()
df_normalized = df.loc[:, df.columns != 'Algorithm'].div(random_means)
df_normalized['Algorithm'] = df['Algorithm']

# Melt the DataFrame to make it suitable for boxplot
melted_df = df_normalized.melt(id_vars='Algorithm')

# Create a boxplot for each algorithm for each metric
plt.figure(figsize=(12, 8))
sns.boxplot(x='variable', y='value', hue='Algorithm', data=melted_df, palette='Set1')

plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Normalized Values', fontsize=12)
plt.title(f'Task Allocation: ([2-30] agents, 30 tasks)  |  Dynamic: fails({fail_rate})', fontsize=14)

plt.legend(loc='upper left', fontsize=13)
plt.tight_layout()
plt.show()


#%%%
#palette = sns.color_palette("Set1",n_colors=num_algorithms)
        
# Create a single plot
fig, ax = plt.subplots(figsize=(6, 3.2))


dfC = pd.read_csv('Cases_UCF_1_ep30_fail0.0_scal_Tasks.csv')
#dfC = dfC[dfC['case'] <= 29]
eval = 'n_Tasks'
dfC['total_process_time'] = dfC['mean_process_time'] * dfC['process_runs']
dfC['mean_F_reward'] =  dfC['mean_R_reward'] 
sns.lineplot(x=eval, y='mean_process_time', hue='algorithm', data=dfC)
plt.title('Process Time for Each Algorithm')

#%%
fig, ax = plt.subplots(figsize=(6, 3.5))

sns.lineplot(x=eval, y='total_process_time', hue='algorithm', data=dfC)
plt.title('Process Time for Each Algorithm')

#%%
fig, ax = plt.subplots(figsize=(6, 3))
dfC = pd.read_csv('Cases_UCF_1_ep30_fail0.0_scal_Tasks.csv')
#dfC = dfC.groupby(eval).mean().reset_index()
# Escolha o algoritmo de referência para normalização
algorithm_reference = 'Random'
MetricRef = 'mean_R_reward'
dfC= dfC[dfC.n_Agents >= 3]

# Filtra o DataFrame para obter apenas as linhas com o algoritmo de referência
df_reference = dfC[dfC['algorithm'] == algorithm_reference]

# Realiza a mesclagem (merge) do DataFrame original com os valores de referência
dfC_merged = dfC.merge(df_reference[[eval, MetricRef]], on=eval, suffixes=('', '_reference'))

# Calcula a coluna com os valores normalizados
dfC_merged['normalized_'+ MetricRef] = dfC_merged[MetricRef] / dfC_merged[MetricRef + '_reference']

# Plota o line plot com os valores normalizados
sns.lineplot(x=eval, y='normalized_'+ MetricRef, hue='algorithm', data=dfC_merged)
#plt.title('Normalized Process Time for Each Algorithm')
ax.set_ylim(0.0, 2.0)
            
#sns.lineplot(x=eval, y='normalized_mean_F_reward', hue='algorithm', data=dfC)
plt.title('R_Reward for Each Algorithm')


#%%

def plot_convergence(df, n_agents, n_tasks, algorithm):
        
        cumulative_means = df.expanding().mean() #/ df.mean()
    
        palette = sns.color_palette("Set1",n_colors=len(df)-1)
              
        fig, ax = plt.subplots()
        auxDf = cumulative_means.reset_index()
        
        for i, metric in enumerate(auxDf.columns[1:]):
            ax.plot(auxDf[metric], label=metric, color = palette[i])
    
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Cummulative Means')
        ax.set_title(f'Convergence {algorithm} : ({n_agents} uavs, {n_tasks} tasks)')
        ax.legend()
        #ax.set_ylim(0.5, 1.5)
            
        plt.tight_layout()
        plt.show()

for algorithm in algorithms:
    plot_convergence(metricsDf[metricsDf.Algorithm == algorithm]/ means_rnd, len(worldModel.agents), len(worldModel.tasks), algorithm)

#%%%


fig, ax = plt.subplots(figsize=(7, 5))
ia = 1

def plot_convergence(i, df, n_agents, n_tasks, algorithm):
        
        cumulative_means = df.expanding().mean() #/ df.mean()
    
        palette = sns.color_palette("Set1",n_colors=5)
              
        
        auxDf = cumulative_means.reset_index()
        
        for b, metric in enumerate(auxDf.columns[1:]):
            ax.plot(auxDf[metric], label=metric, color = palette[i])
    
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Cummulative Means')
        ax.set_title(f'Convergence {algorithm} : ({n_agents} uavs, {n_tasks} tasks) - Dynamic: Fail(0.7)')
        ax.legend(['Random','Swarm-GAP','CBBA','TBTA'])
        ax.set_ylim(0.0, 1.2)
    
        plt.tight_layout()
        


for i,algorithm in enumerate(algorithms):
    plot_convergence(i, metricsDf[metricsDf.Algorithm == algorithm].F_Reward, len(worldModel.agents), len(worldModel.tasks), algorithm)
plt.show()