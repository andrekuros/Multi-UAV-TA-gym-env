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

def run_case_algorithm(case_item, algorithm, episodes, resolution_increase, fail_rate, render_speed, same_policy, scal_analysis):
    import numpy as np
    import random
    import time
    import os
    import torch
    import mUAV_TA.MultiDroneEnvUtils as utils
    from mUAV_TA.DroneEnv import MAX_INT, MultiUAVEnv
    from TaskAllocation.BehaviourBased.swarm_gap import SwarmGap
    from TaskAllocation.MarketBased.CBBA import CBBA
    from TaskAllocation.BehaviourBased.Greedy import GreedyAgent
    from TaskAllocation.RL_Policies.Tianshou_Policy import _get_model
    from tianshou.data import Batch
    
    total_reward = [] 
    total_F_reward = []
    time_reward = [] 
    distance_reward = []
    quality_reward = [] 
    losses = [] 
    kills = [] 
    process_time = []  
    process_times = []  
    
    config = utils.agentEnvOptions(     
        render_speed = render_speed,
        simulation_frame_rate = 0.01 * resolution_increase,
        max_time_steps = 150 * resolution_increase,
        action_mode= "TaskAssign",
        agents= {"F1" : case_item.get('F1', 0), "F2" : case_item.get('F2', 0), "R1" : case_item.get('R1', 0), "R2" : case_item.get('R2', 0)},                 
        tasks= { "Att" : case_item.get('Att', 0), "Rec" : case_item.get('Rec', 0),  "Hold" : case_item.get('Hold', 0)},
        random_init_pos = False,
        num_obstacles = 0,
        hidden_obstacles = False,
        multiple_tasks_per_agent = False,
        multiple_agents_per_task = True,
        fail_rate = fail_rate,
        threats_list = [],
        fixed_seed = -1,
        info = algorithm  )                

    worldModel = MultiUAVEnv(config)       
    n_tasks = worldModel.n_tasks 
    n_agents = worldModel.n_agents
    
    policy = None
    policy2 = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for episode in range(episodes):
        episode_seed = episode
        rndGen = random.Random(episode_seed)
        
        observation, info = worldModel.reset(seed=episode_seed)                 
        info = worldModel.get_initial_state()
        
        done = {0 : False}
        truncations = {0 : False}
        
        if episode == 0:
            if algorithm == "Greedy":                            
                policy = GreedyAgent()
            elif algorithm == "Swarm-GAP":
                policy = SwarmGap(worldModel.agents_obj, worldModel.tasks, exchange_interval = 1)
            elif algorithm == "CBBA":
                policy = CBBA(worldModel.agents_obj, worldModel.tasks, worldModel.max_coord, seed = rndGen.randint(0, 1000000))
            elif algorithm in ["TBTA", "TBTA2", "TBTA-PPO"]:
                if algorithm == "TBTA":
                    load_policy_name = 'policy_CustomNetMultiHead_TBTA_NOV06_Emb128.pth'
                    load_policy_path = os.path.join("dqn_Custom", load_policy_name) 
                    policy = _get_model(model="CustomNetMultiHead", env=worldModel, seed = episode_seed, algorithm="DQN")   
                elif algorithm == "TBTA2": 
                    load_policy_name = 'policy_CustomNetMultiHead_Eval_TBTA_02_simplified_UCF1.pth'            
                    load_policy_path = os.path.join("dqn_Custom", load_policy_name)                    
                    policy = _get_model(model="CustomNetMultiHead", env=worldModel, seed = episode_seed, algorithm="DQN")
                elif algorithm == "TBTA-PPO":
                    load_policy_name = 'PPO_CustomNetMultiHead_PPO_OCT01_2.pth'            
                    load_policy_path = os.path.join("ppo_Custom", load_policy_name)                    
                    policy = _get_model(model="CustomNetMultiHead", env=worldModel, seed = episode_seed, algorithm="PPO")
                
                saved_state = torch.load(load_policy_path, map_location=device)           
                policy.load_state_dict(saved_state)
                policy.eval()
                if algorithm != "TBTA-PPO":
                    policy.set_eps(0.0)
            elif algorithm == "CTBTA":
                policy = CBBA(worldModel.agents_obj, worldModel.tasks, worldModel.max_coord, seed = rndGen.randint(0, MAX_INT))
                load_policy_name = 'policy_CustomNetMultiHead_Eval_TBTA_02_simplified_UCF_mask01_seed0All.pth'            
                load_policy_path = os.path.join("dqn_Custom", load_policy_name)                             
                policy2 = _get_model(model="CustomNetMultiHead", env=worldModel, seed = episode_seed)
                saved_state = torch.load(load_policy_path, map_location=device)           
                policy2.load_state_dict(saved_state)
                policy2.eval()
                policy2.set_eps(0.0)

        if algorithm == "Greedy":
            worldModel.multiple_tasks_per_agent = True                
            worldModel.multiple_agents_per_task = True
        elif algorithm == "Swarm-GAP":
            worldModel.multiple_tasks_per_agent = True                
            worldModel.multiple_agents_per_task = True
        elif algorithm == "CBBA":
            worldModel.multiple_tasks_per_agent = True                
            worldModel.multiple_agents_per_task = True
        elif algorithm in ["TBTA", "TBTA2", "TBTA-PPO"]:
            worldModel.multiple_tasks_per_agent = False                
            worldModel.multiple_agents_per_task = True

        episode_reward = 0
        episode_process_time = []           
        latest_metrics = None

        while not all(done.values()) and not all(truncations.values()):
            actions = {}
            
            if algorithm == "Random" or algorithm == "Random2":
                if worldModel.time_steps % 1 == 0 and worldModel.time_steps >= 0:
                    if algorithm == "Random":
                        un_taks_obj = [task for task in worldModel.tasks if (task.status != 2 and task.type != "Hold")]
                        if un_taks_obj != []: 
                            start_time = time.time()
                            task = rndGen.choice(un_taks_obj)      
                            agent = worldModel.agent_by_name[worldModel.current_agent]
                            if agent.tasks == [worldModel.task_idle]:
                                if agent.state != 99:                                
                                    actions = {agent.name : worldModel.last_tasks_info.index(task)}                                                                
                                end_time = time.time()
                                episode_process_time.append(end_time - start_time)
                    elif algorithm == "Random2":
                        un_taks_obj = [task for task in worldModel.tasks if (task.status != 2 and task.type != "Hold")]
                        if un_taks_obj != []: 
                            start_time = time.time()
                            task = rndGen.choice(un_taks_obj)                                
                            agent = worldModel.agent_by_name[worldModel.current_agent]
                            actions = {agent.name : worldModel.last_tasks_info.index(task)}                                                                
                            end_time = time.time()
                            episode_process_time.append(end_time - start_time)
            
            elif algorithm == "Swarm-GAP":
                if worldModel.time_steps % policy.exchange_interval == 0:  
                    un_taks_obj = [task for task in worldModel.tasks 
                                   if task.id != 0 and
                                   task.allocatedReqs[task.typeIdx] < task.currentReqs[task.typeIdx] and 
                                   task.status != 2]               
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
                if worldModel.time_steps % 1 == 0:
                    actions = {}
                    un_taks_obj = [task for task in worldModel.tasks 
                                   if task.id != 0 and
                                       task.allocatedReqs[task.typeIdx] < task.currentReqs[task.typeIdx] and 
                                       task.status != 2]                                                    
                    if un_taks_obj != []:
                        act = policy.allocate_tasks(worldModel.agents_obj, un_taks_obj)
                        actions[act[0][0]] = [worldModel.last_tasks_info.index(act[0][1])]
            
            elif algorithm == "CBBA":
                if worldModel.time_steps % 1 == 0:
                    un_taks_obj = [task for task in worldModel.tasks 
                                   if task.id != 0 and
                                   task.allocatedReqs[task.typeIdx] < task.currentReqs[task.typeIdx] and
                                   task.status != 2]            
                    actions = {}
                    if len(worldModel.get_live_agents()) > 0:
                        if un_taks_obj != []:
                            start_time = time.time()   
                            result_actions = policy.allocate_tasks(worldModel.get_live_agents(), un_taks_obj)                                                            
                            if result_actions is not None:                            
                                for action in result_actions:
                                    actions[action[0]] = [worldModel.last_tasks_info.index(act) for act in action[1]]
                            end_time = time.time()
                            episode_process_time.append(end_time - start_time)
            
            elif algorithm in ["TBTA", "TBTA2", "TBTA-PPO"]:
                un_taks_obj = worldModel.tasks 
                if un_taks_obj != []:
                    start_time = time.time() 
                    actions = {}  
                    agent_id = f'{worldModel.agents_obj[worldModel.agent_selector._current_agent].name}'                                            
                    obs_batch = Batch(obs=[observation[agent_id]], info=[{}])               
                    if worldModel.agents_obj[worldModel.agent_selector._current_agent].type != "AA":                            
                        action = policy(obs_batch).act
                        if algorithm == "TBTA-PPO":
                            action = action.item() if hasattr(action, "item") else action[0]
                        else:
                            action = np.argmax(action)                            
                        actions[agent_id] = action                            
                        end_time = time.time()
                        episode_process_time.append(end_time - start_time)
            
            elif algorithm == "CTBTA":
                un_taks_obj = [worldModel.tasks[i] for i in worldModel.unallocated_tasks()] 
                best_task = -1
                best_task_id = 0
                best_uav = 'agent0'
                if un_taks_obj != []:
                    Qs = {}
                    live_agents = worldModel.get_live_agents()
                    for uav in live_agents:                                       
                        obs_batch = Batch(obs=[observation[uav.name]], info=[{}]) 
                        Qs[uav.name] = policy2(obs_batch).act[0]
                        idx_max = np.argmax(Qs[uav.name])
                        best_task_F = Qs[uav.name][idx_max]
                        if best_task_F > best_task:
                            best_task = best_task_F
                            best_task_id = observation[uav.name]["tasks_info"][idx_max]["id"]
                            best_uav = uav.name                                
                    actions = {best_uav : best_task_id}
            
            observation, reward, done, truncations, info = worldModel.step(actions)
            episode_reward += sum(reward.values())/worldModel.n_agents
            
            if all(done.values()) or all(truncations.values()):            
                latest_metrics = info['metrics']
                latest_metrics['S_Reward'] = episode_reward
                latest_metrics["Algorithm"] = algorithm 
                latest_metrics['seed'] = episode_seed

        total_reward.append(episode_reward)
        total_F_reward.append(latest_metrics['F_Reward'])
        time_reward.append(latest_metrics['F_time'])
        distance_reward.append(latest_metrics['F_distance'])
        quality_reward.append(latest_metrics['F_quality'])
        losses.append(latest_metrics['Losses'])
        kills.append(latest_metrics['Kills'])
        
        process_time.append(np.mean(episode_process_time) if len(episode_process_time) > 0 else 0)         
        process_times.append(len(episode_process_time))          

    worldModel.close()
    
    result = {
        'case_num': case_item['case'],
        'n_agents': n_agents,
        'n_tasks': n_tasks,
        'algorithm': algorithm,
        'mean_S_reward': total_reward,
        'mean_process_time': process_time,
        'process_runs': process_times,
        'mean_R_reward': total_F_reward,
        'time_reward': time_reward,
        'distance_reward': distance_reward,
        'quality_reward': quality_reward,
        'load_reward': [0] * len(total_reward),
        'losses': losses,
        'kills': kills,
        'caseData': case_item,
        'metrics_list': [
            {
                'F_Reward': total_F_reward[i],
                'F_time': time_reward[i],
                'F_distance': distance_reward[i],
                'F_quality': quality_reward[i],
                'Losses': losses[i],
                'Kills': kills[i],
                'S_Reward': total_reward[i],
                'Algorithm': algorithm,
                'seed': i
            }
            for i in range(episodes)
        ]
    }
    return result

algorithms = []
algorithms += ['Random']
algorithms += ["Greedy"]
algorithms += ["Swarm-GAP"]
algorithms += ["CBBA"]
algorithms += ["TBTA"]
algorithms += ["TBTA-PPO"]

episodes = 10
fail_rate = 0.0
render_speed = -1
resolution_increase = 1
print(algorithms)

same_policy = False
   
scal_analysis = "Agents"
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
    cases =  [{'case' : 0, 'Hold': 0, 'Att': 15, 'Rec' : 0,  'F1':0, 'F2': 2, "R1" : 0, "R2" : 0 }]

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

if __name__ == '__main__':
    import concurrent.futures
    import multiprocessing

    num_workers = min(multiprocessing.cpu_count() - 2, 4)
    print(f"Starting parallel execution using {num_workers} processes...")

    tasks_to_run = []
    for case_item in cases:
        for algorithm in algorithms:
            tasks_to_run.append((case_item, algorithm))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                run_case_algorithm,
                case_item,
                algorithm,
                episodes,
                resolution_increase,
                fail_rate,
                render_speed,
                same_policy,
                scal_analysis
            ): (case_item, algorithm)
            for case_item, algorithm in tasks_to_run
        }

        for future in concurrent.futures.as_completed(futures):
            case_item, algorithm = futures[future]
            try:
                res = future.result()
                caseResults.append({
                    'case' : res['n_agents'],
                    'algorithm' : res['algorithm'],
                    'mean_S_reward' : res['mean_S_reward'],
                    'mean_process_time' : res['mean_process_time'],
                    'process_runs' : res['process_runs'],
                    'mean_R_reward' : res['mean_R_reward'],
                    'time_reward' : res['time_reward'],
                    'distance_reward' : res['distance_reward'],
                    'quality_reward' : res['quality_reward'],
                    'load_reward' : res['load_reward'],
                    'losses' : res['losses'],
                    'kills' : res['kills'],
                    'n_Tasks' : res['n_tasks'],
                    'n_Agents' : res['n_agents'],
                    'caseData' : res['caseData']
                })
                totalMetrics.extend(res['metrics_list'])
                print(f"Completed Case {res['case_num']} -> {res['algorithm']} -> S_REW: {np.mean(res['mean_S_reward']):.2f}")
            except Exception as exc:
                print(f"Case {case_item['case']} -> {algorithm} generated an exception: {exc}")

    casesDf = pd.DataFrame(caseResults)
    casesDf.to_csv(f'Cases_{expName}.csv', index=False)

    metricsDf = pd.DataFrame(totalMetrics)
    metricsDf.to_csv(f'Resultados_{expName}.csv', index=False)
    print(f'\nResults Saved to:\nResultados_{expName}.csv')

    try:
        print("\nGenerating comparative plots...")
        import seaborn as sns
        sns.set_style("whitegrid")
        
        if scal_analysis in ["Tasks", "Agents"]:
            eval_param = "n_Tasks" if scal_analysis == "Tasks" else "n_Agents"
            plot_df = casesDf.copy()
            plot_df['avg_S_reward'] = plot_df['mean_S_reward'].apply(np.mean)
            plot_df['avg_R_reward'] = plot_df['mean_R_reward'].apply(np.mean)
            plot_df['avg_process_time'] = plot_df['mean_process_time'].apply(np.mean)
            plot_df['avg_time_reward'] = plot_df['time_reward'].apply(np.mean)
            plot_df['avg_distance_reward'] = plot_df['distance_reward'].apply(np.mean)
            
            # Plot 1: Mean F_Reward vs scaling dimension
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=plot_df, x=eval_param, y='avg_R_reward', hue='algorithm', marker='o', linewidth=2.0)
            plt.title(f'Mean Reward vs Number of {eval_param[2:]} (Episodes: {episodes})', fontsize=14)
            plt.xlabel(f'Number of {eval_param[2:]}', fontsize=12)
            plt.ylabel('Mean Reward (F_Reward)', fontsize=12)
            plt.legend(title='Algorithm', fontsize=11, title_fontsize=12)
            plt.tight_layout()
            plt.savefig(f'Scaling_Reward_{expName}.png')
            print(f"Scaling Reward plot saved to: Scaling_Reward_{expName}.png")
            plt.close()
            
            # Plot 2: Decision Time vs scaling dimension
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=plot_df, x=eval_param, y='avg_process_time', hue='algorithm', marker='o', linewidth=2.0)
            plt.title(f'Mean Decision Time vs Number of {eval_param[2:]} (Episodes: {episodes})', fontsize=14)
            plt.xlabel(f'Number of {eval_param[2:]}', fontsize=12)
            plt.ylabel('Mean Decision Time (seconds)', fontsize=12)
            plt.legend(title='Algorithm', fontsize=11, title_fontsize=12)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'Scaling_Time_{expName}.png')
            print(f"Scaling Decision Time plot saved to: Scaling_Time_{expName}.png")
            plt.close()

            # Plot 3: Distance Penalty vs scaling dimension
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=plot_df, x=eval_param, y='avg_distance_reward', hue='algorithm', marker='o', linewidth=2.0)
            plt.title(f'Mean Distance Reward vs Number of {eval_param[2:]} (Episodes: {episodes})', fontsize=14)
            plt.xlabel(f'Number of {eval_param[2:]}', fontsize=12)
            plt.ylabel('Mean Distance Reward', fontsize=12)
            plt.legend(title='Algorithm', fontsize=11, title_fontsize=12)
            plt.tight_layout()
            plt.savefig(f'Scaling_Distance_{expName}.png')
            print(f"Scaling Distance plot saved to: Scaling_Distance_{expName}.png")
            plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    try:
        if len(cases) == 1:
            fig, ax = plt.subplots(figsize=(7, 5))
            n_agents = cases[0].get('F1', 0) + cases[0].get('R1', 0)
            n_tasks = cases[0].get('Att', 0) + cases[0].get('Rec', 0)
            
            def plot_convergence(i, df, n_agents, n_tasks, algorithm):
                cumulative_means = df.expanding().mean()
                palette = sns.color_palette("Set1", n_colors=5)
                auxDf = cumulative_means.reset_index()
                for b, metric in enumerate(auxDf.columns[1:]):
                    ax.plot(auxDf[metric], label=metric, color=palette[i % len(palette)])
                ax.set_xlabel('Number of Simulations')
                ax.set_ylabel('Cummulative Means')
                ax.set_title(f'Convergence {algorithm} : ({n_agents} uavs, {n_tasks} tasks) - Dynamic: Fail(0.7)')
                ax.legend(['Random','Swarm-GAP','CBBA','TBTA'])
                ax.set_ylim(0.0, 1.2)
                plt.tight_layout()

            for i, algorithm in enumerate(algorithms):
                plot_convergence(i, metricsDf[metricsDf.Algorithm == algorithm].F_Reward, n_agents, n_tasks, algorithm)
            plt.show()
    except Exception as e:
        print(f"Convergence plotting failed: {e}")
