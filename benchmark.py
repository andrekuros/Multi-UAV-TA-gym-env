import time
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

from mUAV_TA.DroneEnv import MultiUAVEnv
import mUAV_TA.MultiDroneEnvUtils as utils
from TaskAllocation.BehaviourBased.Greedy import GreedyAgent
from TaskAllocation.MarketBased.CBBA import CBBA

def run_benchmark():
    config = utils.agentEnvOptions(     
        render_speed = -1,
        simulation_frame_rate = 0.01,
        max_time_steps = 200,
        action_mode= "TaskAssign",
        agents= {"F1" : 4, "F2" : 0, "R1" : 6, "R2" : 0},                 
        tasks= { "Att" : 10, "Rec" : 20,  "Hold" : 0},
        random_init_pos = False,
        num_obstacles = 0,
        hidden_obstacles = False,
        multiple_tasks_per_agent = True,
        multiple_agents_per_task = True,
        fail_rate = 0.0,
        threats_list = [],
        fixed_seed = 42,
        info = "Benchmark"
    )

    algorithms = ["Random", "Greedy", "CBBA"]
    episodes = 5
    
    results = {alg: {"times": [], "steps": [], "rewards": [], "completed": []} for alg in algorithms}
    
    for algorithm in algorithms:
        print(f"\n===========================")
        print(f"Running {algorithm} Benchmark")
        print(f"===========================")
        
        env = MultiUAVEnv(config)
        
        for ep in range(episodes):
            seed = 42 + ep
            rndGen = random.Random(seed)
            
            env.reset(seed=seed)
            info = env.get_initial_state()
            
            done = {0: False}
            truncations = {0: False}
            
            ep_reward = 0
            ep_steps = 0
            
            if algorithm == "Greedy":
                policy = GreedyAgent()
            elif algorithm == "CBBA":
                policy = CBBA(env.agents_obj, env.tasks, env.max_coord, seed=rndGen.randint(0, 999999))
            
            start_time = time.time()
            
            while not all(done.values()) and not all(truncations.values()):
                actions = {}
                
                un_taks_obj = [task for task in env.tasks 
                               if task.id != 0 and
                               task.allocatedReqs[task.typeIdx] < task.currentReqs[task.typeIdx] and 
                               task.status != 2 and task.type != "Hold"]
                
                if algorithm == "Random":
                    if env.time_steps % 1 == 0 and env.time_steps >= 0:
                        if un_taks_obj:
                            task = rndGen.choice(un_taks_obj)
                            agent = env.agent_by_name[env.current_agent]
                            if agent.tasks == [env.task_idle] and agent.state != 99:
                                actions = {agent.name: env.last_tasks_info.index(task)}
                
                elif algorithm == "Greedy":
                    if env.time_steps % 1 == 0:
                        if un_taks_obj:
                            act = policy.allocate_tasks(env.agents_obj, un_taks_obj)
                            if act and len(act) > 0 and len(act[0]) == 2:
                                agent_name = act[0][0]
                                task_obj = act[0][1]
                                actions[agent_name] = [env.last_tasks_info.index(task_obj)]
                                
                elif algorithm == "CBBA":
                    if env.time_steps % 1 == 0:
                        if len(env.get_live_agents()) > 0 and un_taks_obj:
                            result_actions = policy.allocate_tasks(env.get_live_agents(), un_taks_obj)
                            if result_actions:
                                for act in result_actions:
                                    actions[act[0]] = [env.last_tasks_info.index(t) for t in act[1]]
                
                observation, reward, done, truncations, step_info = env.step(actions)
                ep_reward += sum(reward.values()) / env.n_agents
                ep_steps += 1
                
            end_time = time.time()
            duration = end_time - start_time
            
            metrics = step_info.get('metrics', {})
            completion = metrics.get('Tasks_Completed', 0)
            
            results[algorithm]["times"].append(duration)
            results[algorithm]["steps"].append(ep_steps)
            results[algorithm]["rewards"].append(ep_reward)
            results[algorithm]["completed"].append(completion)
            
            sps = ep_steps / duration if duration > 0 else 0
            print(f"Ep {ep+1}/{episodes} | Time: {duration:.3f}s | SPS: {sps:.0f} | Reward: {ep_reward:.1f} | Completed: {completion}")
            
    # Plot results
    print("\nGenerating Benchmark Plots...")
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    avg_sps = [np.mean(results[alg]["steps"]) / np.mean(results[alg]["times"]) for alg in algorithms]
    axs[0].bar(algorithms, avg_sps, color=['#FF9999', '#66B2FF', '#99FF99'])
    axs[0].set_title('Avg Steps Per Second (Efficiency)')
    axs[0].set_ylabel('SPS')
    
    avg_rewards = [np.mean(results[alg]["rewards"]) for alg in algorithms]
    axs[1].bar(algorithms, avg_rewards, color=['#FF9999', '#66B2FF', '#99FF99'])
    axs[1].set_title('Avg Cumulative Reward')
    axs[1].set_ylabel('Reward')
    
    avg_comp = [np.mean(results[alg]["completed"]) for alg in algorithms]
    axs[2].bar(algorithms, avg_comp, color=['#FF9999', '#66B2FF', '#99FF99'])
    axs[2].set_title('Avg Tasks Completed')
    axs[2].set_ylabel('# Tasks')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Saved benchmark_results.png")

if __name__ == "__main__":
    run_benchmark()
