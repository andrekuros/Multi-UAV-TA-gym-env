from DroneEnv import MultiDroneEnv
from DroneEnv import env
from swarm_gap import SwarmGap
from tessi import TessiAgent
import pandas as pd
import argparse
import json
import time
import copy

from gym import spaces
from godot_rl.core.godot_env import GodotEnv
from godot_rl.core.utils import lod_to_dol

import MultiDroneEnvUtils as utils


def createGodotEnv():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--env_path",
        default="W:/OneDrive/Doutorado/SwarmSimASA/GoDot/godot_rl_agents-main/examples/godot_rl_agents_examples-main/godot_rl_agents_examples-main/examples/FlyBy/bin/FlyBy.exe",#envs/example_envs/builds/JumperHard/jumper_hard.x86_64",        
        type=str,
        help="The Godot binary to use, do not include for in editor training",
    )
    
    parser.add_argument("--speedup", default=10, type=int, help="whether to speed up the physics in the env")
    parser.add_argument("--renderize", default=1, type=int, help="whether renderize or not the screen")
    args, extras = parser.parse_known_args()
    env = GodotEnv( env_path=args.env_path,
             port=11008,
             show_window=True,
             seed=0,
             framerate=None,
             action_repeat=60,
             speedup=args.speedup,
             convert_action_space=False,
             renderize=args.renderize
             )

    return env

algorithms =  ["Random"]#["Tessi1"] #"Swarm-GAP"

num_drones = 10
num_targets = 50
num_obstacles = 0

episodes = 30
render_speed = 1


simEnv = "PyGame"

if simEnv == "PyGame":
    env = env(action_mode= "TaskAssign", render_speed=render_speed, num_drones=num_drones, num_targets=num_targets, num_obstacles=num_obstacles, max_time_steps=1500)
    #env = env
    

elif simEnv == "Godot":
    env = GodotEnv()

#from pettingzoo.test import parallel_api_test
#from pettingzoo.test import parallel_seed_test
#parallel_api_test(env, num_cycles=1000)
#parallel_seed_test(env, num_cycles=10000, test_kept_state=True)

totalMetrics = []

for algorithm in algorithms:
    
    start_time = time.time()
    print("\nStarting Algorithm:", algorithm)
        
    for episode in range(episodes):
        
        observation  = env.reset(seed=episode)        
        info         = env.get_initial_state()
        
        drones = info["drones"]
        targets = info["targets"]
        quality_table =  info["quality_table"]
        
        done = {0 : False}
        truncations = {0 : False}
                        
        if algorithm == "Random":            
            planned_actions = utils.generate_random_tasks_all(drones, targets, seed = episode) 
            #print(planned_actions)
        
        if algorithm == "Tessi1":
            agent = TessiAgent(num_drones=num_drones, num_targets=num_targets, tessi_model = 1)
        
        if algorithm == "Tessi2":
            agent = TessiAgent(num_drones=num_drones, num_targets=num_targets, tessi_model = 2)
        
        if algorithm == "Swarm-GAP":
            agent = SwarmGap(drones, targets, quality_table, exchange_interval=1)
    
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
                
                if env.time_steps % agent.exchange_interval == 0:                    
                    actions = agent.process_token()    
            
            elif algorithm == "Tessi1" or algorithm == "Tessi2":            
                
                
                                                        
                if env.time_steps % 10 == 0:
                    # Convert task_allocation to actions
                    actions = agent.allocate_tasks(env.drones, [env.targets[i] for i in env.unallocated_tasks()] )
            
        
            observation, reward, done, truncations, info = env.step(actions)
                        
            #for agent in range(env.NUM_DRONES):                
            #    print(observation[agent])
                #if not all(obs["target_status"], -1):   
                #    print(observation)
            
            if env.render_enabled:
                env.render()
            
            #print(done)
            if all(done.values()):
            #if done:
                info["Algorithm"] = algorithm
                totalMetrics.append(info)
                            
            if all(truncations.values()):                
                print("\nMax Steps Reached:", env.time_steps )
                
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("\nExecution time ", algorithm, execution_time, "seconds")
    
    env.close()



metricsDf = pd.DataFrame(totalMetrics)
# Chamar a função de plotagem

env.plot_metrics(metricsDf, len(env.drones), len(env.targets))
for algorithm in algorithms:
    env.plot_convergence(metricsDf[metricsDf.Algorithm == algorithm], len(env.drones), len(env.targets), algorithm)


#%%%





