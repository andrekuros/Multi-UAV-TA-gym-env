import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger


#from Custom_Classes import CustomNet
from Custom_Classes import CustomCollector
from Custom_Classes import CustomParallelToAECWrapper

from CustomClass_multi_head import CustomNet
from Custom_Classes_simplified import CustomNetSimple
from Custom_Classes_simplified import CustomCollectorSimple
from Custom_Classes_simplified import CustomParallelToAECWrapperSimple

from DroneEnv import MultiDroneEnv

def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    policy_load_path = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    
    env = _get_env()
    agent_name = env.agents[0]  # Get the name of the first agent
    agent_observation_space = env.observation_space[agent_name]  # Get the observation_space for that agent
       
    agent_observation_space = env.observation_space["agent0"]  # assuming 'agent0' is a valid agent name
    state_shape_agent_position = agent_observation_space["agent_position"].shape[0]
    state_shape_agent_state = agent_observation_space["agent_state"].shape[0]
    state_shape_agent_type = agent_observation_space["agent_type"].shape[0]
    state_shape_next_free_time = agent_observation_space["next_free_time"].shape[0]
    state_shape_position_after_last_task = agent_observation_space["position_after_last_task"].shape[0]       
    #state_shape_agent_relay_area = agent_observation_space["agent_relay_area"].shape[0]
    
    
    state_shape_agent = (state_shape_agent_position + state_shape_agent_state +
                     state_shape_agent_type+ state_shape_next_free_time + state_shape_position_after_last_task #+                     
                     #state_shape_agent_relay_area
                     )                 
    
    state_shape_task = env.observation_space["agent0"]["tasks_info"].shape[0]
                  
    action_shape = env.action_space[agent_name].shape[0]
    #action_shape = env.action_space[agent_name].n
               
    if agent_learn is None:
        # model
        #net = CustomNetSimple(
        net = CustomNet(
            state_shape_agent=state_shape_agent,
            state_shape_task=state_shape_task,
            action_shape=action_shape,
            hidden_sizes=[128,128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=5e-4)
    
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.7,
            estimation_step=20,
            target_update_freq=100,
        )  
        
        if load_model == True:
            # Load the saved checkpoint             
            agent_learn.load_state_dict(torch.load(model_load_path))
            print("loaded")
            
        
        agents = [agent_learn for _ in range(len(env.agents))]
        
        
            

    policy = MultiAgentPolicyManager(agents, env)    
        
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env_paralell = MultiDroneEnv()
    #env = parallel_to_aec_wrapper(env_paralell)    
    env = CustomParallelToAECWrapper(env_paralell)
    
    return PettingZooEnv(env)

if __name__ == "__main__":
        
    train_env_num = 1
    test_env_num = 1
    
    model_load_path = os.path.join("dqn_Custom", "policy_03.pth")    
    load_model = False   
    
    torch.set_grad_enabled(True) 
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(train_env_num)])
    test_envs = DummyVectorEnv([_get_env for _ in range(test_env_num)]) 

    # seed
    seed = 1
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()
    

    # ======== Step 3: Collector setup =========
    train_collector = CustomCollector(
        policy,
        train_envs,
        VectorReplayBuffer(100_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = CustomCollector(policy, test_envs, exploration_noise=True)
     
    train_collector.collect(n_step=1000)  # batch size * training_num
    #test_collector.collect(n_step=1000) 
    
    # ======== tensorboard logging setup =========
    log_path = os.path.join('./', "Logs", "dqn")
    writer = SummaryWriter(log_path)
    #writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)
        
    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("dqn_Custom", "policy_03.pth")        
        os.makedirs(os.path.join("dqn_Custom"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 9939.0

    def train_fn(epoch, env_step):
        epsilon = max(0.5, 0.5 - epoch * 0.0005)
        policy.policies[agents[0]].set_eps(epsilon)

    def test_fn(epoch, env_step):
        epsilon = 0.01#max(0.05, 0.1 - epoch * 0.001)
        policy.policies[agents[0]].set_eps(epsilon)
        
    def reward_metric(rews):       
        #print(rews)
        return rews.mean()#[:,0]
                           
    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,        
        max_epoch=10000,
        step_per_epoch=10000 * train_env_num,
        step_per_collect=300 * train_env_num,
        episode_per_test= 10 * test_env_num,
        batch_size=32,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
        show_progress = True
        )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[0]])")

#%%%
from typing import Optional, Tuple

import numpy as np
import torch
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offpolicy_trainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import torch
#from Custom_Classes import CustomCollector

# Create a new instance of the policy with the same architecture as the saved policy
policy, optim, _ = _get_agents()
model_save_path = os.path.join("dqn_Custom", "policy_03.pth")        

# Load the saved checkpoint
policy_test = policy.policies['agent0']
policy_test.load_state_dict(torch.load(model_save_path ))

envs = DummyVectorEnv([_get_env for _ in range(1)])

#policy_test.policies['agent0'].eval()
#policy.policies['agent0'].set_eps(0.9)

policy_test.eval()
policy_test.set_eps(0.00)

#collector = CustomCollector(policy.policies['agent0'], envs, exploration_noise=True)
collector = CustomCollector(policy_test, envs, exploration_noise=True)

#collector.collect(n_episode=10)
collector.collect(n_episode=1, render=1 / 5000)