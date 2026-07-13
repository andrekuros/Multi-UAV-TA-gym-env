
import torch
from tianshou.env.pettingzoo_env import PettingZooEnv

from .Custom_Classes import CustomNet
from .Custom_Classes import CustomCollector
from .Custom_Classes import CustomParallelToAECWrapper

from tianshou.policy import BasePolicy,  MultiAgentPolicyManager, RandomPolicy
from .EvalDqn import DQNPolicy

#from CustomClass_multi_head import CustomNet
from .Custom_Classes_simplified import CustomNetSimple
#from Custom_Classes_simplified import CustomCollectorSimple
#from Custom_Classes_simplified import CustomParallelToAECWrapperSimple

from .CustomClasses_Transformer_Reduced import CustomNetReduced
from .CustomClass_MultiHead_Transformer import CustomNetMultiHead
from mUAV_TA.DroneEnv import MultiUAVEnv

# "CustomNet" or "CustomNetSimple" or "CustomNetReduced" or "CustomNetMultiHead"
def _get_model(model="CustomNetMultiHead", env = None, seed = 0, algorithm="DQN"):
    
    env = _get_env(env, seed)
     
    agent_name = env.agents[0]  # Get the name of the first agent        
    agent_observation_space = env.observation_space # assuming 'agent0' is a valid agent name
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
    

    #state_shape_task = env.observation_space["tasks_info"].shape[0]

    state_shape_task = 30 * 12 #env.observation_space["tasks_info"].shape[0]
                  
    action_shape = env.action_space[agent_name].shape[0]
    #action_shape = env.action_space[agent_name].n

    if algorithm == "PPO":
        from tianshou.policy import PPOPolicy
        from tianshou.utils.net.discrete import Actor, Critic
        from tianshou.utils.net.common import ActorCritic
        from .CustomClass_MultiHead_Transformer_PPO_Critic import CriticNetMultiHead
        from gymnasium.spaces import Box

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model == "CustomNetMultiHead":
            netActor = CustomNetMultiHead(            
                state_shape_agent=state_shape_agent,
                state_shape_task=state_shape_task,
                action_shape=action_shape,
                hidden_sizes=[128,128],
                device=device,
                task_size=13,
                embedding_size=64,
                is_ppo=True,
                deep_task_encoder=False,
            ).to(device)

            netCritic = CriticNetMultiHead(
                state_shape_agent=state_shape_agent,
                state_shape_task=state_shape_task,
                action_shape=action_shape,
                hidden_sizes=[128,128],
                device=device,
                task_size=13,
                embedding_size=64,
                is_ppo=True,
                deep_task_encoder=False,
            ).to(device)

            actor = Actor(netActor, action_shape, device=device).to(device)
            critic = Critic(netCritic, device=device).to(device)
            
            actor_critic = ActorCritic(actor, critic)
            optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)

            dist = torch.distributions.Categorical

            agent_learn = PPOPolicy(
                actor=actor,
                critic=critic,
                optim=optim,
                dist_fn=dist,
                action_scaling=isinstance(env.action_space[env.agents[0]], Box),
                discount_factor=0.99,
                max_grad_norm=0.5,
                eps_clip=0.2,
                vf_coef=0.5,
                ent_coef=0.0,
                gae_lambda=0.95,
                reward_normalization=0,
                dual_clip=None,
                value_clip=0,
                action_space=env.action_space[env.agents[0]],
                deterministic_eval=True,
                advantage_normalization=0,
                recompute_advantage=0,
            )
            agent_learn.action_type = "discrete"
            return agent_learn

    if model == "CustomNetMultiHead":
        net = CustomNetMultiHead(            
            state_shape_agent=state_shape_agent,
            state_shape_task=state_shape_task,
            action_shape=action_shape,
            hidden_sizes=[128,128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    if model == "CustomNetReduced":
         net = CustomNetReduced(            
            state_shape_agent=state_shape_agent,
            state_shape_task=state_shape_task,
            action_shape=action_shape,
            hidden_sizes=[128,128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    CustomNetSimple
    if model == "CustomNetSimple":
         net = CustomNetSimple(            
            state_shape_agent=state_shape_agent,
            state_shape_task=state_shape_task,
            action_shape=action_shape,
            hidden_sizes=[128,128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")


    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            target_update_freq=1500                        
        )  
  
    return agent_learn


def _get_env(env = None, seed = 0):
    """This function is needed to provide callables for DummyVectorEnv."""
   
    if env is None:         
        env_paralell = MultiDroneEnv()                 
    else:
        env_paralell = env          
    #env = parallel_to_aec_wrapper(env_paralell)              
    env = CustomParallelToAECWrapper(env_paralell) 
    
    return PettingZooEnv(env)

