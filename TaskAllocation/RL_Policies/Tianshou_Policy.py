
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
def _get_model(model="CustomNetMultiHead", env = None, seed = 0):
    
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
    
    return PettingZooEnv(env, seed)

