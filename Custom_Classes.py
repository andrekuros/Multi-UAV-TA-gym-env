import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict
from tianshou.utils.net.common import Net
from tianshou.data import Collector
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

import torch.nn.functional as F
class CustomNet(Net):
    def __init__(
        self,
        state_shape_agent: int,
        state_shape_task: int,
        action_shape: int,
        hidden_sizes: List[int],
        device: str,
        nhead: int = 4,
        num_layers: int = 1,
    ):
        super().__init__(            
            state_shape=0,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        
        input_size = 64
                
        sizes = [input_size] + hidden_sizes + [action_shape]
                       
        self.hidden_layers = []
        for i in range(len(sizes) - 2):
            self.hidden_layers.extend([
                nn.Linear(sizes[i], sizes[i + 1]),
                nn.ReLU()
            ])
            
        self.hidden_layers.extend([
            nn.Linear(sizes[-2], sizes[-1])
        ])
        
        self.hidden_layers = nn.Sequential(*self.hidden_layers).to(device)                              
                
        self.drone_encoder = nn.Sequential(
            nn.Linear(state_shape_agent, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ).to(device)
        
        self.task_encoder = nn.Sequential(
            nn.Linear(state_shape_task, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64)
        ).to(device)
               
        self.embedding_size = 128 #sum of drone and task encoder
        
        self.combined_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=nhead),
            num_layers=num_layers,
        ).to(device)
        
        self.query_linear = nn.Linear(self.embedding_size, 64).to(device)
        self.key_linear = nn.Linear(self.embedding_size, 64).to(device)
        self.value_linear = nn.Linear(self.embedding_size, 64).to(device)
              
        self.attention = ScaledDotProductAttention(d_model=64)
        
        self.output = nn.Linear(sizes[-1], action_shape).to(device)  
                
        

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        
        # Drone embeddings: input (12) from agent_obs | output (32) to combined_output
        agent_position = torch.tensor(obs["agent_position"], dtype=torch.float32).to(self.device)
        agent_state = torch.tensor(obs["agent_state"], dtype=torch.float32)
        agent_type = torch.tensor(obs["agent_type"], dtype=torch.float32).to(self.device)
        next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).unsqueeze(-1).to(self.device)
        position_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)         
        drone_embeddings = self.drone_encoder(torch.cat((agent_position, agent_state, agent_type, next_free_time, position_after_last_task), dim=-1))
       
        tasks_info = torch.tensor(obs["tasks_info"], dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor                        
        task_embeddings = self.task_encoder(tasks_info)
        
        # Create a binary mask from tasks_info tensor by checking if the first value of each task (every 5 values) is not -1
        #tasks_info_mask = (tasks_info[:, 0::5] != -1)
        # Expand the mask dimensions to match the required shape for the attention mechanism
        #tasks_info_mask = tasks_info_mask.unsqueeze(1).unsqueeze(2)


                     
        combined_output = torch.cat((drone_embeddings.unsqueeze(1), task_embeddings.unsqueeze(1)), dim=1)
                
        # Transformer layers: input (128) from transformer_input | output (128) to attention mechanism    
        transformer_input = combined_output.view(1, -1, self.embedding_size)
        transformer_output = self.combined_transformer(transformer_input).view(-1, self.embedding_size)
        
       # Attention mechanism: input (transformer_output (128)) | output (64) to remaining layers  
        q = self.query_linear(transformer_output)
        k = self.key_linear(transformer_output)
        v = self.value_linear(transformer_output)     
                      
        attention_output, _ = self.attention(q, k, v, mask=None)
        
        # Process the attention output with the remaining layers
        x = self.hidden_layers(attention_output)
        output = self.output(x)
    
        # Apply the softmax function
        softmax_output = F.softmax(output, dim=-1)        
        
        return softmax_output, state



class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_model)        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))        
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights





class CustomCollector(Collector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #print("CustomCollector initialized")

    def policy_forward(self, obs):
        print(obs)
        obs = deepcopy(obs)
        actions = {}
        #print("CustomCollector policy_forward method called")
        for agent_id, agent_obs in obs.items():
            policy = self.policy.policies[agent_id]
            action, _ = policy(agent_obs)
            actions[agent_id] = action
        return actions

    def collect(self, n_step: Optional[int] = None, n_episode: Optional[int] = None, random: Optional[bool] = False, render: Optional[float] = None) -> Dict[str, Any]:
        #print("CustomCollector collect method called")
        return super().collect(n_step=n_step, n_episode=n_episode, random=random, render=render)



class CustomParallelToAECWrapper(OrderEnforcingWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._initialize_observation_spaces()

    def _initialize_observation_spaces(self):
        if not hasattr(self.env, 'agents') or self.env.agents is None:
            self.env.reset()
        self._observation_spaces = {
            agent: self.env.observation_space
            for agent in self.env.agents
        }
    
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @property
    def observation_spaces(self):
        return self._observation_spaces

