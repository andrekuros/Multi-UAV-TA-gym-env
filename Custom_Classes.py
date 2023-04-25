import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict
from tianshou.utils.net.common import Net
from tianshou.data import Collector
from copy import deepcopy
import numpy as np

import torch.nn.functional as F

class CustomNet(Net):
    def __init__(
        self,
        state_shape_agent: int,
        state_shape_task: int,
        action_shape: int,
        hidden_sizes: List[int],
        device: str,
        nhead: int = 2,
        num_layers: int = 1,
    ):
        super().__init__(            
            state_shape=0,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        
        input_size = 128#state_shape_agent + state_shape_task
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
         
        self.output = nn.Linear(sizes[-1], action_shape).to(device)                
        
        self.drone_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ).to(device)
        
        self.task_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ).to(device)

       
        
        self.drone_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=nhead),
            num_layers=num_layers,
        ).to(device)

        self.task_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=nhead),
            num_layers=num_layers,
        ).to(device)
        
        self.attention = ScaledDotProductAttention(d_model=64)
        

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        
      
        agent_position = torch.tensor(obs["agent_position"], dtype=torch.float32)
        agent_state = torch.tensor(obs["agent_state"], dtype=torch.float32).unsqueeze(dim=1)
        agent_type = torch.tensor(obs["agent_type"], dtype=torch.float32).unsqueeze(dim=1)
        agent_relay_area = torch.tensor(obs["agent_relay_area"], dtype=torch.float32).unsqueeze(dim=1)
        
        drone_embeddings = self.drone_encoder(torch.cat((agent_position, agent_state, agent_type, agent_relay_area), dim=1))
   

        task_position = torch.tensor(obs["task_position"], dtype=torch.float32)
        task_type = torch.tensor(obs["task_type"], dtype=torch.float32).unsqueeze(dim=1)
        
       

        task_embeddings = self.task_encoder(torch.cat((task_position, task_type), dim=1))

        combined_embeddings = torch.cat((drone_embeddings, task_embeddings), dim=1)

        drone_transformed = self.drone_transformer(combined_embeddings)
        task_transformed = self.task_transformer(combined_embeddings)

        drone_mean = drone_transformed.mean(dim=1)
        task_mean = task_transformed.mean(dim=1)
        batch_size = drone_mean.size(0)
        state = torch.cat((drone_mean.view(batch_size, 1), task_mean.view(batch_size, 1)), dim=-1)

        q = k = v = combined_embeddings  # Use the combined embeddings of UAVs and tasks
        context_vector, attention_weights = self.attention(q, k, v)

        x = self.hidden_layers(context_vector)        

          
        
        return self.output(x), state


class CustomCollector(Collector):  
    def _policy_forward(self, obs):
        obs = deepcopy(obs)
        agent_id = list(obs.keys())[0]
        agent_obs = obs[agent_id]
        return self.policy.forward(agent_obs)


from pettingzoo.utils.wrappers import BaseWrapper

from pettingzoo.utils.wrappers import OrderEnforcingWrapper

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

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

    def forward(self, q, k, v):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
