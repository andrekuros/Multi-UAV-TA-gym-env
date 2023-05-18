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

class CustomNetReduced(Net):
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
        self.max_tasks = 10
        self.task_size = int(state_shape_task / self.max_tasks)
                
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
            nn.Linear(int(state_shape_task/self.max_tasks), 32),
            #nn.ReLU(),
            #nn.Linear(32, 64),
            #nn.ReLU(),
            #nn.Linear(64, 16)
        ).to(device)
               
        self.embedding_size = 32 #sum of drone and task encoder
        
        self.combined_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=nhead),
            num_layers=num_layers,
        ).to(device)
        
        self.query_linear = nn.Linear(self.embedding_size, 64).to(device)
        self.key_linear = nn.Linear(self.embedding_size, 64).to(device)
        self.value_linear = nn.Linear(self.embedding_size, 64).to(device)
              
        self.attention = ScaledDotProductAttentionReduced(d_model=self.embedding_size)
        
        
        #self.output = nn.Linear(sizes[-1], action_shape).to(device)  
        self.output = nn.Linear(64, 1).to(device)
                
        
    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        
        # Drone embeddings: input (12) from agent_obs | output (32) to combined_output
        agent_position = torch.tensor(obs["agent_position"], dtype=torch.float32).to(self.device)
        agent_state = torch.tensor(obs["agent_state"], dtype=torch.float32)
        agent_type = torch.tensor(obs["agent_type"], dtype=torch.float32).to(self.device)
        next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).to(self.device)
        position_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)         
        
        #drone_embeddings = self.drone_encoder(torch.cat((tasks_info,position_after_last_task, next_free_time, agent_type ), dim=-1))
       
        #drone_embeddings = self.drone_encoder(torch.cat((agent_position, agent_state, agent_type, next_free_time, position_after_last_task), dim=-1))
        
        tasks_info = torch.tensor(obs["tasks_info"], dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor         
        tasks_info = tasks_info.view(-1, self.max_tasks, self.task_size)#int(len(tasks_info[0]/10))) #calculate the size of each tasks, and consider 10 max tasks                         
        task_embeddings = self.task_encoder(tasks_info)
        

        #position_after_last_task = tasks_info.view(-1, 6, 2)
        #drone_embeddings = self.drone_encoder(position_after_last_task)

        transformer_output = self.combined_transformer(task_embeddings).view(-1, self.max_tasks, self.embedding_size)       
                
        q = self.query_linear(transformer_output)
        k = self.key_linear(transformer_output)
        v = self.value_linear(transformer_output)     
                      
        attention_output, _ = self.attention(q, k, v, mask=None)
        
        # Process the attention output with the remaining layers
        #x = self.hidden_layers(attention_output)       
        
        #print("Attention->", attention_output.shape, attention_output)                        
        output = self.output(attention_output)
        
        #print("OUt->", output.shape, output)
        # Apply the softmax function
        softmax_output = F.softmax(output, dim=1) 
        softmax_output = torch.squeeze(softmax_output, -1)
        
        #print("SOFT->", softmax_output.shape, softmax_output)
        
        return softmax_output, state



class ScaledDotProductAttentionReduced(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttentionReduced, self).__init__()
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_model)        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))        
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights