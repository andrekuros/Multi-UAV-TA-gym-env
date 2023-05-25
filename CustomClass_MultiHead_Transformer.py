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

class CustomNetMultiHead(Net):
    def __init__(
        self,
        state_shape_agent: int,
        state_shape_task: int,
        action_shape: int,
        hidden_sizes: List[int],
        device: str,
        nhead: int = 8,
        num_layers: int = 1,
    ):
        super().__init__(            
            state_shape=0,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        
        input_size = 64

        self.max_tasks = 30
        self.task_size = int(state_shape_task / self.max_tasks)

        self.max_agents = 10
        self.agent_size = 5
                
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
                
        self.agents_encoder = nn.Sequential(
            nn.Linear(self.agent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        ).to(device)

        self.own_encoder = nn.Sequential(
            nn.Linear(self.agent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        ).to(device)
        
        
        self.task_encoder = nn.Sequential(
            nn.Linear(self.task_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        ).to(device)
               
        self.embedding_size = 32 #sum of drone and task encoder
        

        #self.multihead_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead).to(device)
        self.own_attention = nn.MultiheadAttention(embed_dim=32, num_heads=nhead).to(device)
        self.agents_attention = nn.MultiheadAttention(embed_dim=32, num_heads=nhead).to(device)

        self.task_score_seq = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 32 )
            ).to(device)
        
        #self.output = nn.Linear(sizes[-1], action_shape).to(device)  
        self.output = nn.Linear(32, 1).to(device)
                
        
    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        
        # Drone embeddings: input (12) from agent_obs | output (32) to combined_output
        #print(obs)
        agent_position = torch.tensor(obs["agent_position"], dtype=torch.float32).to(self.device)
        agent_state = torch.tensor(obs["agent_state"], dtype=torch.float32)
        agent_type = torch.tensor(obs["agent_type"], dtype=torch.float32).to(self.device)
        next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).to(self.device)
        position_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)         
        
        #drone_embeddings = self.drone_encoder(torch.cat((tasks_info,position_after_last_task, next_free_time, agent_type ), dim=-1))               

        own_embeddings = torch.cat((agent_type,
                                    position_after_last_task,                                                         
                                    next_free_time), dim=1)
                            
        own_embeddings = self.own_encoder(own_embeddings)
        own_embeddings = own_embeddings.unsqueeze(1)  # Now own_embeddings has shape (10, 1, 64)

        tasks_info = torch.tensor(obs["tasks_info"], dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor         
        tasks_info = tasks_info.view(-1, self.max_tasks, self.task_size)#int(len(tasks_info[0]/10))) #calculate the size of each tasks, and consider 10 max tasks                         
        task_embeddings = self.task_encoder(tasks_info)
        
        agents_info = torch.tensor(obs["agents_info"], dtype=torch.float32).to(self.device)  # Convert agents_info to tensor         
        agents_info = agents_info.view(-1, self.max_agents, self.agent_size)#int(len(tasks_info[0]/10))) #calculate the size of each tasks, and consider 10 max tasks                         
        agents_embeddings = self.agents_encoder(agents_info)        
        
        #print("task_embeddings",task_embeddings.shape )
        #print("own_embeddings:", own_embeddings.shape)
        # print("agents_embeddings:", agents_embeddings.shape)
        
        # Prepare the queries, keys, and values
        queries = task_embeddings.transpose(0, 1)  # MultiheadAttention expects input in shape (seq_len, batch, embedding_dim)
        
        own_attention_output, _ = self.own_attention(queries, own_embeddings.transpose(0, 1), own_embeddings.transpose(0, 1))
        agents_attention_output, _ = self.agents_attention(queries, agents_embeddings.transpose(0, 1), agents_embeddings.transpose(0, 1))

        # print("task_embeddings_query",queries.shape )
        #print("own_attention_output:", own_attention_output.shape)
        # print("agents_attention_outputs:", agents_attention_output.shape)

        # Combine the attention outputs
        attention_output = own_attention_output#torch.cat((own_attention_output, agents_attention_output), dim=-1)
        attention_output = attention_output.transpose(0, 1) 
                        
        task_scores = self.task_score_seq(attention_output)
        task_scores = task_scores.squeeze(-1)
        #print("Task_Scores:" , task_scores.shape)
        
        output = self.output(attention_output)
        #task_probabilities = F.softmax(task_scores, dim=-1)            
        
        softmax_output = F.softmax(output, dim=1) 
        #print("softmax_output:" , softmax_output.shape)
        softmax_output = torch.squeeze(softmax_output, -1)
        #print("softmax_output_Final:" , softmax_output.shape)
       
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