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

        self.max_tasks = 50 + 1
        self.task_size = int(state_shape_task / self.max_tasks)

        self.max_agents = 20
        self.agent_size = 5                                                                     
                
        # self.agents_encoder = nn.Sequential(
        #     nn.Linear(self.agent_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32)
        # ).to(device)

        # self.own_encoder = nn.Sequential(
        #     nn.Linear(self.agent_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32)
        # ).to(device)
        
        
        self.task_encoder = nn.Sequential(
            nn.Linear(self.task_size, 64),
            # nn.ReLU(),
            # nn.Linear(64, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64)
        ).to(device)
               
        self.embedding_size = 64 #sum of drone and task encoder
        

        #self.multihead_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead).to(device)
        self.own_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead, batch_first=True).to(device)
        #self.agents_attention = nn.MultiheadAttention(embed_dim=32, num_heads=nhead).to(device)

        self.norm1 = nn.LayerNorm(self.embedding_size)
        self.norm2 = nn.LayerNorm(self.embedding_size)
        self.norm3 = nn.LayerNorm(self.embedding_size)
        
        # self.task_score_seq = nn.Sequential(
        #         nn.Linear(32, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 32 )
        #     ).to(device)
        
        self.decoder_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead, batch_first=True).to(device)
        #self.output = nn.Linear(sizes[-1], action_shape).to(device)  
        
        
        self.output = nn.Linear(64, 1).to(device)
                
        
    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        
        # Drone embeddings: input (12) from agent_obs | output (32) to combined_output
        #print(obs)
        agent_position = torch.tensor(obs["agent_position"], dtype=torch.float32).to(self.device)
        agent_state = torch.tensor(obs["agent_state"], dtype=torch.float32)
        agent_type = torch.tensor(obs["agent_type"], dtype=torch.float32).to(self.device)
        next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).to(self.device)
        position_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)         
        
        #drone_embeddings = self.drone_encoder(torch.cat((tasks_info,position_after_last_task, next_free_time, agent_type ), dim=-1))               

        # own_embeddings = torch.cat((agent_type,
        #                             position_after_last_task,                                                         
        #                             next_free_time), dim=1)                            
        #own_embeddings = self.own_encoder(own_embeddings)
        #own_embeddings = own_embeddings.unsqueeze(1)  # Now own_embeddings has shape (10, 1, 64)

        tasks_info = torch.tensor(obs["tasks_info"], dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor         
        tasks_info = tasks_info.view(-1, self.max_tasks, self.task_size)#int(len(tasks_info[0]/10))) #calculate the size of each tasks, and consider 10 max tasks  
        
        mask = create_padding_mask(tasks_info)
        mask = mask.to(torch.bool)

        #print("MASK:", mask.shape)               
        task_embeddings = self.task_encoder(tasks_info)
        
        # agents_info = torch.tensor(obs["agents_info"], dtype=torch.float32).to(self.device)  # Convert agents_info to tensor         
        # agents_info = agents_info.view(-1, self.max_agents, self.agent_size)#int(len(tasks_info[0]/10))) #calculate the size of each tasks, and consider 10 max tasks                         
        # agents_embeddings = self.agents_encoder(agents_info)        
                        
        own_attention_output, _ = self.own_attention(task_embeddings, task_embeddings, task_embeddings, key_padding_mask=mask)

        #print("own_attention_output1:", own_attention_output)
        # own_attention_output, _ = self.own_attention(queries, own_embeddings.transpose(0, 1), own_embeddings.transpose(0, 1))
        # agents_attention_output, _ = self.agents_attention(queries, agents_embeddings.transpose(0, 1), agents_embeddings.transpose(0, 1))
        own_attention_output = self.norm1(own_attention_output + task_embeddings)  # Add skip connection and normalization        
        #print("own_attention_output:", own_attention_output.shape)
        
        own_attention_output, _ = self.decoder_attention(own_attention_output, own_attention_output, own_attention_output, key_padding_mask=mask)
        own_attention_output = self.norm2(own_attention_output + task_embeddings)  # Add skip connection and normalization
        # Combine the attention outputs
        #print("own_attention_output1:", own_attention_output)
        attention_output = own_attention_output#torch.cat((own_attention_output, agents_attention_output), dim=-1)
        #print("own_attention_output2:", own_attention_output)
        #attention_output = attention_output.transpose(0, 1) 
                        
        #task_scores = self.task_score_seq(attention_output)
        #task_scores = task_scores.squeeze(-1)
        #print("Task_Scores:" , task_scores.shape)
        
        output = self.output(attention_output)
        
        #task_probabilities = F.softmax(task_scores, dim=-1)            
        #print("output:", output)

        
        #softmax_output = F.softmax(output, dim=1) 
        #masked_output = softmax_output.masked_fill(mask == 0, 0)  # This will set the attention weights for the padding tokens to -inf

        #print("soft:", softmax_output)
        #print("shape", softmax_output.shape)

        
        #softmax_output = torch.squeeze(softmax_output, -1)

        softmax_output = F.softmax(output, dim=1) 
        softmax_output = softmax_output.masked_fill(~mask.unsqueeze(-1) == 0, 0)  # This will set the softmax values for the padding tokens to 0
        softmax_output = torch.squeeze(softmax_output, -1)



        #print("softmax_output_Final:" , softmax_output.shape)     
        return softmax_output, state
    
def create_padding_mask(seq):
    seq = torch.eq(seq, -1)  # This will create a mask with True where the sequence is -1 (padding) and False elsewhere
    seq = seq.all(dim=-1)  # This will reduce the last dimension, giving us True only where all values in the task are -1
    return seq  # No need to unsqueeze


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