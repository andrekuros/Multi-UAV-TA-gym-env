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
        num_layers: int = 3,
    ):
        super().__init__(            
            state_shape=0,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        
        input_size = 128
                
        sizes = [input_size] + hidden_sizes + [action_shape]                              
       
        self.drone_encoder = nn.Sequential(
            nn.Linear(state_shape_agent, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ).to(device)
        
        self.task_encoder = nn.Sequential(
            nn.Linear(state_shape_task, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 128)
        ).to(device)
               
        self.embedding_size = 128 #sum of drone and task encoder
        
        
        self.combined_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=nhead),
            num_layers=num_layers,
        ).to(device)
        
       # print(self.combined_transformer.shape)
        
        self.d_model = 128
        self.query_linear = nn.Linear(self.embedding_size, self.d_model ).to(device)
        self.key_linear = nn.Linear(self.embedding_size, self.d_model ).to(device)
        self.value_linear = nn.Linear(self.embedding_size, self.d_model ).to(device)

              
#       self.attention = ScaledDotProductAttention(d_model=64)
        self.attention = MultiHeadAttention(d_model=self.d_model, num_heads=nhead)

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
                 
        self.output = nn.Linear(action_shape , action_shape).to(device)  


    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        
        # Drone embeddings: input (12) from agent_obs | output (32) to combined_output
        agent_position = torch.tensor(obs["agent_position"], dtype=torch.float32).to(self.device)
        agent_state = torch.tensor(obs["agent_state"], dtype=torch.float32)
        agent_type = torch.tensor(obs["agent_type"], dtype=torch.float32).to(self.device)
        next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).to(self.device)
        position_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)         
        drone_embeddings = self.drone_encoder(torch.cat((agent_position, agent_state, agent_type, next_free_time, position_after_last_task), dim=-1))
       
        tasks_info = torch.tensor(obs["tasks_info"], dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor                        
        task_embeddings = self.task_encoder(tasks_info)
    
        print("task_embeddings size:", task_embeddings.size())
        # Create a mask from tasks_info tensor by checking if any value is -1
        mask = (tasks_info != -1)
        # Expand the mask dimensions to match the required shape for the attention mechanism
        mask = mask.unsqueeze(1).unsqueeze(2)
        print("mask size:", mask.size())
        # Create a binary mask from tasks_info tensor by checking if the first value of each task (every 5 values) is not -1
        #tasks_info_mask = (tasks_info[:, 0::5] != -1)
        # Expand the mask dimensions to match the required shape for the attention mechanism
        #tasks_info_mask = tasks_info_mask.unsqueeze(1).unsqueeze(2)

        #combined_output = torch.cat((drone_embeddings.unsqueeze(1), task_embeddings.unsqueeze(1)), dim=1)
        print("drone_embeddings size:", drone_embeddings.size())
        print("task_embeddings size:", task_embeddings.size())
        combined_output = torch.cat((drone_embeddings.unsqueeze(1), task_embeddings.unsqueeze(1)), dim=1)
        print("combined size:", combined_output.size())
                
        # Transformer layers: input (128) from transformer_input | output (128) to attention mechanism    
        transformer_input = combined_output.view(1, -1, self.embedding_size)
        transformer_output = self.combined_transformer(transformer_input).view(-1, combined_output.size(1), combined_output.size(2))

        print("transformer_input size:", transformer_input.size())
        print("transformer_output size:", transformer_output.size())
        
       # Attention mechanism: input (transformer_output (128)) | output (64) to remaining layers  
        q = self.query_linear(transformer_output)
        k = self.key_linear(transformer_output)
        v = self.value_linear(transformer_output)     
                      
        attention_output, _ = self.attention(q, k, v, mask=mask)
        print("attention_output size:", attention_output.size())
        # Process the attention output with the remaining layers
        
        x = self.hidden_layers(attention_output)   
        print("x size:", attention_output.size())
        output = self.output(x)
        
        #output = self.output(attention_output)
        
        #print(output)
    
        # Apply the softmax function
        softmax_output = F.softmax(output, dim=-1) 
        
        #print(softmax_output)
        print(softmax_output.shape)
        
        return softmax_output, state



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # transform inputs to multi-head format
        print("qyery_shape", query.shape)
        query = self.query_linear(query).view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2)       
        key = self.key_linear(key).view(N, key_len, self.num_heads, self.head_dim ).transpose(1, 2)
        value = self.value_linear(value).view(N, value_len, self.num_heads, self.head_dim).transpose(1, 2)


        # calculate the attention weights
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.d_model ** (1 / 2)), dim=3)

        # apply attention weights to values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).transpose(1, 2).contiguous().view(N, query_len, self.d_model)

        # apply dropout and layer normalization
        out = self.dropout(out)
        out = self.layer_norm(out)

        # pass through final linear layer
        out = self.fc_out(out)

        return out

