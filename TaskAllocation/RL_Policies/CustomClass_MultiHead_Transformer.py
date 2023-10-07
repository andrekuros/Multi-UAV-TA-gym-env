from math import inf
from sympy import PythonIntegerRing
import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict
from tianshou.utils.net.common import Net
import numpy as np
import torch.nn.functional as F

from mUAV_TA.MultiDroneEnvData import SceneData

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

        self.nhead = nhead

        self.obs_mode = "Pre_Process"
        #self.obs_mode = "Raw_Data"

        self.max_tasks = 31
        self.task_size = 12#int(state_shape_task / self.max_tasks)

        self.max_agents = 10
        self.agent_size = 5 

        
        self.random_weights = False  

        self.sceneData = SceneData()                                                                 
                
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
            nn.ReLU(),
            nn.Linear(64, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)
                       
        self.embedding_size = 64 #sum of drone and task encoder        

        #self.multihead_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead).to(device)
        self.own_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=self.nhead, batch_first=True).to(device)
        #self.agents_attention = nn.MultiheadAttention(embed_dim=32, num_heads=nhead).to(device)

        # self.norm1 = nn.LayerNorm(self.embedding_size).to(device)
        # self.norm2 = nn.LayerNorm(self.embedding_size).to(device)
       
        #  self.task_score_seq = nn.Sequential(
        #         nn.Linear(32, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 32 )
        #     ).to(device)
        
        # self.decoder_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead, batch_first=True).to(device)
        #self.output = nn.Linear(sizes[-1], action_shape).to(device)  
     
        self.output = nn.Linear(64, 1).to(device) 

        if self.random_weights:
            self.weigths_reset()      
                

    def weigths_reset(self):

        with torch.no_grad():
            # Initialize the weights of the query linear layer
            self.own_attention.in_proj_weight[:self.embedding_size].uniform_(-1, 1)
            
            # Initialize the weights of the key linear layer
            self.own_attention.in_proj_weight[self.embedding_size:2*self.embedding_size].uniform_(-1, 1)
            
            # Initialize the weights of the value linear layer
            self.own_attention.in_proj_weight[2*self.embedding_size:].uniform_(-1, 1)
            
            # If you also want to initialize the biases:
            self.own_attention.in_proj_bias[:self.embedding_size].uniform_(-1, 1) # for query
            self.own_attention.in_proj_bias[self.embedding_size:2*self.embedding_size].uniform_(-1, 1) # for key
            self.own_attention.in_proj_bias[2*self.embedding_size:].uniform_(-1, 1) # for value

        with torch.no_grad():
            # Initialize the weights of the query linear layer
            self.decoder_attention.in_proj_weight[:self.embedding_size].uniform_(-1, 1)
            
            # Initialize the weights of the key linear layer
            self.decoder_attention.in_proj_weight[self.embedding_size:2*self.embedding_size].uniform_(-1, 1)
            
            # Initialize the weights of the value linear layer
            self.decoder_attention.in_proj_weight[2*self.embedding_size:].uniform_(-1, 1)
            
            # If you also want to initialize the biases:
            self.decoder_attention.in_proj_bias[:self.embedding_size].uniform_(-1, 1) # for query
            self.decoder_attention.in_proj_bias[self.embedding_size:2*self.embedding_size].uniform_(-1, 1) # for key
            self.decoder_attention.in_proj_bias[2*self.embedding_size:].uniform_(-1, 1) # for value

        with torch.no_grad():
            self.output.weight.uniform_(-1, 1)  # random values between -1 and 1

        with torch.no_grad():
            for layer in self.task_encoder:
                if isinstance(layer, nn.Linear):
                    layer.weight.uniform_(-1, 1)
                    layer.bias.uniform_(-1, 1)

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
              
        # Agent States
        #agent_position   = torch.tensor(obs["agent_position"], dtype=torch.float32).to(self.device)
        #agent_caps       = torch.tensor(obs["agent_caps"], dtype=torch.float32).to(self.device)
        #agent_attack_cap = torch.tensor(obs["agent_attack_cap"], dtype=torch.float32).to(self.device)
        #next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).to(self.device)
        #pos_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)
        #alloc_task       = torch.tensor(obs["alloc_task"], dtype=torch.float32).to(self.device) 

        agent_position   = obs["agent_position"]
        agent_caps       = obs["agent_caps"]
        #agent_attack_cap = torch.tensor(obs["agent_attack_cap"], dtype=torch.float32).to(self.device)
        #next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).to(self.device)
        #pos_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)
        alloc_task       = obs["alloc_task"]
        
        # agent_type = torch.tensor(obs["agent_type"], dtype=torch.float32).to(self.device)
        # next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).to(self.device)
        # position_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)                 

        task_values = []       
               
        if self.obs_mode == "Pre_Process":
        
            for i,batch in enumerate(obs["tasks_info"]):
                        
                #ownType = int(np.argmax(obs["agent_type"][i]))
                
                batch_tasks = []                

                for task in batch:
                                                   
                    #print("tasks", task)
                    
                    distance = self.euclidean_distance(agent_position[i], task["position"])  # Compute the distance
                    # Calculate the heading (angle) from A to B
                    theta = np.arctan2(task["position"][1] - agent_position[i][1], task["position"][0] - agent_position[i][0])                    
                    sin_theta = np.sin(theta)
                    cos_theta = np.cos(theta)

                    is_alloc_task = 1 if task['id'] == alloc_task[i] else 0
                                    
                    if task['id'] != 0:
                        if task['id'] != alloc_task[i]:
                            #print(task['current_reqs'],task['alloc_reqs'], agent_caps[i])
                            reqs_result = task['current_reqs']  - task['alloc_reqs'] + agent_caps[i]
                            
                        else:
                            reqs_result = task['current_reqs']  - task['alloc_reqs'] 
                    else:
                        reqs_result = agent_caps[i] * 0
                        task['init_time'] = 0
                        task['end_time'] = 0
                        distance = 0
                        theta = 0
                        sin_theta = 0
                        cos_theta = 0
                    
                    #task_values.extend(self._one_hot(task.typeIdx, 2))

                    batch_tasks.append([
                        distance,      #1
                        sin_theta,     #1
                        cos_theta,     #1                        
                        task['init_time'], #1
                        task['end_time'],   #1
                        is_alloc_task,    #1
                        #list(obs["agent_type"][i]) + #change to append and others as list too                        
                        #list(self._one_hot(task["type"], 2))
                        
                        ])
                    
                    batch_tasks[-1].extend(reqs_result)
                
                # Pad the task_values array to match the maximum number of tasks
                #batch_tasks.extend( [[-1] * (self.task_size) for _ in range(self.max_tasks - len(batch_tasks)) ])
                num_padding_needed = self.max_tasks - len(batch_tasks)
                padding = [[-1] * self.task_size for _ in range(num_padding_needed)]
                batch_tasks.extend(padding)
                task_values.append(batch_tasks)
       
        #print(task_values)
        #tasks_info = torch.tensor(obs["tasks_info"], dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor                 
        tasks_info = torch.tensor(task_values, dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor     
        #print("Task_Info:\n ",tasks_info)
        #print(tasks_info)
        #tasks_info = tasks_info.view(-1, self.max_tasks, self.task_size)#int(len(tasks_info[0]/10))) #calculate the size of each tasks, and consider 10 max tasks                         
        task_embeddings = self.task_encoder(tasks_info)

        #print("Task_Embeddings:\n ",task_embeddings.shape)
           
        # Create the mask based on your tasks_info tensor
        #mask = tasks_info.ne(-1).any(dim=-1).bool()
        #mask = torch.tensor(obs["mask"], dtype=torch.bool).to(self.device)
        #attn_mask = ~mask                       
        # Expected shape is (L, N) where L is target sequence length and N is batch size
        #attn_mask = attn_mask.unsqueeze(1).expand(-1, tasks_info.size(1), -1)
        #attn_mask = attn_mask.repeat(self.nhead, 1, 1)

        attn_mask = torch.tensor(obs["mask"], dtype=torch.bool).to(self.device)
        attn_mask = None#~attn_mask
        # Use the mask in your attention layers
        
        # Use the mask in your attention layers
        attention_output1, _ = self.own_attention(task_embeddings, task_embeddings, task_embeddings, key_padding_mask=attn_mask)
        attention_output1 = attention_output1 + task_embeddings
        #print("attention_output:\n ",attention_output.shape)
        #print("attention_output1:\n ",attention_output1.shape)
        #attention_output.masked_fill_(~mask.unsqueeze(-1), 0.0)        
        
        #attention_output1 = self.norm1(attention_output1)

        # attention_output2, _ = self.decoder_attention(attention_output1, attention_output1, attention_output1, key_padding_mask=attn_mask)
        # attention_output2 = attention_output2 + attention_output1
        #attention_output.masked_fill_(~mask.unsqueeze(-1), 0.0)        
        #attention_output2 = self.norm2(attention_output2)                                                            
        #print("attention_output2:\n ",attention_output.shape)

        softmax_output = self.output(attention_output1)     

        # print("output:\n ",output)
        #softmax_output = output
        softmax_output = F.softmax(softmax_output, dim=1) 
        softmax_output = torch.squeeze(softmax_output, -1)            
        
        #print("softmax_output:",softmax_output)

        return softmax_output, state   


    
    def _one_hot(self, idx, num_classes):
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[idx] = 1
        return one_hot_vector   
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


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