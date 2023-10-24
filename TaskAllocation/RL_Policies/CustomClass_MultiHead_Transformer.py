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
        self.task_size = 13#int(state_shape_task / self.max_tasks)

        self.max_agents = 20
        self.agent_size = 5 
        
        self.random_weights = False  

        self.sceneData = SceneData()                                                                 
                               
        self.task_encoder = nn.Sequential(
            nn.Linear(self.task_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)
                       
        self.embedding_size = 64 #sum of drone and task encoder        
        
        self.own_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=self.nhead, batch_first=True).to(device)        
        
        self.norm1 = nn.LayerNorm(self.embedding_size).to(device)
        self.norm2 = nn.LayerNorm(self.embedding_size).to(device)
       
        self.decoder_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead, batch_first=True).to(device)    
     
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
        agent_type      = obs["agent_type"]
        agent_position  = obs["agent_position"]
        agent_caps      = obs["agent_caps"]
        alloc_task      = obs["alloc_task"]
                
        task_values = []       
               
        if self.obs_mode == "Pre_Process":
        
            for i,batch in enumerate(obs["tasks_info"]):
                                                        
                batch_tasks = []                

                for task in batch:
                                                                       
                    distance = self.euclidean_distance(agent_position[i], task["position"])  # Compute the distance
                    
                    # Calculate the heading (angle) from A to B
                    theta = np.arctan2(task["position"][1] - agent_position[i][1], task["position"][0] - agent_position[i][0])                    
                    sin_theta = np.sin(theta)
                    cos_theta = np.cos(theta)

                    is_alloc_task = 1 if task['id'] == alloc_task[i] else 0
                                    
                    if task['id'] != 0:
                        if task['id'] != alloc_task[i]:
                            #print(task['current_reqs'],task['alloc_reqs'], agent_caps[i])
                            reqs_result = task['current_reqs']  - (task['alloc_reqs'] + agent_caps[i])
                            
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
                                        
                    batch_tasks.append([
                        agent_type[i] / 4, #1
                        distance,      #1
                        sin_theta,     #1
                        cos_theta,     #1                        
                        task['init_time'], #1
                        task['end_time'],   #1
                        is_alloc_task,    #1                                                
                        ])
                                    
                    batch_tasks[-1].extend(reqs_result)
                
                # Pad the task_values array to match the maximum number of tasks                
                num_padding_needed = self.max_tasks - len(batch_tasks)
                padding = [[-1] * self.task_size for _ in range(num_padding_needed)]
                batch_tasks.extend(padding)
                task_values.append(batch_tasks)
               
        tasks_info = torch.tensor(np.array(task_values), dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor                 
        task_embeddings = self.task_encoder(tasks_info)

        # Create the mask based on your tasks_info tensor
        #mask = tasks_info.ne(-1).any(dim=-1).bool()        
        #attn_mask = ~mask                                       
        # Expected shape is (L, N) where L is target sequence length and N is batch size
        #attn_mask = attn_mask.unsqueeze(1).expand(-1, tasks_info.size(1), -1)
        #attn_mask = attn_mask.repeat(self.nhead, 1, 1)
        #attn_mask = torch.tensor(obs["mask"], dtype=torch.bool).to(self.device)
        attn_mask = None#~attn_mask
                    
        attention_output1, _ = self.own_attention(task_embeddings, task_embeddings, task_embeddings, key_padding_mask=attn_mask)
        attention_output1 = attention_output1 + task_embeddings        
        attention_output1 = self.norm1(attention_output1)

        attention_output2, _ = self.decoder_attention(attention_output1, attention_output1, attention_output1, key_padding_mask=attn_mask)
        attention_output2 = attention_output2 + attention_output1        
        attention_output2 = self.norm2(attention_output2)                                                                    

        output = self.output(attention_output2)     
        
        softmax_output = F.softmax(output, dim=1) 
        softmax_output = torch.squeeze(softmax_output, -1).to(self.device)                    

        return softmax_output, state   

    
    def _one_hot(self, idx, num_classes):
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[idx] = 1
        return one_hot_vector   
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
