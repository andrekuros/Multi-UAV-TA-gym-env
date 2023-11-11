import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict
from tianshou.utils.net.common import Net
import numpy as np
import torch.nn.functional as F
from mUAV_TA.MultiDroneEnvData import SceneData


class CriticNetMultiHead(Net):
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

        self.embedding_size = 128 #sum of drone and task encoder  

        self.max_tasks = 31
        self.task_size = 2 + 6 #int(state_shape_task / self.max_tasks)

        self.max_agents = 20
        self.agent_size = 5 
        
        self.random_weights = False  

        self.sceneData = SceneData()                                                                 
                               
        self.task_encoder = nn.Sequential(
            nn.Linear(self.task_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_size)
        ).to(device)

        #self.dropout = nn.Dropout(0.1) # 0.5 is the dropout probability
        
        self.own_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=self.nhead, batch_first=True).to(device)                
        self.norm1 = nn.LayerNorm(self.embedding_size).to(device)
               
        self.decoder_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead, batch_first=True).to(device)    
        self.norm2 = nn.LayerNorm(self.embedding_size).to(device)
     
        # self.output = nn.Linear(self.embedding_size, 1).to(device) 

        self.output = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

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
        # agent_type      = obs["agent_type"]
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
                            #reqs_result = task['current_reqs']  - (task['alloc_reqs'] + agent_caps[i])
                            
                            missingCap = np.maximum( task['current_reqs'] - (task['alloc_reqs'] ), 0)                                                            
                                
                            reqs_result = missingCap -  np.maximum(missingCap - agent_caps[i], 0) #AddedCap

                            # reqs_result = agent_caps[i] 

                           
                        else:
                            missingCap = np.maximum( task['current_reqs'] - (task['alloc_reqs'] - agent_caps[i] ), 0)                                                            
                                
                            reqs_result = missingCap -  np.maximum(missingCap - agent_caps[i], 0) #AddedCap
                            
                            #reqs_result = task['current_reqs']  - task['alloc_reqs'] 
                            #  reqs_result =  np.full(agent_caps[i].size, 0.05)
                    else:
                        reqs_result = np.full(agent_caps[i].size, -0.03)
                        # task['init_time'] = 0.5
                        # task['end_time'] = 0.5
                        distance = 1.0
                        theta = 0.0
                        sin_theta = 0.0
                        cos_theta = 0.0
                                        
                    batch_tasks.append([
                        # agent_type[i] / 4, #1
                        distance,      #1
                        sin_theta,     #1
                        cos_theta,     #1                        
                        # task['init_time'], #1
                        # task['end_time'],   #1
                        is_alloc_task,    #1                                                
                        ])
                                    
                    batch_tasks[-1].extend(reqs_result)
                
                # Pad the task_values array to match the maximum number of tasks                
                # num_padding_needed = self.max_tasks - len(batch_tasks)
                # padding = [[-1] * self.task_size for _ in range(num_padding_needed)]
                # batch_tasks.extend(padding)
                task_values.append(batch_tasks)
               
        
        # attn_mask = torch.tensor(obs["mask"], dtype=torch.bool).to(self.device) 
        # attn_mask = ~attn_mask                 
        
        # Convert tasks_info to tensor
        # Convert tasks_info to tensor
        # tasks_info = torch.tensor(np.array(task_values), dtype=torch.float32).to(self.device)

        # Convert task_values (which contains only valid tasks) to tensor
        valid_task_embeddings = self.task_encoder(torch.tensor(np.array(task_values), dtype=torch.float32).to(self.device))

        # Initialize a tensor for all tasks, filled with zeros
        # Assuming the first dimension of valid_task_embeddings is the batch size
        all_task_embeddings = torch.zeros((valid_task_embeddings.shape[0], self.max_tasks, self.embedding_size), device=self.device)

        # Create a boolean mask for valid tasks in the batch
        # Assuming obs["mask"] contains a mask for each batch where True indicates a valid task
        batch_mask = torch.tensor(obs["mask"], dtype=torch.bool).to(self.device)

        # Copy the valid task embeddings into the all_task_embeddings tensor using the batch mask
        for b in range(valid_task_embeddings.shape[0]):  # Loop over the batch
            num_valid_tasks = batch_mask[b].sum()  # Count the number of valid tasks for this batch entry
            all_task_embeddings[b, :num_valid_tasks] = valid_task_embeddings[b, :num_valid_tasks]

        # attn_mask should be the same size as the second dimension of all_task_embeddings and should indicate padding with True
        attn_mask = ~batch_mask

        attention_output1, _ = self.own_attention(all_task_embeddings, all_task_embeddings, all_task_embeddings, key_padding_mask=attn_mask)               
        attention_output1 = attention_output1 + all_task_embeddings        
        attention_output1 = self.norm1(attention_output1)
        mask_expanded = attn_mask.unsqueeze(-1).expand_as(attention_output1)               
        attention_output1 = attention_output1.masked_fill(mask_expanded, 0.0)

        attention_output2, _ = self.decoder_attention(attention_output1, attention_output1, attention_output1, key_padding_mask=attn_mask)               
        attention_output2 = attention_output2 + attention_output1        
        attention_output2 = self.norm2(attention_output2)
        attention_output2 = attention_output2.masked_fill(mask_expanded, 0.0)
                            
        output = self.output(attention_output2)                   
        output = torch.squeeze(output, -1).to(self.device)    

        # print("output", output[0])     
        return output, state   

    
    def _one_hot(self, idx, num_classes):
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[idx] = 1
        return one_hot_vector   
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
    
