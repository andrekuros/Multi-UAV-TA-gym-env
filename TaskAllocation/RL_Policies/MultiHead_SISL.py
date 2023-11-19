import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict
from tianshou.utils.net.common import Net
import numpy as np
import torch.nn.functional as F
from mUAV_TA.MultiDroneEnvData import SceneData


class MultiHead_SISL(Net):
    def __init__(
        self,
        obs_shape: int,        
        action_shape: int,        
        device: str,
        nhead: int = 8,        
    ):
        super().__init__(  
            state_shape=0,                      
            action_shape=action_shape,            
            device=device,
        )               

        self.nhead = nhead
        
        self.embedding_size = 128 #sum of drone and task encoder  
                
        self.random_weights = False          
                               
        self.task_encoder = nn.Sequential(
            nn.Linear(3, 64),
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
         
        self.decoder_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=self.nhead, batch_first=True).to(device)    
        self.norm2 = nn.LayerNorm(self.embedding_size).to(device)
     
        self.output = nn.Linear(self.embedding_size, 1).to(device) 

        # self.output = nn.Sequential(
        #     nn.Linear(self.embedding_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # ).to(device)

        self.reencoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
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
        # print(obs)      
        # print(obs.shape)
        
        obs_sequence = obs.reshape(-1,9,3)
       
        # Convert task_values (which contains only valid tasks) to tensor
        all_task_embeddings = self.task_encoder(torch.tensor(np.array(obs_sequence), dtype=torch.float32).to(self.device))
                
        attention_output1, _ = self.own_attention(all_task_embeddings, all_task_embeddings, all_task_embeddings)               
        attention_output1 = attention_output1 + all_task_embeddings        
        attention_output1 = self.norm1(attention_output1)
        
        attention_output2, _ = self.decoder_attention(attention_output1, attention_output1, attention_output1)               
        attention_output2 = attention_output2 + attention_output1        
        attention_output2 = self.norm2(attention_output2)  
                            
        output = self.output(attention_output2)                          
        
        output = torch.squeeze(output, -1).to(self.device)    

        output = self.reencoder(output)

         # print("output.shape", output)     
        #  print("output", output.shape)     
        return output, state   

    
