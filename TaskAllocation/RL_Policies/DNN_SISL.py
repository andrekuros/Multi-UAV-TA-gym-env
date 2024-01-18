import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict
from tianshou.utils.net.common import Net
import numpy as np



class DNN_SISL(Net):
    def __init__(
        self,
        obs_shape: int,        
        action_shape: int,        
        device: str,               
    ):
        super().__init__(  
            state_shape=0,                      
            action_shape=action_shape,            
            device=device,
        )               
           
                
        self.random_weights = False          
                               
        self.scene_encoder = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        ).to(device)

        

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        # print(obs)      
                
        obs_sequence = obs.reshape(-1,192)

        # print(obs_sequence)
       
        # Convert task_values (which contains only valid tasks) to tensor
        output = self.scene_encoder(torch.tensor(np.array(obs_sequence), dtype=torch.float32).to(self.device))
                                    
        # output = torch.squeeze(output, -1).to(self.device)    
        # print("output.shape", output)     
        # print("output", output)    
        
        return output, state   

    
