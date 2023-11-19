import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict
from tianshou.utils.net.common import Net
import numpy as np
import torch.nn.functional as F


class DNN_LOTZ(Net):
    def __init__(
        self,
        obs_shape: int,        
        action_shape: int,        
        device: str,               
    ):
        super().__init__(  
            state_shape=obs_shape.shape[0],                      
            action_shape=action_shape.n,            
            device=device,
        )               
        
        agent_observation_size = obs_shape.shape[0]
   
        action_shape_size = action_shape.n
                
        self.random_weights = False  

        ref_size = agent_observation_size        
        
        self.scene_encoder = nn.Sequential(
            nn.Linear(ref_size, ref_size*4),
            nn.ReLU(),            
            nn.Linear(ref_size*4, ref_size*6),
            nn.ReLU(),
            nn.Linear(ref_size*6, ref_size*6),
            nn.ReLU(),
            nn.Linear(ref_size*6, ref_size*4),
            nn.ReLU(),
            nn.Linear(ref_size*4, action_shape_size)
        ).to(device)

        

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):

        obs_sequence = obs["agent0"]
        # Convert task_values (which contains only valid tasks) to tensor
        output = self.scene_encoder(torch.tensor(np.array(obs_sequence), dtype=torch.float32).to(self.device))
        
        return output, state   

    
