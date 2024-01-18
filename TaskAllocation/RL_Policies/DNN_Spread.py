import torch
import torch.nn as nn
from typing import Optional, Any, Dict
from tianshou.utils.net.common import Net
import numpy as np


class DNN_Spread(Net):
    def __init__(
        self,
        obs_shape: int,  # Update the observation shape as per MPE Spread requirements
        action_shape: int,  # Update the action shape as per MPE Spread requirements
        device: str,
    ):
        super().__init__(
            state_shape=0,  # Assuming state shape is not used
            action_shape=action_shape,
            device=device,
        )  

        self.obs_shape = obs_shape
       
        self.scene_encoder = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
           
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),  
            nn.Linear(128, 64),
            nn.ReLU(),  
            
            nn.Linear(64, 32),
            nn.ReLU(),       
            nn.Linear(32, action_shape)  # Ensure the output matches the action shape
        ).to(device)

    def forward(self, obs: torch.Tensor, state: Optional[Any] = None, info: Optional[Any] = None):

        # reduced_obs = np.delete(obs, [2, 3], axis=1)

        output = self.scene_encoder(torch.tensor(np.array(obs), dtype=torch.float32).to(self.device))
       
        # Generate random values with the same shape as the output tensor
        #random_values = torch.rand_like(output, requires_grad=True)
        # zeros = torch.zeros_like(output,  requires_grad=True)
       
       
        return output, state
