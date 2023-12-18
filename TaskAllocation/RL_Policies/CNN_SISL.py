import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict
from tianshou.utils.net.common import Net
import numpy as np
import torch.nn.functional as F


class CNN_SISL(Net):
    def __init__(self, obs_shape, action_shape, device):
        
        super().__init__(state_shape=0, action_shape=action_shape, device=device)
                
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(1024, 64)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, 5)
        self.value_fn = nn.Linear(64, 1)
     

    def forward(self, obs, state=None, info=None):
        # Permute the input dimensions to (batch_size, channels, height, width)        

        current_obs = obs[:, :1, :, :]  # Current obs is the first channel
        current_obs = current_obs.squeeze(1)
        obs_tensor = torch.tensor(current_obs, dtype=torch.float32).to(self.device)
        obs_permuted = obs_tensor.permute(0, 3, 1, 2)        
        
        model_out = self.model(obs_permuted)
        
        self._value_out = self.value_fn(model_out)
        
        # print("policy_output", self.policy_fn(model_out).shape)
        # print("value_output", state.shape)
              
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()
    
    def freeze_conv_layers(self):
            for param in self.model.parameters():
                param.requires_grad = False