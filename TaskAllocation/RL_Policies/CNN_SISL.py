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
        # self.conv_net = nn.Sequential(
        #     # Assuming obs_shape is (height, width, channels)
        #     nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=1, padding=1),  # Output: (batch_size, 32, height, width)
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 32, height/2, width/2)
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (batch_size, 64, height/2, width/2)
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=1),  # Output: (batch_size, 64, height/2-1, width/2-1)
        #     nn.Flatten(),  # Flatten the output for the linear layer
        # )

        

        # # Calculate the size of the flattened features after convolution and pooling
        # conv_height = (obs_shape[0] // 2) - 1
        # conv_width = (obs_shape[1] // 2) - 1
        # linear_input_size = conv_height * conv_width * 64

        # self.fc_net = nn.Sequential(
        #     nn.Linear(linear_input_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_shape)
        # ).to(device)

    def forward(self, obs, state=None, info=None):
        # Permute the input dimensions to (batch_size, channels, height, width)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        obs_permuted = obs_tensor.permute(0, 3, 1, 2)        
        model_out = self.model(obs_permuted)
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()
        
        

        conv_output = self.conv_net(obs_permuted)
   
        output = self.fc_net(conv_output)
        
        
        return output, state


    
