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

class CustomNetSimple(Net):
    def __init__(
        self,
        state_shape_agent: int,
        state_shape_task: int,
        action_shape: int,
        hidden_sizes: List[int],
        device: str,
        nhead: int = 4,
        num_layers: int = 1,
    ):
        super().__init__(            
            state_shape=0,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            device=device,
        )
                                       
        
        # Drone Encoder
        self.drone_encoder = nn.Sequential(
            nn.Linear(state_shape_agent, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ).to(device)

        # Task Encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(state_shape_task, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(device)
                       
        # Task Encoder
        self.DNN = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(device)
        

        # Output Layer
        self.output = nn.Linear(256, action_shape).to(device)
                
        

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
                   
            #print("agent pos: "", agent_position.shape")
            # Drone embeddings: input (12) from agent_obs | output (32) to combined_output
            agent_position = torch.tensor(obs["agent_position"], dtype=torch.float32).to(self.device)
            agent_state = torch.tensor(obs["agent_state"], dtype=torch.float32)
            agent_type = torch.tensor(obs["agent_type"], dtype=torch.float32).to(self.device)
            next_free_time = torch.tensor(obs["next_free_time"], dtype=torch.float32).to(self.device)
            position_after_last_task = torch.tensor(obs["position_after_last_task"], dtype=torch.float32).to(self.device)         
            #drone_embeddings = self.drone_encoder(torch.cat((agent_position, agent_state, agent_type, next_free_time, position_after_last_task), dim=-1))
            
            tasks_info = torch.tensor(obs["tasks_info"], dtype=torch.float32).to(self.device)  # Convert tasks_info to tensor                        
            task_embeddings = self.task_encoder(tasks_info)

            # print("agent pos: ", agent_position.shape)
            
            # print("task info: ", tasks_info.shape)
            # print("task_embeddings: ", task_embeddings.shape)
            
            # Combine drone and task embeddings
            #combined_output = torch.cat((drone_embeddings, task_embeddings), dim=1)
                    
            #print("combined_output info: ", combined_output.shape)

            # Process the combined output with the hidden layers
            x = self.DNN(task_embeddings)            
            # print("X", x.shape)
            output = self.output(x)
            # print("OUt->", output.shape)
            # Apply the softmax function
            softmax_output = torch.nn.functional.softmax(output, dim=-1) 
            softmax_output = softmax_output.unsqueeze(1)
            # softmax_output = torch.squeeze(softmax_output, -1)
            # # print("SOFT->", softmax_output.shape)
            #print("softmax_output_Final:" , softmax_output.shape)     

            #rint(softmax_output.shape)
        

            return softmax_output, state




