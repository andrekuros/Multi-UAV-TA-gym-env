import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import Net
from typing import Optional, Any, Dict
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        attention_scores = torch.tanh(self.attention_weights(x))
        attention_weights = F.softmax(self.context_vector(attention_scores), dim=1)
        weighted_sum = torch.sum(attention_weights * x, dim=1)
        return weighted_sum

class ATT_LOTZ(Net):
    def __init__(
        self,
        obs_shape: int,        
        action_shape: int,        
        device: str,  
        attention_dim: int = 64,  # Dimension for the attention layer
    ):
        super().__init__(  
            state_shape=obs_shape.shape[0],                      
            action_shape=action_shape.n,            
            device=device,
        )               
        
        agent_observation_size = obs_shape.shape[0]
        action_shape_size = action_shape.n
        
        self.attention_layer = AttentionLayer(agent_observation_size, attention_dim)

        # Define the rest of the network
        self.fc_layers = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 2),
            nn.ReLU(),            
            nn.Linear(attention_dim * 2, attention_dim * 2),
            nn.ReLU(),
            nn.Linear(attention_dim * 2, action_shape_size)
        ).to(device)

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        obs_sequence = obs["agent0"]
        # Convert task_values to tensor
        obs_tensor = torch.tensor(np.array(obs_sequence), dtype=torch.float32).to(self.device)

        # Apply attention mechanism
        attention_output = self.attention_layer(obs_tensor)

        # Pass through the rest of the network
        output = self.fc_layers(attention_output)
        
        return output, state
