import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import Net

class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=1):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, x):
        # x shape: (batch_size, seq_length, feature_dim), but MultiheadAttention expects (seq_length, batch_size, feature_dim)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2)  # Permute back to (batch_size, seq_length, feature_dim)

class CNN_ATT_SISL(Net):
    def __init__(self, obs_shape, action_shape, device):
        super().__init__(state_shape=0, action_shape=action_shape, device=device)
        
        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Define self-attention layer
        self.attention_layer = SelfAttention(feature_dim=64, num_heads=1)

        # Define policy and value function layers
        self.policy_fn = nn.Linear(64, action_shape)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, obs, state=None, info=None):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        obs_permuted = obs_tensor.permute(0, 3, 1, 2)
        
        # Pass through convolutional layers
        conv_out = self.conv_layers(obs_permuted)
        
        # Pass through self-attention layer
        attention_out = self.attention_layer(conv_out.unsqueeze(1))  # Unsqueezing since we expect a sequence
        attention_out = attention_out.squeeze(1)  # Squeezing back to original shape
        
        # Pass through policy and value function layers
        self._value_out = self.value_fn(attention_out)
        return self.policy_fn(attention_out), state

    def value_function(self):
        return self._value_out.flatten()

# Usage Example
# model = CNN_SISL(obs_shape=(height, width, channels), action_shape=num_actions, device=torch.device('cuda'))
