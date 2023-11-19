import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import Net
from typing import Optional, Any, Dict
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHead_LOTZ(Net):
    def __init__(
        self,
        obs_shape: int,        
        action_shape: int,        
        device: str,
        d_model: int = 4,  # Dimension of the model
        nhead: int = 1,  # Number of attention heads
    ):
        super().__init__(state_shape=obs_shape.shape[0], action_shape=action_shape.n, device=device)
        self.d_model = d_model

        # Embedding layer for binary input
        self.embedding = nn.Embedding(2, d_model)  # 2 for binary input (0 and 1)

        # Multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(d_model, nhead)

        self.positional_encoding = PositionalEncoding(d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_shape.n)  # Ensure this matches the action space
        ).to(device)

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        obs_sequence = obs["agent0"]
        obs_tensor = torch.tensor(np.array(obs_sequence), dtype=torch.long).to(self.device)

        # Embedding and reshaping for Multihead Attention
        obs_tensor = self.embedding(obs_tensor).transpose(0, 1)  # Shape: [sequence_length, batch_size, d_model]

        # Adding positional encoding
        obs_tensor = self.positional_encoding(obs_tensor)

        # Applying multihead attention
        attention_output, _ = self.multihead_attention(obs_tensor, obs_tensor, obs_tensor)

        # Decoding the output for the final prediction
        output = self.decoder(attention_output[-1])  # Assuming you want the output corresponding to the last input

        return output, state
