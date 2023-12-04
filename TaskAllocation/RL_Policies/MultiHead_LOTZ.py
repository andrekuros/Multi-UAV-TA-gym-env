import torch
import torch.nn as nn
import numpy as np
from tianshou.utils.net.common import Net
from typing import Optional, Any, Dict

class PositionalEncoding(nn.Module):
    
    def __init__(self, max_len=12):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len

    def forward(self, x):
        # Generate positional encodings
        position = (torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1) - self.max_len/2) / self.max_len
        return position


class MultiHead_LOTZ(Net):
    def __init__(
        self,
        obs_shape: int,        
        action_shape: int,        
        device: str,
        d_model: int = 4,  # Dimension of the model
        nhead: int = 1,  # Number of attention heads
        max_len: int = 12, # Maximum length of the sequence
    ):
        super().__init__(state_shape=obs_shape.shape[0], action_shape=action_shape.n, device=device)
        self.d_model = d_model
        self.max_len = max_len

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(max_len)

        # Combined embedding layer for binary input and its position
        # self.embedding = nn.Embedding(2 * max_len, d_model)  # For each position-value pair
        self.embedding = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, d_model)  # Ensure this matches the action space
        ).to(device)


        # Multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True).to(device)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Ensure this matches the action space
        ).to(device)

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, info: Optional[Any] = None):
        obs_sequence = obs["agent0"]  # Assuming shape [batch_size, sequence_length]
        batch_size, seq_len = obs_sequence.shape

        obs_tensor = torch.tensor(obs_sequence, dtype=torch.float32).to(self.device)

        # Generate positional encodings for the sequence length
        position = self.positional_encoding(obs_tensor).to(self.device)[:, :seq_len]  # Adjust for actual seq_len
        # Replicate positional encoding for each item in the batch
        position = position.repeat(batch_size, 1, 1)  # Shape [batch_size, seq_len, 1]

        # Reshape obs_tensor for concatenation
        obs_tensor = obs_tensor.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Concatenate binary input with its positional encoding
        combined_input = torch.cat((obs_tensor, position), dim=-1)  # [batch_size, seq_len, 2]

        # Flatten combined_input for embedding, then reshape back
        flat_input = combined_input.view(-1, 2)
        embedded_input = self.embedding(flat_input)
        embedded_input = embedded_input.view(batch_size, seq_len, self.d_model)

        # Applying multihead attention
        attention_output, weights = self.multihead_attention(embedded_input, embedded_input, embedded_input)
        
#         print("weights:", weights.shape)
#         print("weights:", weights)
        
#         print("attention_output:", attention_output.shape)
#         print("attention_output:", attention_output)

        # Decoding the output for the final prediction
        output = self.decoder(attention_output)
        output = torch.squeeze(output, -1).to(self.device)

        return output, state