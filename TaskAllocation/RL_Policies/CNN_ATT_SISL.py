import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
import numpy as np

def positional_encoding(seq_len, d_model, device):
        """
        Generates and returns positional encodings.
        
        :param seq_len: Length of the state memory sequence.
        :param d_model: The dimension of the embeddings/feature vectors.
        :param device: The device to which tensors will be sent.
        :return: A tensor of shape [seq_len, d_model] representing the positional encodings.
        """
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return torch.tensor(pe, dtype=torch.float32).to(device)


class CNN_ATT_SISL(Net):
    def __init__(self, obs_shape, action_shape, device):
        super().__init__(state_shape=0, action_shape=action_shape, device=device)
        
        # CNN model for processing current state
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
        )

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True).to(device)
        self.attention_fc = nn.Linear(64, 64).to(device)

        # Decision making layers
        self.policy_fn = nn.Linear(64, 5).to(device)
        self.value_fn = nn.Linear(64, 1).to(device)

    
    def forward(self, obs, state=None, info=None):
        # Convert obs to tensor if it is a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        # Extract current observation and state memory from the combined observation
        current_obs = obs[:, :1, :, :]  # Current obs is the first channel
        state_memory = obs[:, 1:, :, :]  # State memory are the subsequent channels

        # Generate positional encodings
        seq_len = state_memory.shape[1]  # Assuming state_memory shape is [batch, seq_len, height, width, channels]
        d_model = 64  # Assuming feature size after CNN is 64
        pos_encodings = positional_encoding(seq_len, d_model, self.device)


        # Remove the channel dimension from current_obs since it's only one channel
        current_obs = current_obs.squeeze(1)
        # Process current observation through CNN
        current_obs_permuted = current_obs.permute(0, 3, 1, 2) 
        # print(" current_obs_permuted" , current_obs_permuted.shape)  
        
        current_state_features = self.cnn_model(current_obs_permuted).unsqueeze(1)
        # print("current_state_features.shape", current_state_features.shape)
        
        # Process each state memory observation through the CNN
        # print(" MemoryShape" , state_memory.shape)
        batch_size, memory_channels, height, width, channels = state_memory.shape        
        state_memory = state_memory.reshape(-1, channels, height, width)  # Flatten memory observations for batch processing
        # print(" MemoryShape2" , state_memory.shape)
        state_memory_permuted = state_memory.permute(0, 1, 3, 2)  # Adjust dimensions to match CNN input (N, C, H, W)
        # state_memory_permuted = state_memory.permute(0, 3, 1, 2)  # Adjust dimensions to match CNN input
        # print(" MemoryShape3" , state_memory_permuted.shape)
        

        # Process through CNN
        state_memory_features = self.cnn_model(state_memory_permuted)

        # Reshape back to original memory shape with features
        state_memory_features = state_memory_features.view(batch_size, memory_channels, -1)
        
        # # Add positional encodings to state memory features
        # state_memory_features = state_memory_features.view(batch_size, seq_len, -1)
        # print("state_memory_features.shape", state_memory_features.shape, state_memory_features)
        state_memory_features = state_memory_features + pos_encodings.unsqueeze(0)
        
        # print("state_memory_features.shape", state_memory_features.shape, state_memory_features)

        # Apply attention
        attn_output, weights = self.attention(current_state_features, state_memory_features, state_memory_features)
        attn_output = self.attention_fc(attn_output.squeeze(0))

        # print("weights:", weights.shape)
        # Combine CNN output and attention output
        combined_output = current_state_features + attn_output
        # print("att_out",attn_output.shape)
        # Compute policy and value
        policy_output = self.policy_fn(combined_output).squeeze(1)
        # policy_output = torch.squeeze(policy_output, -1).to(self.device) 

        value_output = self.value_fn(combined_output)

        # print("policy_output", policy_output.shape)
        # print("policy_output", policy_output)
        # print("value_output", value_output.shape)

        return policy_output, None#value_output.squeeze(0)  # Adjust dimensions if necessary


    def value_function(self):
        return self._value_out.flatten()
    
    
