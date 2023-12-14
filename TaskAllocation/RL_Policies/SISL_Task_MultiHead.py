import torch
import torch.nn as nn

class SISL_Task_MultiHead(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        num_features_per_task: int,
        device: str,
        nhead: int = 4,
    ):
        super().__init__()

        self.device = device
        self.num_tasks = num_tasks  # Number of tasks (e.g., 20)
        self.embedding_size = 64  # Customizable

        # Task encoder: Fully connected layers
        # Each task has num_features_per_task feature      

        self.task_encoder = nn.Sequential(
            nn.Linear(num_features_per_task, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
             nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)  # Ensure this matches the action space
        ).to(device)

        # Multi-head attention layers
        self.own_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead, batch_first=True).to(device)
        # self.norm1 = nn.LayerNorm(self.embedding_size).to(device)

        # self.decoder_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=nhead, batch_first=True).to(device)
        # self.norm2 = nn.LayerNorm(self.embedding_size).to(device)
        # Output layer to produce values for each task
        self.output = nn.Linear(self.embedding_size, 1).to(device)
      

    def forward(self, obs: torch.Tensor, state=None, info=None):
        # obs shape is expected to be [batch_size, num_tasks, num_features_per_task]
       
        # Process each task independently through the task encoder
        batch_size, num_tasks, num_features = obs.shape
        obs_reshaped = obs.view(-1, num_features)  # [batch_size * num_tasks, num_features]

        # print("obs_reshaped", obs_reshaped)

        task_embeddings = self.task_encoder(obs_reshaped)  # [batch_size * num_tasks, embedding_size]
        task_embeddings = task_embeddings.view(batch_size, num_tasks, -1)  # Reshape back to [batch_size, num_tasks, embedding_size]

        # print("task_embeddings", task_embeddings)

        # Attention mechanism
        attention_output1, weights = self.own_attention(task_embeddings, task_embeddings, task_embeddings)
        # attention_output1 = self.norm1(attention_output1 + task_embeddings)

        # print("attention_output1", attention_output1)
        # print("weights", weights)
        

        # attention_output2, _ = self.decoder_attention(attention_output1, attention_output1, attention_output1)
        # attention_output2 = self.norm2(attention_output2 + attention_output1)

        # Output layer: Produces a vector of Q-values for each task
        # task_q_values = self.output(attention_output1)#.mean(dim=1))

        task_q_values = self.output(attention_output1)                   
        task_q_values = torch.squeeze(task_q_values, -1).to(self.device)   

        # print("task_q_values.shape",task_q_values.shape)
        # print("task_q_values",task_q_values)

        # Assuming the last feature in the feature vector is the desired action
        #desired_actions = obs[:, :, -1].long()  # Extracting desired actions

        # # Initialize Q-values for each action
        # action_q_values = torch.zeros((batch_size, 5), device=self.device)  # Assuming 5 actions

        # for action in range(5):  # For each action
        #     # Mask to select only tasks with the current action as desired
        #     action_mask = (desired_actions == action)

        #     # Gather Q-values for tasks with the current action, apply the mask
        #     q_values_for_action = task_q_values.masked_fill(~action_mask, float('-inf'))

        #     # Take the max Q-value for each instance in the batch
        #     max_q_values, _ = torch.max(q_values_for_action, dim=1)

        #     # Assign the max Q-values to the corresponding action            
        #     action_q_values[:, action] = max_q_values
        #     action_q_values[:, action] = torch.where(action_q_values[:, action] == -float('inf'), torch.tensor(-1000.0), action_q_values[:, action])

        # # print("action_values", action_q_values)

        # return action_q_values, state
        
        #print("task_qvalues.shape", task_q_values.shape)

        return task_q_values, state
