from tianshou.data import VectorReplayBuffer
import numpy as np

class PrioritizedFeatureVectorReplayBuffer(VectorReplayBuffer):
    def __init__(self, size, buffer_num, alpha=0.6, beta=0.4, history_length=10, feature_size=None, **kwargs):
        super().__init__(size, buffer_num, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.history_length = history_length
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.max_priority = 1.0
        self.feature_buffer = np.zeros((size, feature_size))  # Adjust feature_size accordingly


    def add(self, data, buffer_ids=None):
        # Handle buffer_ids to correctly add data to the specific buffer
        if buffer_ids is None:
            buffer_ids = np.arange(len(data.obs))  # Default to all buffers

        for i in buffer_ids:
            idx = self._index[i]  # Current position in buffer i
            self.priorities[idx] = self.max_priority

            # Process observation through CNN here and store the feature
            # Ensure that policy.model.cnn_process is correctly implemented
            feature = policy.model.cnn_process(data.obs[i])
            self.feature_buffer[idx] = feature

            # Call add method of VectorReplayBuffer for the i-th buffer
            super(VectorReplayBuffer, self).add(
                data=data[i], buffer_ids=[i]
            )

    def sample(self, batch_size):
        # Sample based on priorities
        if self._size == 0:
            return {}
        sampling_probabilities = self.priorities ** self.alpha
        sampling_probabilities /= sampling_probabilities.sum()
        indices = np.random.choice(self._size, batch_size, p=sampling_probabilities)
        return self.get(indices)

    def get(self, index):
        # Override to return a sequence of features along with raw observations
        buffer_index = (self._index - self.history_length + self.maxsize) % self.maxsize
        index = (index - self.history_length + self.maxsize) % self.maxsize
        feature_seq = [self.feature_buffer[i] for i in range(index, buffer_index)]
        return {k: self._meta[k][index] if k != "obs" else (self._meta["obs"][index], feature_seq) for k in self._meta}

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
