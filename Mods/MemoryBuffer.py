from typing import Any, List, Optional, Union
import numpy as np
from tianshou.data import Batch, VectorReplayBuffer, ReplayBufferManager, HERReplayBuffer, ReplayBuffer
import torch
from collections import deque
from typing import Tuple, Dict


class StateMemoryVectorReplayBuffer(VectorReplayBuffer):
    def __init__(self, total_size: int, buffer_num: int, memory_size: int, **kwargs: Any):
        super().__init__(total_size, buffer_num, **kwargs)
        self.memory_size = memory_size
        # State memory storage for each sub-buffer
        self.state_memory = [deque(maxlen=memory_size) for _ in range(buffer_num)]

    def add(self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None, state_memory: Optional[List[np.ndarray]] = None):    
        # Call the super().add method and store its return value
        add_info = super().add(batch, buffer_ids)

        if state_memory is not None:
            for i, buffer_id in enumerate(buffer_ids or range(len(batch))):
                self.state_memory[buffer_id].append(state_memory[i])

        # Return the information received from the super().add call
        return add_info

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        sampled_data, indices = super().sample(batch_size)
        # Retrieve state memory for the sampled indices
        sampled_memory = []
        for idx in indices:
            buffer_id, buffer_idx = self._get_buffer_id(idx)
            sampled_memory.append(self.state_memory[buffer_id][buffer_idx])
        sampled_data.state_memory = sampled_memory
        return sampled_data, indices

    def _get_buffer_id(self, index: int) -> Tuple[int, int]:
        # Find which buffer and index within the buffer a global index corresponds to
        for i, offset in enumerate(self._offset):
            if index < offset + self.buffers[i].maxsize:
                return i, index - offset
        raise IndexError("Index out of range.")



    

from tianshou.trainer import OffpolicyTrainer


class MemoryOffpolicyTrainer(OffpolicyTrainer):

    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Perform off-policy updates."""
        assert self.train_collector is not None
        for _ in range(round(self.update_per_step * result["n/st"])):
            self.gradient_step += 1
            # Sample a batch of experiences, including state memory
            batch, indices = self.train_collector.buffer.sample(self.batch_size)
            state_memory = batch.state_memory  # Assuming state memory is stored in the batch

            # Update the policy with the sampled batch and state memory
            losses = self.policy.update(batch, state_memory=state_memory)

            # Log losses and update statistics
            self.log_update_data(data, losses)


