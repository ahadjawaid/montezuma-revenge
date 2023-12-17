import random
import torch

class ReplayBuffer:
    def __init__(self, batch_size: int, min_size: int, max_size: int) -> None:
        self.batch_size = batch_size
        self.min_size = min_size
        self.max_size = max_size
        self._buffer = []
        self.size = 0
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index: int) -> list:
        return self._buffer[index]
    
    def __repr__(self) -> str:
        return f"ReplayBuffer({self._buffer})"
    
    def is_ready(self) -> bool:
        return self.size >= self.min_size

    def add(self, item) -> None:
        if self.size < self.max_size:
            self._buffer.append(item)
            self.size += 1
        else:
            self._buffer[0] = item

    def sample(self) -> list:
        if self.is_ready():
            sample = random.sample(self._buffer, self.batch_size)
            return list(map(lambda x: torch.stack(x), zip(*sample)))

        raise ValueError(f"Replay buffer is not ready for sampling, current size: {self.size}, batch size: {self.batch_size}")