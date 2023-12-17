import numpy as np
import torch
from typing import Sequence

class EpsilonGreedyExploration:
    def __init__(self, epsilon: float = None, decay_schedule: Sequence = None):
        self.epsilon = epsilon
        self.step = 0
        self.decay_schedule = decay_schedule

    def _get_decay_step(self):
        if self.decay_schedule is not None:
            epsilon = self.decay_schedule[self.step]
            self.step += 1
            return epsilon
 
        return self.epsilon

    def get_action(self, q_values: torch.Tensor):
        if self.decay_schedule is not None:
            self.epsilon = self._get_decay_step()

        if np.random.rand() < self.epsilon:
            return np.random.randint(q_values.shape[0])
        
        return q_values.argmax()
    

def quadratic_decay_schedule(initial_epsilon: float, final_epsilon: float, max_step: int) -> Sequence:
    return np.linspace(initial_epsilon **2 , final_epsilon **2, max_step) **0.5

def linear_decay_schedule(initial_epsilon: float,  final_epsilon: float,  max_step: int) -> Sequence:
     return np.linspace(initial_epsilon, final_epsilon, max_step)
