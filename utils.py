# utils.py

import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_experience(self, state, action, reward, next_state, done):
        """Saves a (s, a, r, s') tuple."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        """Samples a random batch of experiences from memory."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)