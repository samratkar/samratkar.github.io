import random
from collections import deque 
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        experience_tuple = (state, action, reward, next_state, done)
        self.memory.append(experience_tuple)

    def __len__(self):
        return len(self.memory)
    
    def sample(self, batch_size):
        # draw a random sample of size batch_size
        batch = random.sample(self.memory, batch_size)
        # transform the batch into tuple of lists
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        rewards_tensor = torch.as_tensor(np.asarray(rewards), dtype=torch.float32)
        next_states_tensor = torch.as_tensor(np.asarray(next_states), dtype=torch.float32)
        dones_tensor = torch.as_tensor(np.asarray(dones), dtype=torch.float32)
        # ensure actions_tensor has shape (batch_size, 1) for gathering
        actions_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.long).unsqueeze(1)
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
