from collections import deque, namedtuple
import random
import torch
import numpy as np

class ReplayBuffer:
    """Simle experience replay buffer for deep reinforcement algorithms."""
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.device = device
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=0)).float().to(self.device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None], axis=0)).float().to(self.device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None], axis=0)).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis=0)).float().to(self.device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None], axis=0).astype(np.uint8)).float().unsqueeze(-1).to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)