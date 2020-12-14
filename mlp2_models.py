import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random

class Actor(nn.Module):
    def __init__(self, state_size=24, action_size=4, fc_layer=128):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size= action_size
        self.Layer_in = nn.Linear(state_size, fc_layer)
        self.Layer_out= nn.Linear(fc_layer, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = state
        x = self.relu(self.Layer_in(x))
        x = self.tanh(self.Layer_out(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_size=24, action_size=4, fc1_layer=256, fc2_layer=128):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size= action_size
        self.Layer_1 = nn.Linear(state_size, fc1_layer)
        self.Layer_2 = nn.Linear(fc1_layer+action_size, fc2_layer)
        self.Layer_3 = nn.Linear(fc2_layer, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = state
        x = self.relu(self.Layer_1(x))
        x = torch.cat((x, action), dim=1)
        x = self.relu(self.Layer_2(x))
        x = self.Layer_3(x)
        return x    
