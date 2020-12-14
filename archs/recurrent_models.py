import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random

#https://www.researchgate.net/publication/320296763_Recurrent_Network-based_Deterministic_Policy_Gradient_for_Solving_Bipedal_Walking_Challenge_on_Rugged_Terrains

class Actor(nn.Module):
    def __init__(self, state_size=24, action_size=4, fc_layer=128):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size= action_size
        self.bn = nn.BatchNorm1d(state_size)
        self.lstm = nn.LSTM(state_size, fc_layer, batch_first=True, bidirectional=False) 
        self.linear= nn.Linear(fc_layer, action_size)
        self.tanh = nn.Tanh()

    def forward(self, state):
        # state: batch x seq x observation
        #x = self.bn(state)
        x = state
        x, (_, _) = self.lstm(x)
        x = self.tanh(self.linear(x[:,0]))
        return x

class Critic(nn.Module):
    def __init__(self, state_size=24, action_size=4, fc1_layer=128, fc2_layer=128):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size= action_size
        self.bn = nn.BatchNorm1d(state_size)
        self.lstm = nn.LSTM(state_size, fc1_layer, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(fc1_layer+action_size, fc2_layer)
        self.linear2 = nn.Linear(fc2_layer, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        # state: batch x seq x observation
        x = state
        x, (_, _) = self.lstm(x)
        x = torch.cat((x[:,0], action), dim=1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x    
