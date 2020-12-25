#https://www.researchgate.net/publication/320296763_Recurrent_Network-based_Deterministic_Policy_Gradient_for_Solving_Bipedal_Walking_Challenge_on_Rugged_Terrains
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random

# https://github.com/vy007vikas/PyTorch-ActorCriticRL

EPS = 0.003

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class NormalizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, batch_first=True, bidirectional=False, num_layers=1):
        super(NormalizedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, batch_first=True, bidirectional=False, num_layers=1)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x[:, -1]
        x = self.layernorm(x)
        return x

class Critic(nn.Module):

    def __init__(self, state_dim=24, action_dim=4):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = NormalizedLSTM(input_size=self.state_dim, hidden_size=128, batch_first=True, bidirectional=False, num_layers=1)

        self.action_encoder = nn.Sequential(nn.Linear(self.action_dim, 128), nn.ReLU())
        self.action_encoder[0].weight.data = fanin_init(self.action_encoder[0].weight.data.size())
        
        self.fc1 = nn.Linear(256,128)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(128,1)
        self.fc2.weight.data.uniform_(-EPS,EPS)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s = self.state_encoder(state)
        a = self.action_encoder(action)
        x = torch.cat((s,a),dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)*100
        return x


class Actor(nn.Module):

    def __init__(self, state_dim=24, action_dim=4):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = nn.LSTM(input_size=self.state_dim, hidden_size=128, batch_first=True, bidirectional=False, num_layers=1)

        self.fc = nn.Linear(128,action_dim)
        self.fc.weight.data.uniform_(-EPS,EPS)
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        s, (_, _) = self.state_encoder(state)
        s = s[:, -1]
        action = self.tanh(self.fc(s))
        return action