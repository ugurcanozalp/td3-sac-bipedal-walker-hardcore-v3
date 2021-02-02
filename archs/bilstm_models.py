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

class LastStatePooler(nn.Module):
    def forward(self,x):
        return x[:, -1]

class MaxPooler(nn.Module):
    def forward(self,x):
        x, _ = x.max(axis=-2) # -2 -> sequence dimension
        return x 

class NormalizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=96, batch_first=True, dropout=0.1):
        super(NormalizedLSTM, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        nn.init.xavier_uniform_(self.embedding[0].weight, gain=nn.init.calculate_gain('tanh'))
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size=hidden_size, batch_first=batch_first, bidirectional=True, num_layers=1, dropout=dropout)
        self.pooler = LastStatePooler()

    def forward(self, x):
        x = self.embedding(x)
        #h = torch.stack((x[:,0], x[:,-1])).contiguous()
        #c = torch.zeros_like(h).contiguous()
        #x, (_, _) = self.lstm(x, (h, c))
        x = self.dropout(x)
        x, (_, _) = self.lstm(x)
        x = self.pooler(x)
        x = 0.5*(x[:, :self.lstm.hidden_size]+x[:, self.lstm.hidden_size:])
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

        self.state_encoder = NormalizedLSTM(input_size=self.state_dim, hidden_size=96, batch_first=True, dropout=0.0)
        self.action_encoder = nn.Sequential(nn.Linear(self.action_dim, 96), nn.Tanh())
        nn.init.xavier_uniform_(self.action_encoder[0].weight, gain=nn.init.calculate_gain('tanh'))

        self.fc2 = nn.Linear(96,256)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        
        self.fc_out = nn.Linear(256,1)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        self.act = nn.GELU()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s = self.state_encoder(state)
        a = self.action_encoder(action)
        #x = torch.cat((s,a),dim=1)
        x = s + a
        x = self.act(self.fc2(x))
        x = self.fc_out(x)*10
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

        self.state_encoder = NormalizedLSTM(input_size=self.state_dim, hidden_size=96, batch_first=True, dropout=0.0)

        self.fc = nn.Linear(96,action_dim)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.fc.bias)
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
        s = self.state_encoder(state)
        action = self.tanh(self.fc(s))
        return action