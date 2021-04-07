#https://www.researchgate.net/publication/320296763_Recurrent_Network-based_Deterministic_Policy_Gradient_for_Solving_Bipedal_Walking_Challenge_on_Rugged_Terrains
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from torch.distributions import Normal

# https://github.com/vy007vikas/PyTorch-ActorCriticRL

EPS = 0.003

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
        self.embedding = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh())
        nn.init.xavier_uniform_(self.embedding[0].weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.embedding[0].bias)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size=hidden_size, batch_first=batch_first, bidirectional=True, num_layers=1, dropout=dropout)
        self.lstm.bias_hh_l0.data.fill_(-0.1) # force lstm to output to depend on only last state at the initialization.
        self.lstm.bias_hh_l0_reverse.data.fill_(-0.1) # force lstm to output to depend more on last state at the initialization.
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

        self.state_encoder = NormalizedLSTM(input_size=self.state_dim, hidden_size=72, batch_first=True, dropout=0.0)

        self.fc2 = nn.Linear(72 + self.action_dim, 128)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        
        self.fc_out = nn.Linear(128, 1, bias=False)
        #nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.uniform_(self.fc_out.weight, -0.003,+0.003)
        #self.fc_out.bias.data.fill_(0.0)

        self.act = nn.GELU()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s = self.state_encoder(state)
        x = torch.cat((s,action),dim=1)
        x = self.act(self.fc2(x))
        x = self.fc_out(x)*10
        return x


class Actor(nn.Module):

    def __init__(self, state_dim=24, action_dim=4, stochastic=False):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stochastic = stochastic

        self.state_encoder = NormalizedLSTM(input_size=self.state_dim, hidden_size=72, batch_first=True, dropout=0.0)

        self.fc = nn.Linear(72,action_dim)
        nn.init.uniform_(self.fc.weight, -0.003,+0.003)
        nn.init.zeros_(self.fc.bias)

        if self.stochastic:
            self.log_std = nn.Linear(72, action_dim)
            nn.init.uniform_(self.log_std.weight, -0.003,+0.003)
            nn.init.zeros_(self.log_std.bias)      
                  
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        returns deterministic policy function mu(s) as policy action.
        this function returns actions lying in (-1,1) 
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        s = self.state_encoder(state)
        if self.stochastic:
            means = self.fc(s)
            log_stds = self.log_std(s)
            log_stds = torch.clamp(log_stds, min=-10.0, max=2.0)
            dists = Normal(means, stds)
            x = dists.rsample()
            actions = self.tanh(x)
            log_probs = dists.log_prob(x) - torch.log(1-actions.pow(2) + 1e-6)
            entropies = -log_probs.sum(dim=1, keepdim=True)
            return actions, entropies

        else:
            actions = self.tanh(self.fc(s))
            return actions
