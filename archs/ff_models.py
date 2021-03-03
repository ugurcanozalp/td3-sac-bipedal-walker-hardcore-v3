import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random

# https://github.com/vy007vikas/PyTorch-ActorCriticRL

EPS = 0.003

class FeedForwardEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, ff_size):
        super(FeedForwardEncoder, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.lin1.weight, gain=nn.init.calculate_gain('tanh'))
        self.layn = nn.LayerNorm(hidden_size)
        self.lin2 = nn.Linear(hidden_size, ff_size)
        nn.init.xavier_uniform_(self.lin2.weight, gain=nn.init.calculate_gain('relu'))
        self.lin3 = nn.Linear(ff_size, hidden_size)
        nn.init.xavier_uniform_(self.lin3.weight, gain=nn.init.calculate_gain('relu'))
        self.tanh = nn.Tanh()
        self.act = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.tanh(self.layn(self.lin1(x)))
        # Residual connection starts
        xx = self.lin3(self.act(self.lin2(x)))
        o = self.act(self.layernorm(x+xx))
        return o 


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

        self.state_encoder = FeedForwardEncoder(self.state_dim, 96, 192)

        self.fc2 = nn.Linear(96 + self.action_dim, 128)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        
        self.fc_out = nn.Linear(128, 1)
        #nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.uniform_(self.fc_out.weight, -0.003,+0.003)
        self.fc_out.bias.data.fill_(1.0)

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

        self.state_encoder = FeedForwardEncoder(self.state_dim, 96, 192)

        self.fc = nn.Linear(96,action_dim)
        nn.init.uniform_(self.fc.weight, -0.003,+0.003)
        nn.init.zeros_(self.fc.bias)
        self.tanh = nn.Tanh()


    def forward(self, state):
        """
        returns deterministic policy function mu(s) as policy action.
        this function returns actions lying in (-1,1) 
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        s = self.state_encoder(state)
        action = self.tanh(self.fc(s))
        return action