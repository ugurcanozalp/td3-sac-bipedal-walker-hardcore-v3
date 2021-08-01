import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from torch.distributions import Normal

EPS = 0.003

class FeedForwardEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, ff_size):
        super(FeedForwardEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        self.block = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, ff_size), nn.GELU(), nn.Linear(ff_size, hidden_size))

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.block(x)
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

        self.state_encoder = FeedForwardEncoder(self.state_dim, 96, 192)

        self.fc2 = nn.Linear(96 + self.action_dim, 192)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))
        
        self.fc_out = nn.Linear(192, 1, bias=False)
        nn.init.uniform_(self.fc_out.weight, -0.003,+0.003)

        self.act = nn.Tanh()

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

        self.state_encoder = FeedForwardEncoder(self.state_dim, 96, 192)

        self.fc = nn.Linear(96, action_dim, bias=False)
        nn.init.uniform_(self.fc.weight, -0.003,+0.003)
        #nn.init.zeros_(self.fc.bias)

        if self.stochastic:
            self.log_std = nn.Linear(96, action_dim, bias=False)
            nn.init.uniform_(self.log_std.weight, -0.003,+0.003)
            #nn.init.zeros_(self.log_std.bias)   

        self.tanh = nn.Tanh()


    def forward(self, state, explore=True):
        """
        returns either:
        - deterministic policy function mu(s) as policy action.
        - stochastic action sampled from tanh-gaussian policy, with its entropy value.
        this function returns actions lying in (-1,1) 
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        s = self.state_encoder(state)
        if self.stochastic:
            means = self.fc(s)
            log_stds = self.log_std(s)
            log_stds = torch.clamp(log_stds, min=-10.0, max=2.0)
            stds = log_stds.exp()
            #print(stds)
            dists = Normal(means, stds)
            if explore:
                x = dists.rsample()
            else:
                x = means
            actions = self.tanh(x)
            log_probs = dists.log_prob(x) - torch.log(1-actions.pow(2) + 1e-6)
            entropies = -log_probs.sum(dim=1, keepdim=True)
            return actions, entropies

        else:
            actions = self.tanh(self.fc(s))
            return actions
