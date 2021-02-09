import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random

# https://github.com/vy007vikas/PyTorch-ActorCriticRL

#EPS = 0.003

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class FeedForwardEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, ff_size):
        super(FeedForwardEncoder, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.lin1.weight, gain=nn.init.calculate_gain('tanh'))
        self.lin2 = nn.Linear(hidden_size, ff_size)
        nn.init.xavier_uniform_(self.lin2.weight, gain=nn.init.calculate_gain('relu'))
        self.lin3 = nn.Linear(ff_size, hidden_size)
        nn.init.xavier_uniform_(self.lin3.weight, gain=nn.init.calculate_gain('relu'))
        self.tanh = nn.Tanh()
        self.act = nn.GELU()
        #self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.lin1(x)
        xx = self.tanh(x)
        xx = self.lin2(xx)
        xx = self.act(xx)
        xx = self.lin3(xx)
        return x+xx

"""
class Embedder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Embedder, self).__init__()
        self.lin = nn.Linear(input_size, output_size)
        self.lin.weight = fanin_init(self.lin.weight.size())
        self.layernorm = nn.LayerNorm(output_size)
        #self.tanh = nn.Tanh()
    def forward(self, x):
        return self.layernorm(self.lin(x))
"""

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

        self.state_encoder = FeedForwardEncoder(self.state_dim, 128, 384)
        self.action_encoder = nn.Sequential(nn.Linear(self.action_dim, 128), nn.GELU())
        nn.init.xavier_uniform_(self.action_encoder[0].weight, gain=nn.init.calculate_gain('relu'))

        self.fc2 = nn.Linear(128,256)
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

        self.state_encoder = FeedForwardEncoder(self.state_dim, 128, 384)

        self.fc = nn.Linear(128,action_dim)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.01*nn.init.calculate_gain('tanh'))
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