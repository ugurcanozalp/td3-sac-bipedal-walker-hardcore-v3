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

class FeedForwardEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedForwardEncoder, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin1.weight.data = fanin_init(self.lin1.weight.data.size())
        self.lin2 = nn.Linear(hidden_size, input_size)
        self.lin2.weight.data = fanin_init(self.lin2.weight.data.size())
        self.relu = nn.ReLU()
        #self.do = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(input_size)
    
    def forward(self, x):
        res = x
        x = self.lin1(x)
        #x = self.do(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.layernorm(x+res) # residual connection, similar to transformer
        return x

class Embedder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Embedder, self).__init__()
        self.lin = nn.Linear(input_size, output_size)
        self.lin.weight.data = fanin_init(self.lin.weight.data.size())
        #self.layernorm = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.tanh(self.lin(x))


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

        self.state_embedding = Embedder(state_dim, 64)
        self.state_encoder = FeedForwardEncoder(64,256)

        self.action_embedding = Embedder(action_dim, 64)

        self.fc1 = nn.Linear(128,64)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(64,1)
        self.fc2.weight.data.uniform_(-EPS,EPS)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s = self.state_embedding(state)
        s = self.state_encoder(s)
        a = self.action_embedding(action)
        x = torch.cat((s,a),dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
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

        self.state_embedding = Embedder(state_dim, 64)
        self.state_encoder = FeedForwardEncoder(64,256)

        self.fc = nn.Linear(64,action_dim)
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
        s = self.state_embedding(state)
        s = self.state_encoder(s)
        action = self.tanh(self.fc(s))
        return action