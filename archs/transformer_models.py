#https://www.researchgate.net/publication/320296763_Recurrent_Network-based_Deterministic_Policy_Gradient_for_Solving_Bipedal_Walking_Challenge_on_Rugged_Terrains
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from .utils.stable_transformer import PositionalEncoding, StableTransformerLayer, TransformerEncoder

# https://github.com/vy007vikas/PyTorch-ActorCriticRL

EPS = 0.003

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class LastStatePooler(nn.Module):
    def __init__(self, d_in, d_out):
        super(LastStatePooler, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.linear.weight.data = fanin_init(self.linear.weight.data.size())
        self.tanh = nn.Tanh()
    def forward(self,x):
        x = self.linear(x)
        x = x[:, -1]
        return self.tanh(x)

class MaxPooler(nn.Module):
    def __init__(self, d_in, d_out):
        super(LastStatePooler, self).__init__()
        self.linear = nn.Sequential(nn.Linear(d_in, d_out))
        self.linear.weight.data = fanin_init(self.linear.weight.data.size())
    def forward(self,x):
        x = self.linear(x)
        x = x.max(axis=-2) # -2 -> sequence dimension
        return x

class StableTransformerEncoder(nn.Module):

    def __init__(self, num_layers, d_in, d_model, nhead, dim_feedforward=192, d_out=128, dropout=0.1, use_gate = False):
        super(StableTransformerEncoder,self).__init__()
        self.inp_embedding = nn.Sequential(nn.Linear(d_in, d_model), nn.Tanh())
        self.inp_embedding[0].weight.data = fanin_init(self.inp_embedding[0].weight.data.size())
        self.pos_embedding = PositionalEncoding(d_model, max_len=32)
        st_layer = StableTransformerLayer(d_model, nhead, dim_feedforward, dropout, use_gate)
        self.encoder = TransformerEncoder(st_layer, num_layers)
        self.pooler = LastStatePooler(d_model, d_out)

    def forward(self, src, mask=None):
        x = src
        x = self.inp_embedding(x)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        x = self.pooler(x)
        return x


class Critic(nn.Module):

    def __init__(self, state_dim=24, action_dim=4, max_len=32):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.state_encoder = StableTransformerEncoder(num_layers=2, d_in=self.state_dim, 
            d_model=64, nhead=2, dim_feedforward=192, d_out=128, dropout=0.0, use_gate = False)

        self.action_encoder = nn.Sequential(nn.Linear(self.action_dim, 128), nn.LayerNorm(128), nn.ReLU())
        self.action_encoder[0].weight.data = fanin_init(self.action_encoder[0].weight.data.size())

        self.fc1 = nn.Linear(256,128)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(128,1)
        self.fc2.weight.data.uniform_(-EPS,EPS)
        self.fc2.bias.data.fill_(-1.0)

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
        x = self.fc2(x)*10
        return x


class Actor(nn.Module):

    def __init__(self, state_dim=24, action_dim=4, max_len=32):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.state_encoder = StableTransformerEncoder(num_layers=2, d_in=self.state_dim, 
            d_model=64, nhead=2, dim_feedforward=192, d_out=128, dropout=0.0, use_gate = False)

        self.fc = nn.Linear(128,action_dim)
        self.fc.weight.data.uniform_(-EPS,EPS)
        self.fc.bias.data.zero_()
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
