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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _log10000 = 9.21034037198
        _log1000 = 6.907755278982137
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-_log1000 / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x = x + torch.flip(self.pe[:, :x.size(1), :], dims=[1])
        x = x + self.pe[:, :x.size(1), :]
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super(LearnablePositionalEncoding, self).__init__()
        self.embedding = nn.Parameter(0.01*torch.randn(max_len, d_model))
    def forward(self, x):
        return x + self.embedding[:x.size(1)].unsqueeze(0)


class MyTransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, dim_feedforward, output_size, nhead=4, num_layers=2, max_len=32):
        super(MyTransformerEncoder, self).__init__()
        self.linear_embedding = nn.Linear(input_size, d_model)
        self.linear_embedding.weight.data = fanin_init(self.linear_embedding.weight.data.size())
        self.pos_embedding = PositionalEncoding(d_model=d_model, max_len=max_len)
        #self.pos_embedding = LearnablePositionalEncoding(d_model, max_len=max_len)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation='gelu', dropout=0.0)
        encoder.linear1.weight.data = fanin_init(encoder.linear1.weight.data.size())
        encoder.linear2.weight.data = fanin_init(encoder.linear2.weight.data.size())
        encoder.self_attn.in_proj_weight.data = fanin_init(encoder.self_attn.in_proj_weight.data.size())
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.output_layer = nn.Sequential(nn.Linear(d_model, output_size), nn.ReLU())
        self.output_layer[0].weight.data = fanin_init(self.output_layer[0].weight.data.size())

    def forward(self, x): 
        x = self.linear_embedding(x)
        x = self.pos_embedding(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
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

        self.state_encoder = MyTransformerEncoder(input_size = self.state_dim, d_model=64, 
            dim_feedforward=192, output_size=128, nhead=2, num_layers=2, max_len=max_len)

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
        s = s[:,-1]
        a = self.action_encoder(action)
        x = torch.cat((s,a),dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
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

        self.state_encoder = MyTransformerEncoder(input_size=self.state_dim, d_model=64, 
            dim_feedforward=192, output_size=128, nhead=2, num_layers=2, max_len=max_len)

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
        s = self.state_encoder(state)
        s = s[:,-1]
        action = self.tanh(self.fc(s))
        return action
