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

class Critic(nn.Module):

    def __init__(self, state_dim=24, action_dim=4, seq_len=32):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.pos_embedding = nn.Embedding(seq_len, state_dim)
        encoder = nn.TransformerEncoderLayer(d_model=state_dim, nhead=4, dim_feedforward=64, activation='relu')
        encoder.linear1.weight.data = fanin_init(self.encoder.linear1.data.size())
        encoder.linear2.weight.data = fanin_init(self.encoder.linear2.data.size())
        self.transformer = nn.TransformerEncoder(encoder, num_layers=4) 


        self.fca = nn.Linear(action_dim,64)
        self.fca.weight.data = fanin_init(self.fca.weight.data.size())

        self.fc1 = nn.Linear(128,64)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(64,1)
        self.fc2.weight.data.uniform_(-EPS,EPS)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1, (_, _) = self.lstm(state)
        s1 = s1[:,0]
        a1 = self.tanh(self.fca(action))
        x = torch.cat((s1,a1),dim=1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Actor(nn.Module):

    def __init__(self, state_dim=24, action_dim=4, seq_len=32):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.pos_embedding = nn.Embedding(seq_len, state_dim)
        encoder = nn.TransformerEncoderLayer(d_model=state_dim, nhead=4, dim_feedforward=64, activation='relu')
        encoder.linear1.weight.data = fanin_init(self.encoder.linear1.data.size())
        encoder.linear2.weight.data = fanin_init(self.encoder.linear2.data.size())
        self.transformer = nn.TransformerEncoder(encoder, num_layers=4) 

        self.fc1 = nn.Linear(64,32)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(32,action_dim)
        self.fc2.weight.data.uniform_(-EPS,EPS)

        self.relu = nn.ReLU()
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
        x, (_, _) = self.lstm(state)
        x = x[:,0]
        x = self.relu(self.fc1(x))
        action = self.tanh(self.fc2(x))

        return action
