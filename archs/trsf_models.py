#https://www.researchgate.net/publication/320296763_Recurrent_Network-based_Deterministic_Policy_Gradient_for_Solving_Bipedal_Walking_Challenge_on_Rugged_Terrains
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from .utils.stable_transformer import PositionalEncoding, LearnablePositionalEncoding, StableTransformerLayer, TransformerEncoder

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

class WeightedMeanPooling(nn.Module):
    def __init__(self, seq_len):
        super(WeightedMeanPooling,self).__init__()
        self.eps = torch.finfo(torch.float).eps
        w_tensor = (1e-3)*torch.ones(seq_len, dtype=torch.float); w_tensor[-1]=1.0
        self.w = nn.Parameter(w_tensor, requires_grad=True)

    def forward(self, x):
        return x.permute(0,2,1)@(self.w*self.w)

class StableTransformerEncoder(nn.Module):

    def __init__(self, num_layers, d_in, d_model, nhead, dim_feedforward=128, dropout=0.1, use_gate=True, seq_len=16):
        super(StableTransformerEncoder,self).__init__()
        self.inp_embedding = nn.Sequential(nn.Linear(d_in, d_model), nn.Tanh()) # 
        nn.init.xavier_uniform_(self.inp_embedding[0].weight, gain=nn.init.calculate_gain('tanh')) # 
        #self.pos_embedding = LearnablePositionalEncoding(d_model, seq_len=8)
        self.pos_embedding = PositionalEncoding(d_model, seq_len=seq_len, ratio=None)
        st_layer = StableTransformerLayer(d_model, nhead, dim_feedforward, dropout, use_gate)
        self.encoder = TransformerEncoder(st_layer, num_layers)
        self.pooler = WeightedMeanPooling(seq_len=seq_len)
        #self.pooler = LastStatePooler()
        self.layn = nn.LayerNorm(d_model)

    def forward(self, src):
        x = src
        x = self.inp_embedding(x)
        x = self.pos_embedding(x)
        x = self.encoder(x.permute(1,0,2)) # batch, seq, emb -> seq, batch, emb
        x = self.pooler(x.permute(1,0,2)) # seq, batch, emb -> batch, emb 
        x = self.layn(x)
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
        
        self.state_encoder = StableTransformerEncoder(num_layers=1, d_in=self.state_dim,
            d_model=96, nhead=4, dim_feedforward=192, dropout=0.0, use_gate = False)
        self.action_encoder = nn.Sequential(nn.Linear(self.action_dim, 96), nn.Tanh()) # 
        nn.init.xavier_uniform_(self.action_encoder[0].weight, gain=nn.init.calculate_gain('tanh')) # 

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
        
        self.state_encoder = StableTransformerEncoder(num_layers=1, d_in=self.state_dim,
            d_model=96, nhead=4, dim_feedforward=192, dropout=0.0, use_gate = False)

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
