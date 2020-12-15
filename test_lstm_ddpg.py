import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from agent import Agent
from ddpg import train_ddpg, test_ddpg
from archs.recurrent_models import Actor, Critic
from env_wrappers import BoxToHistoryBox

env_raw = gym.make('BipedalWalker-v3')
env = BoxToHistoryBox(env_raw)
agent = Agent(Actor, Critic, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1])
env.seed(0)
print("Action dimension : ",env.action_space.shape)
print("State  dimension : ",env.observation_space.shape)
print("Action sample : ",env.action_space.sample())
print("State sample  : \n ",env.reset())    
   
try:
    agent.train_actor.load_state_dict(torch.load(os.path.join('models', 'best_lstm_ckpt_actor.pth'), map_location={'cuda:0': 'cpu'}))
    agent.train_critic.load_state_dict(torch.load(os.path.join('models', 'best_lstm_ckpt_critic.pth'), map_location={'cuda:0': 'cpu'}))
except:
    agent.train_actor.load_state_dict(torch.load(os.path.join('models', 'lstm_ckpt_actor.pth'), map_location={'cuda:0': 'cpu'}))
    agent.train_critic.load_state_dict(torch.load(os.path.join('models', 'lstm_ckpt_critic.pth'), map_location={'cuda:0': 'cpu'}))

scores = test_ddpg(env, agent)

env.close()