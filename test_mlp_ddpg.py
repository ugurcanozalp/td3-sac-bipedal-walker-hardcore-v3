import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent
from ddpg import train_ddpg, test_ddpg
from archs.mlp3_models import Actor, Critic

env = gym.make('BipedalWalker-v3')
agent = Agent(Actor, Critic, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1])
env.seed(0)
print("Action dimension : ",env.action_space.shape)
print("State  dimension : ",env.observation_space.shape)
print("Action sample : ",env.action_space.sample())
print("State sample  : \n ",env.reset())    
   

agent.train_actor.load_state_dict(torch.load('models/mlp3_ckpt_actor.pth', map_location={'cuda:0': 'cpu'}))
agent.train_critic.load_state_dict(torch.load('models/mlp3_ckpt_critic.pth', map_location={'cuda:0': 'cpu'}))
scores = test_ddpg(env, agent)

env.close()
