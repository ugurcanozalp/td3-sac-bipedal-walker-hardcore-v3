import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent
from ddpg import train_ddpg, test_ddpg
from archs.mlp4_models import Actor, Critic

env = gym.make('BipedalWalker-v3')
agent = Agent(Actor, Critic, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1],
	lr=5e-4, gamma=0.99, tau=0.001, batch_size=64, buffer_size=int(2e5))
env.seed(0)
print("Action dimension : ",env.action_space.shape)
print("State  dimension : ",env.observation_space.shape)
print("Action sample : ",env.action_space.sample())
print("State sample  : \n ",env.reset())    
   
scores = train_ddpg(env, agent, trainer_name='mlp4')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()
