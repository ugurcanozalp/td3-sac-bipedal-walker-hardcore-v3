import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent
from fcn_train_test import train, test
from env_wrappers import BoxToHistoryBox
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--flag", type=str, choices=['train', 'test'],
                    default='train', help="train or test?")
parser.add_argument("-e", "--env", type=str, choices=['classic', 'hardcore'],
                    default='classic', help="environment type, classic or hardcore?")
parser.add_argument("-m", "--model_type", type=str, choices=['ff','lstm','bilstm','trsf'],
                    default='ff', help="model type")
parser.add_argument("-r", "--rl_type", type=str, choices=['ddpg', 'td3'], default='ddpg', help='RL method')
parser.add_argument("-l", "--lr", type=float, default=5e-4, help='Learning Rate')
parser.add_argument("-w", "--wd", type=float, default=1e-6, help='Weight Decay')
parser.add_argument("-c", "--ckpt", type=str, default='ep100', help='checkpoint to start with')

args = parser.parse_args()

if args.model_type=='ff':
    from archs.ff_models import Actor, Critic
elif args.model_type=='lstm':
    from archs.lstm_models import Actor, Critic
elif args.model_type=='bilstm':
    from archs.lstm_models import Actor, Critic
elif args.model_type=='trsf':
    from archs.trsf_models import Actor, Critic
else:
    print('Wrong model type!'); exit(0);

if args.env == 'classic':
    env = gym.make('BipedalWalker-v3')
    #env._max_episode_steps = 1200
elif args.env == 'hardcore':
    env = gym.make('BipedalWalkerHardcore-v3')
    #env._max_episode_steps = 1600
    
if args.model_type in ['lstm', 'bilstm','trsf']:
    env = BoxToHistoryBox(env, h=8)

if args.rl_type=='ddpg':
    agent = DDPGAgent(Actor, Critic, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1], lr=args.lr, weight_decay=args.wd)
elif args.rl_type=='td3':
    agent = TD3Agent(Actor, Critic, clip_low=-1, clip_high=+1, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1],lr=args.lr, weight_decay=args.wd)
else:
    print('Wrong learning algorithm type!'); exit(0);

agent.load_ckpt(args.model_type, args.ckpt)

print("Action dimension : ",env.action_space.shape)
print("State  dimension : ",env.observation_space.shape)
print("Action sample : ",env.action_space.sample())
print("State sample  : \n ",env.reset())    

if args.flag == 'train':
    agent.train_mode()   
    scores, test_scores = train(env, agent, model_type=args.model_type)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(scores[0],scores[1],'b-.', alpha=.5)
    ax.step(test_scores[0],test_scores[1], 'b-', label=args.model_type+'-'+args.rl_type)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.set_title('Score History')
    ax.legend()
    fig.savefig(os.path.join("results", args.model_type+'-'+args.rl_type+'.png'))
    fig.show()
    env.close()
    np.savetxt(os.path.join("results", "train"+"-"+args.model_type+'-'+args.rl_type+'.txt'), scores, fmt="%.6e")
    np.savetxt(os.path.join("results", "test"+"-"+args.model_type+'-'+args.rl_type+'.txt'), test_scores, fmt="%.6e")

elif args.flag == 'test':
    #agent.freeze_networks()
    agent.eval_mode()
    #env.seed(0)
    scores = test(env, agent)
    env.close()
else:
    print('Wrong flag!')
