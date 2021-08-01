import gym
from gym.wrappers import Monitor
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent
from sac_agent import SACAgent
from fcn_train_test import train, test
from env_wrappers import BoxToHistoryBox, MyWalkerWrapper
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--flag", type=str, choices=['train', 'test', 'test-record', 'test-exp', 'test-100'],
                    default='train', help="train or test?")
parser.add_argument("-e", "--env", type=str, choices=['classic', 'hardcore'],
                    default='hardcore', help="environment type, classic or hardcore?")
parser.add_argument("-m", "--model_type", type=str, choices=['ff','mlp','rnn','lstm','bilstm','trsf'],
                    default='ff', help="model type")
parser.add_argument("-r", "--rl_type", type=str, choices=['ddpg', 'td3', 'sac'], default='sac', help='RL method')
parser.add_argument("-l", "--lr", type=float, default=4e-4, help='Learning Rate')
parser.add_argument("-w", "--wd", type=float, default=0, help='Weight Decay')
parser.add_argument("-b", "--batch_size", type=int, default=64, help='batch size')
parser.add_argument("-c", "--ckpt", type=str, default='seed', help='checkpoint to start with')
parser.add_argument("-x", "--explore_episode", type=int, default=50, help='number of exploration steps')
parser.add_argument("-g", "--gamma", type=float, default=0.98, help='discount rate')
parser.add_argument("-a", "--alpha", type=float, default=0.01, help='entropy regularization term in SAC')
parser.add_argument("-hl", "--history_length", type=int, default=12, help='history length for sequential models')
parser.add_argument("-d", "--device", type=str, default=None, help='device to train on')

args = parser.parse_args()

if args.model_type=='ff':
    from archs.ff_models import Actor, Critic
elif args.model_type=='trsf':
    from archs.trsf_models import Actor, Critic
elif args.model_type=='mlp':
    from archs.mlp_models import Actor, Critic
elif args.model_type=='rnn':
    from archs.rnn_models import Actor, Critic
elif args.model_type=='lstm':
    from archs.lstm_models import Actor, Critic
elif args.model_type=='bilstm':
    from archs.bilstm_models import Actor, Critic
else:
    print('Wrong model type!'); exit(0);

if args.env == 'classic':
    env = gym.make('BipedalWalker-v3')
    env = MyWalkerWrapper(env, skip=2)
elif args.env == 'hardcore':
    env = gym.make('BipedalWalkerHardcore-v3')
    env = MyWalkerWrapper(env, skip=2)
    
if args.model_type in ['rnn', 'lstm', 'bilstm', 'trsf']:
    env = BoxToHistoryBox(env, h=args.history_length)
    env_type = args.env + "-" + str(args.history_length)
else:
    env_type = args.env


if args.rl_type=='ddpg':
    agent = DDPGAgent(Actor, Critic, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1], lr=args.lr, weight_decay=args.wd, gamma=args.gamma, batch_size=args.batch_size, device=args.device)
elif args.rl_type=='td3':
    agent = TD3Agent(Actor, Critic, clip_low=-1, clip_high=+1, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1],lr=args.lr, weight_decay=args.wd, gamma=args.gamma, batch_size=args.batch_size, device=args.device)
elif args.rl_type=='sac':
    agent = SACAgent(Actor, Critic, clip_low=-1, clip_high=+1, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1],lr=args.lr, weight_decay=args.wd, gamma=args.gamma, alpha=args.alpha, batch_size=args.batch_size, device=args.device)
else:
    print('Wrong learning algorithm type!'); exit(0);

agent.load_ckpt(args.model_type, env_type, args.ckpt)

print("Action dimension : ",env.action_space.shape)
print("State  dimension : ",env.observation_space.shape)
print("Action sample : ",env.action_space.sample())
print("State sample  : \n ",env.reset())    

if args.flag == 'train':
    agent.train_mode()   
    scores, test_scores = train(env, agent, model_type=args.model_type, env_type=env_type, explore_episode=args.explore_episode)
    # Generate Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(scores[0],scores[1],'b.',alpha=0.5)
    ax.plot(scores[0],scores[2],'b-',alpha=1.0, label=args.model_type+'-'+args.rl_type)
    #ax.step(test_scores[0],test_scores[1], 'b-', label=args.model_type+'-'+args.rl_type)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.set_title('Score History')
    ax.legend()
    fig.savefig(os.path.join("results", args.model_type+'-'+args.rl_type+'.png'))
    fig.show()
    env.close()
    np.savetxt(os.path.join("results", "train"+"-"+env_type+'-'+args.model_type+'-'+args.rl_type+'.txt'), scores, fmt="%.6e")
    np.savetxt(os.path.join("results", "test"+"-"+env_type+'-'+args.model_type+'-'+args.rl_type+'.txt'), test_scores, fmt="%.6e")

elif args.flag == 'test' or args.flag == 'test-exp':
    agent.eval_mode()
    #env.seed(0)
    explore = (args.flag == 'test-exp')
    scores = test(env, agent, explore=explore)
    env.close()

elif args.flag == 'test-record':
    # sudo apt-get install ffmpeg
    agent.eval_mode()
    #env.seed(0)
    env = Monitor(env, os.path.join('.', 'results', 'video'), force=False)
    scores = test(env, agent)
    env.close()

elif args.flag == 'test-100':
    agent.eval_mode()
    #env.seed(0)
    scores = test(env, agent, render=False, explore=False, n_times=100)

else:
    print('Wrong flag!')
