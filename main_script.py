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
parser.add_argument("-m", "--model_type", type=str, choices=['mlp','mlp2','lstm','transformer'],
                    default='mlp2', help="model type")
parser.add_argument("-r", "--rl_type", type=str, choices=['ddpg', 'td3'], default='ddpg', help='RL method')

args = parser.parse_args()

if args.model_type=='mlp':
    from archs.mlp_models import Actor, Critic
elif args.model_type=='mlp2':
    from archs.mlp2_models import Actor, Critic
elif args.model_type=='lstm':
    from archs.lstm_models import Actor, Critic
elif args.model_type=='transformer':
    from archs.transformer_models import Actor, Critic
else:
    print('Wrong model type!'); exit(0);

if args.env == 'classic':
    env = gym.make('BipedalWalker-v3')
elif args.env == 'hardcore':
    env = gym.make('BipedalWalkerHardcore-v3')

if args.model_type in ['lstm', 'transformer']:
    env = BoxToHistoryBox(env, h=16)

if args.rl_type=='ddpg':
    agent = DDPGAgent(Actor, Critic, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1])
elif args.rl_type=='td3':
    agent = TD3Agent(Actor, Critic, clip_low=-1, clip_high=+1, state_size = env.observation_space.shape[-1], action_size=env.action_space.shape[-1])
else:
    print('Wrong learning algorithm type!'); exit(0);
    
print("Action dimension : ",env.action_space.shape)
print("State  dimension : ",env.observation_space.shape)
print("Action sample : ",env.action_space.sample())
print("State sample  : \n ",env.reset())    

if args.flag == 'train':
    env._max_episode_steps = 1000
    agent.train_mode()   
    scores = train(env, agent, model_type=args.model_type)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    env.close()
elif args.flag == 'test':
    env._max_episode_steps = 1600
    #agent.freeze_networks()
    agent.eval_mode()
    try:
        assert False
        actor_file = os.path.join("models", args.rl_type, "_".join(["best", args.model_type, "actor.pth"]))
        agent.train_actor.load_state_dict(torch.load(actor_file, map_location=agent.device))
    except:
        actor_file = os.path.join("models", args.rl_type, "_".join(["last", args.model_type, "actor.pth"]))
        agent.train_actor.load_state_dict(torch.load(actor_file, map_location=agent.device))
    agent.train_actor.eval()

    scores = test(env, agent)

    env.close()
else:
    print('Wrong flag!')
