import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent


def train_ddpg(env, agent, n_episodes=5000, max_t=700, populate_episode=20, trainer_name='type_x'):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.get_action(state, explore=True)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            #if i_episode>20:
            #    env.render()
            if i_episode>populate_episode:
                agent.learn_with_batches(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")

        if i_episode % 100 == 0:
            torch.save(agent.train_actor.state_dict(), os.path.join('models',trainer_name+'_ckpt_actor.pth'))
            torch.save(agent.train_critic.state_dict(), os.path.join('models',trainer_name+'_ckpt_critic.pth'))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque))) 
            test_ddpg(env, agent, render=False)
        if score >=200:
            torch.save(agent.train_actor.state_dict(), os.path.join('models',trainer_name+'_ckpt_actor.pth'))
            torch.save(agent.train_critic.state_dict(), os.path.join('models',trainer_name+'_ckpt_critic.pth'))
            break
    return scores

def test_ddpg(env, agent, render=True):
    state = env.reset()
    score = 0
    done=False
    while not done:
        action = agent.get_action(state, explore=False)
        action = action.clip(min=env.action_space.low, max=env.action_space.high)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += reward
        if render:
            env.render()

    print('\rTest Episode\tScore: {:.2f}'.format(score))


if __name__=='__main__':
    env = gym.make('BipedalWalker-v3')
    agent = Agent(state_size = env.observation_space.shape[0], action_size=env.action_space.shape[0])
    env.seed(0)
    print("Action dimension : ",env.action_space.shape)
    print("State  dimension : ",env.observation_space.shape)
    print("Action sample : ",env.action_space.sample())
    print("State sample  : \n ",env.reset())        

    scores = train_ddpg(env, agent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()