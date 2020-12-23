import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque

def train(env, agent, n_episodes=5000, max_t=700, model_type='unk', score_limit=250.0):
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
            
            #if i_episode>0*20:
            #    env.render()
            
            agent.learn_with_batches(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")

        if i_episode % 100 == 0:
            agent.save_ckpt(model_type)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque))) 
            test_score = test(env, agent, render=False)
            if test_score >= score_limit:
                agent.save_ckpt(model_type, 'best')
                score_limit=test_score
        if score >=score_limit:
            pass
    return scores

def test(env, agent, render=True):
    state = env.reset()
    score = 0
    done=False
    while not done:
        action = agent.get_action(state, explore=False)
        #print(action)
        action = action.clip(min=env.action_space.low, max=env.action_space.high)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += reward
        if render:
            env.render()

    print('\rTest Episode\tScore: {:.2f}'.format(score))

    return score
