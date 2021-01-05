import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque

def train(env, agent, n_episodes=5000, model_type='unk', score_limit=250.0, explore_episode=50):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        done = False

        while not done:
            if i_episode>explore_episode:
                action = agent.get_action(state, explore=True)
                action = action.clip(min=env.action_space.low, max=env.action_space.high)
                next_state, reward, done, _ = env.step(action)
                agent.learn_with_batches(state, action, reward, next_state, done)

                env.render()
            else:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)

            state = next_state
            score += reward

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque))) 
            test_score = test(env, agent, render=False)
            if test_score >= score_limit:
                agent.save_ckpt(model_type, 'best')
                score_limit=test_score
            else:
                agent.save_ckpt(model_type)

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
