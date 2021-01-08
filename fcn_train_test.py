import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque

def train(env, agent, n_episodes=3000, model_type='unk', score_limit=250.0, explore_episode=50, test_f=100):
    scores_deque = deque(maxlen=100)
    scores = []
    test_scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        done = False

        while not done:
            if i_episode>explore_episode:
                action = agent.get_action(state, explore=True)
                action = action.clip(min=env.action_space.low, max=env.action_space.high)
            else:
                action = env.action_space.sample()
                
            next_state, reward, done, _ = env.step(action)
            agent.learn_with_batches(state, action, reward, next_state, done)

            state = next_state
            score += reward

        scores_deque.append(score)
        scores.append(score)
        avg_score_100 = np.mean(scores_deque)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, avg_score_100, score), end="")

        if i_episode % test_f == 0 or avg_score_100>200.0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            agent.eval_mode() # test in eval mode.
            test_score = test(env, agent, render=False)
            test_scores.append(test_score)
            if test_score >= score_limit:
                agent.save_ckpt(model_type, 'best'+str(int(test_score)))
                score_limit=test_score
            else:
                agent.save_ckpt(model_type)
            if avg_score_100>200.0:
                break
            agent.train_mode() # when the test done, come back to train mode.

    return scores, test_scores

def test(env, agent, render=True):
    prev_max_episode_steps = env._max_episode_steps
    env._max_episode_steps = 1600
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
    env._max_episode_steps = prev_max_episode_steps
    return score

#fig = plt.figure()
#ax = fig.add_subplot(111)
#...
#ax.title('Score History')
#ax.legend()
#f.savefig(savename)
#f.show()

def create_graph(ax, scores, test_scores, test_f=100, label='FFRC'):
    episodes = np.arange(1, len(scores)+1)
    test_episodes = np.arange(test_f, (len(test_scores)+1)*test_f, test_f)
    ax.plot(episodes, scores, alpha=.5)
    ax.step(test_episodes, test_scores, label=label)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')

