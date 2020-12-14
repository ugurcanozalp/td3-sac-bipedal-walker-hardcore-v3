import gym
from collections import deque
import numpy as np

# https://alexandervandekleut.github.io/gym-wrappers/
#env = gym.make('BipedalWalker-v3')

class BoxToHistoryBox(gym.ObservationWrapper):
    def __init__(self, env, h=8):
        super().__init__(env)
        self.h = h
        self.obs_memory = deque(maxlen=self.h)
        shape = (h,) + self.observation_space.shape
        low = np.repeat(np.expand_dims(self.observation_space.low, 0), h, axis=0)
        high = np.repeat(np.expand_dims(self.observation_space.high, 0), h, axis=0)    
        self.observation_space = gym.spaces.Box(low, high, shape)

    def add_to_memory(self, obs):
        self.obs_memory.append(np.expand_dims(obs, axis=0))

    def observation(self, obs):
        self.add_to_memory(obs)
        return np.concatenate(self.obs_memory)

    def reset(self):
        reset_state = self.env.reset()
        for i in range(self.h-1):
            self.add_to_memory(reset_state)
        return self.observation(reset_state)
