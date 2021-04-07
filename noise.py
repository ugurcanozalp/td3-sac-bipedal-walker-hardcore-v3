import numpy as np
#np.random.seed(0)

class AbstractNoise:
    def __init__(self):
        pass

    def step_end(self):
        pass

    def episode_end(self):
        pass

class GaussianNoise(AbstractNoise):
    def __init__(self, mu, sigma, clip=None):
        self.mu = mu
        self.sigma = sigma
        self.clip = clip        

    def __call__(self):
        delta = self.sigma*np.random.normal(size=self.mu.shape)
        if self.clip is not None:
            delta = delta.clip(-self.clip,+self.clip)

        return self.mu + delta

class OrnsteinUhlenbeckNoise(AbstractNoise):
    def __init__(self, mu, theta = 0.15, sigma = 0.2, dt=0.02):
        # 5.0, 0.02, 1.0 # 1.0, 0.02, 0.25 # 7.5, 0.02, 1.4 # 5.0, 0.02, 0.7
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt        
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class DecayingGaussianNoise(AbstractNoise):
    def __init__(self, mu, sigma = 0.12, clip=None, sigma_decay = 0.999):
        # 5.0, 0.02, 1.0 # 1.0, 0.02, 0.25 # 7.5, 0.02, 1.4 # 5.0, 0.02, 0.7
        self.mu = mu
        self.clip = clip   
        self.sigma = end_sigma
        self.sigma_decay = sigma_decay

    def episode_end(self):
        self.sigma *= self.sigma_decay

    def __call__(self):
        delta = self.sigma*np.random.normal(size=self.mu.shape)
        if self.clip is not None:
            delta = delta.clip(-self.clip,+self.clip)

        return x

class DecayingOrnsteinUhlenbeckNoise(AbstractNoise):
    def __init__(self, mu, theta = 0.15, sigma = 0.2, dt=0.02, sigma_decay = 0.999):
        # 5.0, 0.02, 1.0 # 1.0, 0.02, 0.25 # 7.5, 0.02, 1.4 # 5.0, 0.02, 0.7
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt        
        self.x_prev = np.zeros_like(self.mu)
        self.sigma_decay = sigma_decay

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def episode_end(self):
        self.sigma *= self.sigma_decay

class RandomNoise(AbstractNoise):
    def __init__(self, mu, minval, maxval, prob=0.5):
        self.mu = mu
        self.minval = minval
        self.maxval = maxval
        self.prob = prob        

    def __call__(self):
        rand_or_not = np.random.rand(*self.mu.shape) < self.prob
        delta = self.minval + (self.maxval-self.minval)*np.random.rand(*self.mu.shape) 
        return self.mu + delta * rand_or_not

class DecayingRandomNoise(AbstractNoise):
    def __init__(self, mu, minval, maxval, prob=0.5, decay = 0.999):
        self.mu = mu
        self.minval = minval
        self.maxval = maxval
        self.prob = prob  
        self.decay = decay      

    def __call__(self):
        rand_or_not = np.random.rand(*self.mu.shape) < self.prob
        delta = self.minval + (self.maxval-self.minval)*np.random.rand(*self.mu.shape) 
        return self.mu + delta * rand_or_not

    def episode_end(self):
        self.prob *= self.decay
