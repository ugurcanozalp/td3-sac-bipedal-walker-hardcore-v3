import numpy as np

class GaussianNoise:
    def __init__(self, mu, sigma, clip=None):
        self.mu = mu
        self.sigma = sigma
        self.clip = clip        

    def __call__(self):
        delta = self.sigma*np.random.normal(size=self.mu.shape)
        if self.clip is not None:
            delta = delta.clip(-self.clip,+self.clip)

        return self.mu + delta

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, theta = 0.15, sigma = 0.2):
        # 5.0, 0.02, 1.0 # 1.0, 0.02, 0.25 # 7.5, 0.02, 1.4 # 5.0, 0.02, 0.7
        self.mu = mu
        self.theta = theta
        self.sigma = sigma        
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) + \
                self.sigma * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


"""
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, theta = 7.5, dt = 0.02, sigma_max = 1.3, sigma_min = 1.3, n_steps_annealing = 2000):
        # 5.0, 0.02, 1.0 # 1.0, 0.02, 0.25 # 7.5, 0.02, 1.4 # 5.0, 0.02, 0.7
        self.mu = mu
        self.theta = theta
        self.dt = dt
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min 
        self._delta_sigma = -float(self.sigma_max - self.sigma_min) / float(n_steps_annealing)
        self._current_sigma = self.sigma_max
        self._n_step = 0
        
        self.x_prev = np.zeros_like(self.mu)

    @property
    def current_sigma(self):
        return self._current_sigma
    
    def update_sigma(self):
        self._n_step +=1
        self._current_sigma = max(self.sigma_min, self._delta_sigma * float(self._n_step) + self.sigma_max)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

"""
