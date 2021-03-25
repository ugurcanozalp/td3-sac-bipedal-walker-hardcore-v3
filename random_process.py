import numpy as np
import matplotlib.pyplot as plt
from noise import *
t = np.arange(0,50,0.04)
mu=np.array([0])
noise_1 = OrnsteinUhlenbeckNoise(mu=mu, theta=0.5 ,sigma=1.0, dt=0.04)
noise_2 = OrnsteinUhlenbeckNoise(mu=mu, theta=2.0 ,sigma=2.0, dt=0.04)
noise_3 = GaussianNoise(mu=mu, sigma=1.0)

x1 = np.array([ noise_1() for tt in t])
x2 = np.array([ noise_2() for tt in t])
x3 = np.array([ noise_3() for tt in t])

fig = plt.figure(figsize=(12.8,7.2))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
fig.subplots_adjust(hspace=0.35)

ax1.plot(t,x1, color="b", linestyle="dotted", alpha=1.0)
ax1.set_ylabel('x')
ax1.set_title(r"$OU( \theta=0.5,\sigma=1.0), \Delta t=0.04$")
ax2.plot(t,x2, color="g", linestyle="dotted", alpha=1.0)
ax2.set_ylabel('x')
ax2.set_title(r"$OU( \theta=2.0,\sigma=2.0), \Delta t=0.04$")
ax3.plot(t,x3, color="r", linestyle="dotted", alpha=1.0)
ax3.set_ylabel('x')
ax3.set_title(r"$\mathcal{N}(\mu=0, \sigma=1.0), \Delta t=0.04$")
ax3.set_xlabel('Time')
#fig.suptitle('Random Processes')
fig.savefig('random_processes.png')
fig.show()