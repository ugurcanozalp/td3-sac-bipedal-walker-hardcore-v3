import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os

def plot_result(file_names, model_types, rl_type, plot_type=0, ma_length=200):
    if not isinstance(file_names, list):
        file_names = [file_names]

    if not isinstance(model_types, list):
        model_types = len(file_names) * [model_types]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = colors[:len(file_names)]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for file_name, model_type, color in zip(file_names, model_types, colors):
        results = np.loadtxt(file_name)
        episodes = results[0]
        cum_rewards = results[1]
        avg_rewards = uniform_filter1d(cum_rewards, ma_length, origin=ma_length//2-1) #
        datarep = np.tile(cum_rewards, (ma_length, 1))
        for i in range(1, ma_length):
          datarep[i, i:] = datarep[i, :-i]

        stds = np.sqrt(np.mean(np.square(datarep - avg_rewards[None, :]), 0))
        ax.plot(episodes,avg_rewards, color=color, alpha=0.9, label=model_type) #
        if plot_type==0:
            ax.scatter(episodes,cum_rewards, s=1, color=color, alpha=0.15)
        elif plot_type==1: 
            ax.fill_between(episodes, avg_rewards - stds, avg_rewards + stds, color=color, alpha=0.15) # label=model_type
        else:
            print("wrong plot type.")

    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.set_title('Score History with ' + rl_type)
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(rl_type+ '_'+'_'.join(model_types) + '.png'))
    fig.show()

if __name__=='__main__':
    file_names = ["train-hardcore-ff-td3.txt", "train-hardcore-6-lstm-td3.txt", "train-hardcore-6-trsf-td3.txt"]
    model_types = ["RFFNN", "LSTM-6", "TRSF-6"]
    rl_type = "TD3"
    plot_type = 0 # 0: scatter plot, 1: std plot
    plot_result(file_names, model_types, rl_type, plot_type)