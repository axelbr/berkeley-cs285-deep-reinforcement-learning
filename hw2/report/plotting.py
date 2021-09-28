import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_small_scale():
    def plot(exp_type: str, title: str):
        files = glob.glob(f'results/small_scale/*_{exp_type}_*.json')
        curves = dict()
        for file in files:
            with open(file) as f:
                data = np.array(json.load(f))
            if 'no_rtg_dsa' in file:
                curves['no_rtg'] = data
            elif 'rtg_dsa' in file:
                curves['rtg'] = data
            else:
                curves['rtg_normalized'] = data
        for k, v in curves.items():
            plt.plot(v[:, 1], v[:, 2], label=k)
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('Average Return')
        plt.title(title)
        plt.savefig(f'figures/q1_{exp_type}.svg')
        plt.cla()

    plot('lb', title='CartPole - Large Batch')
    plot('sb', title='CartPole - Small Batch')

if __name__ == '__main__':
    plot_small_scale()