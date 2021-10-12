import glob
import json
from typing import Any
import os

import numpy as np
import matplotlib.pyplot as plt

def load_data(path: str) -> Any:
    with open(path) as f:
        data = json.load(f)
    return data

def plot_q2():

    def aggregate_seeds(algorithm: str):
        data = []
        for log in glob.glob(f'results/question_2/*_{algorithm}_*.json'):
            data.append(load_data(log)['Train_AverageReturn'])
        data = np.stack(data, axis=-1)
        steps, avg, std = data[:, 1, 0], data.mean(-1)[:, -1], data.std(-1)[:, -1]
        return steps.astype(int), avg, std

    plt.xlabel('steps')
    plt.ylabel('Average Return')
    plt.title('Question 2: Double DQN')

    for algorithm, label in [('dqn', 'DQN'), ('doubledqn', 'Double DQN')]:
        steps, avg, std = aggregate_seeds(algorithm)
        plt.plot(steps, avg, label=label)
        plt.fill_between(steps, avg - std, avg + std, alpha=0.2)

    plt.legend()
    plt.savefig('figures/q2_ddqn.svg')

def plot_q4():

    def plot(path: str):
        data = load_data(path)
        avg, std, steps = np.array(data['Eval_AverageReturn']), np.array(data['Eval_StdReturn']), np.array(data['Train_EnvstepsSoFar'])
        avg, std, steps = avg[:, -1], std[:, -1], steps[:, -1]
        target_updates = os.path.basename(path).split('_')[1]
        gradient_updates = os.path.basename(path).split('_')[2]
        plt.plot(steps, avg, label=f'ntu={target_updates}, ngsptu={gradient_updates}')
        plt.fill_between(steps, avg - std, avg + std, alpha=0.2)

    plt.xlabel('steps')
    plt.ylabel('Average Return')
    plt.title('Question 4: Actor Critic Cartpole')

    for log in glob.glob('results/question_4/*.json'):
        plot(log)

    plt.legend(loc='upper left')
    plt.savefig('figures/q4_cartpole.svg')


if __name__ == '__main__':
    plot_q4()