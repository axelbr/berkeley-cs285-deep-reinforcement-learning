import glob
import json
from typing import Any

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
    plt.title('Queston 2: Double DQN')

    for algorithm, label in [('dqn', 'DQN'), ('doubledqn', 'Double DQN')]:
        steps, avg, std = aggregate_seeds(algorithm)
        plt.plot(steps, avg, label=label)
        plt.fill_between(steps, avg - std, avg + std, alpha=0.2)

    plt.legend()
    plt.savefig('figures/q2_ddqn.svg')


if __name__ == '__main__':
    plot_q2()