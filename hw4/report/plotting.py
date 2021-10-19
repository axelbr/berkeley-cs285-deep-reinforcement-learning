import glob
import json
import re
from typing import Any
import os

import numpy as np
import matplotlib.pyplot as plt

def load_data(path: str) -> Any:
    with open(path) as f:
        data = json.load(f)
    return data

def plot_q1():

  def plot_loss(file):
      data = np.load(file)
      expr = re.search(r'_n(.*?)_arch(.*?)_', file)
      label = f'n={expr.group(1)}, arch={expr.group(2)}'
      plt.plot(np.arange(data.shape[0]), data, label=label)

  plt.xlabel('steps')
  plt.ylabel('Model Loss')
  plt.title('Question 1: Losses')

  for file in glob.glob('results/q1/*.npy'):
      plot_loss(file)

  plt.legend()
  plt.savefig('figures/q1_losses.svg')

def plot_q2():
    data = load_data('results/q2/hw4_q2_obstacles_singleiteration_obstacles-cs285-v0_18-10-2021_16-24-44.json')
    plt.xlabel('steps')
    plt.ylabel('Costs')
    plt.title('Question 2: Random Action Selection')
    y_values = [-data['Eval_AverageReturn'][0][-1], -data['Train_AverageReturn'][0][-1]]
    plt.bar(['Eval', 'Train'], y_values)
    plt.savefig('figures/q2.svg')

def plot_q3():

    def plot(file: str):
        data = load_data(file)
        label = file[file.find('q3_'):].split('_')[1].capitalize()
        avg, std = np.array(data['Eval_AverageReturn']), np.array(data['Eval_StdReturn'])
        steps = avg[:, 1].astype(int)
        avg, std = avg[:, -1], std[:, -1]
        plt.plot(steps, avg, label=label)
        plt.fill_between(steps, avg - std, avg + std, alpha=0.2)
    plt.xticks(list(range(20)))
    plt.xlabel('steps')
    plt.ylabel('Returns')
    plt.title('Question 3: Iterative Model Training')

    for file in glob.glob('results/q3/*.json'):
        plot(file)

    plt.legend()
    plt.savefig('figures/q3.svg')

def plot_q5():

    def plot(file: str):
        data = load_data(file)
        if 'cem' in file:
            iters = re.search(r'_cem_(.*?)_', file).group(1)
            label = f'CEM, iters={iters}'
        else:
            label = 'Random Shooting'

        avg, std = np.array(data['Eval_AverageReturn']), np.array(data['Eval_StdReturn'])
        steps = avg[:, 1].astype(int)
        avg, std = avg[:, -1], std[:, -1]
        plt.plot(steps, avg, label=label)
        plt.fill_between(steps, avg - std, avg + std, alpha=0.2)

    plt.xticks([0, 1,2,3,4,5])
    plt.xlabel('Epochs')
    plt.ylabel('Returns')
    plt.title('Question 5: Random-Shooting vs. CEM')

    for file in glob.glob('results/q5/*.json'):
        plot(file)

    plt.legend()
    plt.savefig('figures/q5.svg')

if __name__ == '__main__':
    plot_q5()