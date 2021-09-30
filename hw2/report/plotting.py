import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


def load_data(path: str, as_numpy=True) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
        if as_numpy:
            data = np.array(data)
    return data


def plot_experiment1():
    def plot(exp_type: str, title: str):
        files = glob.glob(f'results/experiment1/*_{exp_type}_*.json')
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


def plot_experiment3():
    avg = load_data(
        'results/experiment3/run-q2_pg_q3_b40000_r0.005_LunarLanderContinuous-v2_29-09-2021_08-33-57-tag-Eval_AverageReturn.json')[
          :, 2]
    std = load_data(
        'results/experiment3/run-q2_pg_q3_b40000_r0.005_LunarLanderContinuous-v2_29-09-2021_08-33-57-tag-Eval_StdReturn.json')[
          :, 2]
    iterations = np.arange(len(avg))
    plt.plot(iterations, avg)
    plt.fill_between(iterations, avg - std, avg + std, alpha=0.2)
    plt.xlabel('iterations')
    plt.ylabel('Average Return')
    plt.title('Experiment 3 - LunarLander')
    plt.savefig('figures/q3_lunar_lander.svg')


def plot_experiment4():
    def plot_search():
        for exp in glob.glob('results/experiment4/*search*'):
            lr = re.search(r'_lr(.*?)_', exp).group(1)
            batch_size = re.search(r'_b(.*?)_', exp).group(1)
            data = load_data(exp, as_numpy=False)
            avg = np.array(data['Eval_AverageReturn'])[:, 2]
            std = np.array(data['Eval_StdReturn'])[:, 2]
            iters = np.arange(len(avg))
            plt.plot(iters, avg, label=f'batch_size: {batch_size}, lr: {lr}')
            plt.fill_between(iters, avg - std, avg + std, alpha=0.2)
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('Average Return')
        plt.title('Experiment 4 - GridSearch HalfCheetah')
        plt.savefig('figures/q4_half_cheetah_search.svg')
        plt.cla()

    def plot_comparison():
        for exp in glob.glob('results/experiment4/q2_pg_q4_b*'):
            label = 'pg'
            if 'rtg' in exp:
                label += '_rtg'
            if 'nnbaseline' in exp:
                label += '_nnbaseline'
            data = load_data(exp, as_numpy=False)
            avg = np.array(data['Eval_AverageReturn'])[:, 2]
            std = np.array(data['Eval_StdReturn'])[:, 2]
            iters = np.arange(len(avg))
            plt.plot(iters, avg, label=label)
            plt.fill_between(iters, avg - std, avg + std, alpha=0.2)
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('Average Return')
        plt.title('Experiment 4 - Comparison')
        plt.savefig('figures/q4_half_cheetah_comparison.svg')
        plt.cla()

    plot_search()
    plot_comparison()


def plot_experiment5():
    for exp in glob.glob('results/experiment5/q2_pg_q5*'):
        lambda_ = re.search(r'_lambda(.*?)_', exp).group(1)
        data = load_data(exp, as_numpy=False)
        avg = np.array(data['Eval_AverageReturn'])[:, 2]
        std = np.array(data['Eval_StdReturn'])[:, 2]
        iters = np.arange(len(avg))
        plt.plot(iters, avg, label=f'lambda={lambda_}')
        plt.fill_between(iters, avg - std, avg + std, alpha=0.2)
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('Average Return')
    plt.title('Experiment 5 - HopperV2 with GAE')
    plt.savefig('figures/q5_hopper_gae.svg')

def plot_parallelization():
    parallel_run = load_data('results/experiment4/q2_pg_q4_b30000_r0.02_rtg_nnbaseline_HalfCheetah-v2_29-09-2021_15-53-57.json', as_numpy=False)
    run = load_data('results/experiment4/q2_pg_q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_29-09-2021_11-28-41.json', as_numpy=False)

    parallel_durations = np.array(parallel_run['Eval_AverageReturn'])[:, 0] / 60.0
    single_durations = np.array(run['Eval_AverageReturn'])[:, 0] / 60.0

    plt.bar(['1 worker', '5 workers'], [single_durations[-1] - single_durations[0], parallel_durations[-1] - parallel_durations[0]], width=.5)
    plt.yticks(np.arange(0., 1.0 + single_durations[-1] - single_durations[0], 3.0))
    plt.title('Parallel Sampling for HalfCheetah (batch_size=30000, iterations=100)')
    pl.ylabel('Minutes')
    plt.savefig('figures/bonus_parallelization.svg')




if __name__ == '__main__':
    plot_parallelization()
