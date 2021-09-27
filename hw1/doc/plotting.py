import matplotlib.pyplot as plt
import numpy as np
import json


def plot_q1():
    scores_gradient_steps = {
        1000: (757.7908325195312, 438.3332214355469),
        3000: (2868.809814453125, 1435.5955810546875),
        5000: (4032.78271484375, 1176.8306884765625),
        8000: (3942.4189453125, 1438.37109375),
        11000: (4450.61083984375, 800.330810546875),
        15000: (4579.6298828125, 116.30001068115234),
        19000: (4196.84375, 1126.9827880859375),
        25000: (4243.40380859375, 1157.8594970703125),
    }

    x = np.array(sorted(scores_gradient_steps.keys()))
    data = np.array([scores_gradient_steps[i] for i in x])

    plt.plot(x, data[:, 0], label='Trained Policy')
    plt.plot(x, len(x) * [4713.6533203125], label='Expert Policy')
    plt.fill_between(x, data[:, 0] - data[:, 1], data[:, 0] + data[:, 1], alpha=0.2)
    plt.title('Effect of training steps on Ant-Task')
    plt.xlabel('num_agent_train_steps_per_iter')
    plt.ylabel('Avg. Return (n=10)')
    plt.legend()
    plt.show()

    print(data)

def plot_q2():

    expert_return = {
        'Ant-v2': 4713.,
        'Humanoid-v2': 10344.517578125
    }

    def plot(env, log_scale=False):
        with open(f'results/q1_bc_{env}.json') as f:
            bc_data = json.load(f)
        with open(f'results/q2_dagger_{env}.json') as f:
            dagger_data = json.load(f)

        dagger_avg = np.array(dagger_data['avg_return'])
        dagger_std = np.array(dagger_data['std_return'])
        x = dagger_avg[:, 1]
        bc_avg = len(x) * [bc_data['avg_return'][0][2]]
        expert = len(x) * [expert_return[env]]
        plt.plot(x, dagger_avg[:, 2], label='Dagger')
        plt.plot(x, bc_avg, '--', label='Behavior Cloning')
        plt.plot(x, expert, label='Expert')
        plt.fill_between(x, dagger_avg[:, 2] - dagger_std[:, 2], dagger_avg[:, 2] + dagger_std[:, 2], alpha=0.2)
        plt.title(f'Dagger vs. Behavior Cloning ({env})')
        plt.xlabel('iterations')
        plt.ylabel('Avg. Return (n=10)')
        if log_scale:
            plt.yscale('log')
        plt.legend()
        plt.show()


    plot('Ant-v2')
    plot('Humanoid-v2', log_scale=True)


if __name__ == '__main__':
    plot_q2()