import os

import numpy as np

from Agent import *
from matplotlib import pyplot as plt
from Problem import Problem, Dist


def plot_average(avg_rewards, dist_type, num):
    """
    Plots a graph for the average performance of multiple agents per distribution.
    :param num: Number of iterations
    :param avg_rewards: List of average rewards over time per agent.
    :param dist_type: Distribution type
    """
    fig = plt.figure(figsize=(10, 8))
    for i in range(len(avg_rewards)):
        label = Mode(i).name
        plt.plot(avg_rewards[i], label=label)
    plt.legend(loc='upper right')
    plt.xlabel("Time-step")
    plt.ylabel("Average Reward")
    plt.title("Average reward over " + str(num) + " runs")
    plt.show()
    fname = os.getcwd() + '/plots/avg/' + dist_type.name + '.png'
    fig.savefig(fname)

def plot_hyperparameter(avg_rewards, mode, num, tune_space):
    """
    Plots a graph for the average performance of multiple agents per distribution.
    :param num: Number of iterations
    :param avg_rewards: List of average rewards over time per agent.
    :param dist_type: Distribution type
    """
    fig = plt.figure(figsize=(10, 8))
    # plot hyperparam val for 0.1, 0.2, ..
    i = 0
    for value in tune_space:
        plt.plot(avg_rewards[i], label = round(value,2))
        i += 1
    plt.legend(loc='upper right')
    plt.xlabel("Time-step")
    plt.ylabel("Average Reward")
    plt.title("Tuning of parameters for " + mode.name + " over " + str(num) + " runs")
    plt.show()
    fname = os.getcwd() + '/plots/tuning/' + mode.name + '.png'
    fig.savefig(fname)


def run_and_plot_avg(dist_type=Dist.GAUSS, k=7, num=1000, reward_dists=None):
    """
    Runs and plots the average results of multiple agents for every algorithm, on a certain distribution.
    :param dist_type: Reward distribution type
    :param k: Number of arms
    :param num: Number of agents
    """
    avg_rewards = []  # contains average rewards over all agents per mode

    env = Problem(k, dist_type=dist_type, verbose=True, reward_dists=reward_dists)
    for mode in Mode:
        rewards = None
        for _ in range(num):
            # using optimal parameters based on hyperparameter tuning
            agent = Agent(env, mode=mode, epsilon=.3, ucb_c=0.7, alpha=.02, tau=0.5)
            agent.run(verbose=False, plot=False, max_steps=1000)
            if rewards is None:
                rewards = agent.average_rewards
            else:
                rewards = np.add(rewards, agent.average_rewards)
        rewards = [x / num for x in rewards]
        avg_rewards.append(rewards)
        print(mode.name, 'complete!')
    plot_average(avg_rewards, dist_type, num)



def run_tuning(mode, dist_type, reward_dists, tune_num=9, num=100):
    """
    Tunes hyperparameters and plots results
    :param mode: Mode for which params are tuned
    :param dist_type: Distribution to use
    :param reward_dists: Reward distributions
    :param tune_num: Number of iterations
    """
    env = Problem(7, reward_dists, dist_type=dist_type, verbose=True)
    tune_space = np.linspace(.2, .4, num=tune_num, endpoint=True)

    rewards = None
    avg_rewards = []

    match mode:
        case Mode.EPSILON_GREEDY:
            for val in tune_space:
                for _ in range(num):
                    agent = Agent(env, mode=mode, epsilon=val)
                    agent.run(verbose=False, plot=False, max_steps=1000)
                    if rewards is None:
                        rewards = agent.average_rewards
                    else:
                        rewards = np.add(rewards, agent.average_rewards)
                rewards = [x / num for x in rewards]
                avg_rewards.append(rewards)
        case Mode.SOFTMAX:
            for val in tune_space:
                for _ in range(num):
                    agent = Agent(env, mode=mode, tau=val)
                    agent.run(verbose=False, plot=False, max_steps=1000)
                    if rewards is None:
                        rewards = agent.average_rewards
                    else:
                        rewards = np.add(rewards, agent.average_rewards)
                rewards = [x / num for x in rewards]
                avg_rewards.append(rewards)
        case Mode.ACTION_PREFERENCES:
            for val in tune_space:
                for _ in range(num):
                    agent = Agent(env, mode=mode, alpha=val)
                    agent.run(verbose=False, plot=False, max_steps=1000)
                    if rewards is None:
                        rewards = agent.average_rewards
                    else:
                        rewards = np.add(rewards, agent.average_rewards)
                rewards = [x / num for x in rewards]
                avg_rewards.append(rewards)
        case Mode.UCB:
            for val in tune_space:
                for _ in range(num):
                    agent = Agent(env, mode=mode, ucb_c=val)
                    agent.run(verbose=False, plot=False, max_steps=1000)
                    if rewards is None:
                        rewards = agent.average_rewards
                    else:
                        rewards = np.add(rewards, agent.average_rewards)
                rewards = [x / num for x in rewards]
                avg_rewards.append(rewards)
        case _:
            print("ooga booga you're an idiot")

    plot_hyperparameter(avg_rewards,mode,num,tune_space)

def main():
    dist_type = Dist.GAUSS
    rewards0 = [[0.517, 0.133],
                     [0.675, 0.061],
                     [0.611, 0.125],
                     [0.614, 0.225],
                     [0.542, 0.291],
                     [0.617, 0.272],
                     [0.519, 0.278]]
    rewards1 = [(0, 1), (1, 1), (5, 2), (-3, 2), (3, 8), (5,2), (-1,1)]
    rewards2 = [0.1,0.2,0.3,0.4,0.5,.6,.7]
    k = 7
    num = 1000
    mode = Mode.UCB
    #run_and_plot_avg(dist_type, k, num, reward_dists=rewards1)
    run_tuning(mode,dist_type,reward_dists=rewards1)

if __name__ == "__main__":
    main()
