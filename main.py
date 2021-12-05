import numpy as np

from Agent import *
from matplotlib import pyplot as plt
from Problem import Problem, Dist


def plot_average(avg_rewards, dist_type, num, modes):
    """
    Plots a graph for the average performance of multiple agents per distribution.
    :param avg_rewards: List of average rewards over time per agent.
    :param dist_type: Distribution type
    """
    fig = plt.figure(figsize=(10, 8))
    for i in range(len(avg_rewards)):
        label = modes[i].name
        plt.plot(avg_rewards[i], label=label)
    plt.legend(loc='upper right')
    plt.xlabel("Time-step")
    plt.ylabel("Average Reward")
    plt.ylim([0, 1.1])
    plt.title("Average reward over " + str(num) + " runs")
    plt.show()
    fname = os.getcwd() + '/plots/avg/' + dist_type.name + '.png'
    fig.savefig(fname)


def run_and_plot_avg(dist_type=Dist.GAUSS, k=7, num=1000, modes=Mode.__iter__(),reward_dists=None):
    """
    Runs and plots the average results of multiple agents for every algorithm, on a certain distribution.
    :param dist_type: Reward distribution type
    :param k: Number of arms
    :param num: Number of agents
    :param modes: Modes to run the simulation with
    """
    avg_rewards = []  # contains average rewards over all agents per mode

    env = Problem(k, dist_type=dist_type, verbose=True, reward_dists=reward_dists)
    for mode in modes:
        rewards = None
        for _ in range(num):
            agent = Agent(env, mode=mode, epsilon=0.1, ucb_c=0.4, alpha=1, tau=5)
            agent.run(verbose=False, plot=False, max_steps=1000)
            if rewards is None:
                rewards = agent.average_rewards
            else:
                rewards = np.add(rewards, agent.average_rewards)
        rewards = [x / num for x in rewards]
        avg_rewards.append(rewards)
        print(mode.name, 'complete!')
    plot_average(avg_rewards, dist_type, num, modes)


def main():
    dist_type = Dist.BERNOULLI
    modes = [Mode.EPSILON_GREEDY]
    rewards_gauss = [[0.517, 0.133],
                     [0.675, 0.061],
                     [0.611, 0.125],
                     [0.614, 0.225],
                     [0.542, 0.291],
                     [0.617, 0.272],
                     [0.519, 0.278]]
    k = 7
    num = 1000
    run_and_plot_avg(dist_type, k, num, reward_dists=rewards_gauss)


if __name__ == "__main__":
    main()
