import numpy as np

from Agent import *
from matplotlib import pyplot as plt
from Problem import Problem, Dist


def plot_average(avg_rewards, dist_type, num):
    """
    Plots a graph for the average performance of multiple agents per distribution.
    :param avg_rewards: List of average rewards over time per agent.
    :param dist_type: Distribution type
    """
    fig = plt.figure(figsize=(10, 8))
    for i in range(len(avg_rewards)):
        label = Mode(i).name
        plt.plot(avg_rewards[i], label=label)
    plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.xlabel("Time-step")
    plt.ylabel("Average Reward")
    plt.ylim([0, 1.1])
    plt.title("Average reward over " + num + " runs")
    plt.show()
    fname = os.getcwd() + '/plots/avg/' + dist_type.name + '.png'
    fig.savefig(fname)


def run_agent(dist_type, mode, k):
    """
    Runs the agent with the given parameters
    :param dist_type: Distribution type
    :param mode: Selection mode
    :param k: Number of arms
    """

    env = Problem(k, dist_type=dist_type, verbose=False)
    agent = Agent(env, mode=mode, epsilon=.1)
    agent.run(verbose=False, plot=False, max_steps=1000)


def run_and_plot_avg(dist_type=Dist.GAUSS, k=1000, num=1000):
    """
    Runs and plots the average results of multiple agents for every algorithm, on a certain distribution.
    :param dist_type: Reward distribution type
    :param k: Number of arms
    :param num: Number of agents
    :param modes: Modes to run the simulation with
    """
    avg_rewards = []  # contains average rewards over all agents per mode
    env = Problem(k, dist_type=dist_type, verbose=False)
    for mode in Mode:
        rewards = None
        for _ in range(num):
            agent = Agent(env, mode=mode, epsilon=.1)
            agent.run(verbose=False, plot=False, max_steps=1000)
            if rewards is None:
                rewards = agent.average_rewards
            else:
                rewards = np.add(rewards, agent.average_rewards)
        rewards = [x / num for x in rewards]
        avg_rewards.append(rewards)
        print(mode.name, 'complete!')
    plot_average(avg_rewards, dist_type, num)


def main():
    dist_type = Dist.GAUSS
    mode = Mode.EPSILON_GREEDY
    k = 7
    num = 1000
    run_and_plot_avg(dist_type, k, num)
    # run_agent(dist_type, mode, k)


if __name__ == "__main__":
    main()
