import os
from Agent import *
from matplotlib import pyplot as plt
from Problem import Problem, Dist


def plot_average(avg_rewards, dist_type, num, plot_acc):
    """
    Plots a graph for the average performance of multiple agents per distribution.
    :param plot_acc: If True, plots the average accuracy w.r.t best action. Otherwise plots the average reward
    :param num: Number of iterations
    :param avg_rewards: List of average rewards over time per agent.
    :param dist_type: Distribution type
    """
    fig = plt.figure(figsize=(10, 8))
    for i in range(len(avg_rewards)):
        label = Mode(i).name
        plt.plot(avg_rewards[i], label=label)
    plt.legend(loc='upper right')
    if not plot_acc:
        plt.xlabel("Time-step")
        plt.ylabel("Average Reward")
        plt.title("Average reward over " + str(num) + " runs for the " + dist_type.name.title() + " distribution")
        plt.show()
        fname = os.getcwd() + '/plots/avg/' + dist_type.name + '.png'
        fig.savefig(fname)
    else:
        plt.xlabel("Time-step")
        plt.ylabel("Arm accuracy")
        plt.title("Average accuracy over " + str(num) + " runs for the " + dist_type.name.title() + " distribution")
        plt.show()
        fname = os.getcwd() + '/plots/acc/' + dist_type.name + '.png'
        fig.savefig(fname)


def run_and_plot_avg(dist_type=Dist.GAUSS, k=7, num=1000):
    """
    Runs and plots the average results of multiple agents for every algorithm, on a certain distribution.
    :param dist_type: Reward distribution type
    :param k: Number of arms
    :param num: Number of agents
    """
    avg_rewards = []  # contains average rewards over all agents per mo
    accuracies = []
    for mode in Mode:
        rewards = None
        selections = None
        for _ in range(num):
            # initialize bandit problem
            env = Problem(k, dist_type=dist_type)

            # initialize agent using optimal parameters based on hyperparameter tuning
            agent = Agent(env, mode=mode, epsilon=.38, ucb_c=0.38, alpha=.9, tau=0.12)

            # run agent and update rewards/learning
            agent.run(verbose=False, max_steps=1000)
            if rewards is None:
                rewards = agent.average_rewards
            else:
                rewards = np.add(rewards, agent.average_rewards)
            if selections is None:
                selections = agent.accuracy
            else:
                selections = np.add(selections, agent.accuracy)
        # get average learning/reward
        rewards = [x / num for x in rewards]
        selections = [x / num for x in selections]
        avg_rewards.append(rewards)
        accuracies.append(selections)
        print(mode.name, 'complete!')

    # plot performance
    plot_average(avg_rewards, dist_type, num, False)
    plot_average(accuracies, dist_type, num, True)


def plot_hyperparameter(avg_rewards, mode, num, tune_space):
    """
    Plots a graph for the average performance of multiple agents for one distribution.
    :param tune_space: Values for which agents were trained
    :param mode: Learning for which hyperparameters were tuned
    :param num: Number of iterations
    :param avg_rewards: List of average rewards over time per agent.
    """
    # create figure and plot
    fig = plt.figure(figsize=(10, 8))
    # plot hyperparam val for 0.1, 0.2, ..
    i = 0
    for value in tune_space:
        plt.plot(avg_rewards[i], label=round(value, 2))
        i += 1
    plt.legend(loc='upper right')
    plt.xlabel("Time-step")
    plt.ylabel("Average Reward")
    plt.title("Tuning of parameters for " + mode.name + " over " + str(num) + " runs")
    plt.show()
    # save
    fname = os.getcwd() + '/plots/tuning/' + mode.name + '.png'
    fig.savefig(fname)


def run_tuning(mode, dist_type, tune_num=10, num=300):
    """
    Tunes hyperparameters and plots results.
    :param mode: Learning for which params are tuned
    :param dist_type: Distribution to use
    :param tune_num: Number of hyperparameter values
    :param num: Number of iterations
    """
    # get hyperparameter tuning values
    tune_space = np.linspace(0, 1, num=tune_num, endpoint=True)

    rewards = None
    avg_rewards = []

    match mode:  # matches correct algorithm, each one is a copy paste for a different parameter
        case Mode.EPSILON_GREEDY:
            for val in tune_space:
                for _ in range(num):
                    env = Problem(7, dist_type=dist_type, verbose=False)
                    agent = Agent(env, mode=mode, epsilon=val)
                    agent.run(verbose=False, max_steps=1000)
                    if rewards is None:
                        rewards = agent.average_rewards
                    else:
                        rewards = np.add(rewards, agent.average_rewards)
                rewards = [x / num for x in rewards]
                avg_rewards.append(rewards)
        case Mode.SOFTMAX:
            for val in tune_space:
                for _ in range(num):
                    env = Problem(7, dist_type=dist_type, verbose=False)
                    agent = Agent(env, mode=mode, tau=val)
                    agent.run(verbose=False, max_steps=1000)
                    if rewards is None:
                        rewards = agent.average_rewards
                    else:
                        rewards = np.add(rewards, agent.average_rewards)
                rewards = [x / num for x in rewards]
                avg_rewards.append(rewards)
        case Mode.ACTION_PREFERENCES:
            for val in tune_space:
                for _ in range(num):
                    env = Problem(7, dist_type=dist_type, verbose=False)
                    agent = Agent(env, mode=mode, alpha=val)
                    agent.run(verbose=False, max_steps=1000)
                    if rewards is None:
                        rewards = agent.average_rewards
                    else:
                        rewards = np.add(rewards, agent.average_rewards)
                rewards = [x / num for x in rewards]
                avg_rewards.append(rewards)
        case Mode.UCB:
            for val in tune_space:
                for _ in range(num):
                    env = Problem(7, dist_type=dist_type, verbose=False)
                    agent = Agent(env, mode=mode, ucb_c=val)
                    agent.run(verbose=False, max_steps=1000)
                    if rewards is None:
                        rewards = agent.average_rewards
                    else:
                        rewards = np.add(rewards, agent.average_rewards)
                rewards = [x / num for x in rewards]
                avg_rewards.append(rewards)
        case _:
            print("Invalid algorithm!")

    # plot hyperparameter graph
    plot_hyperparameter(avg_rewards, mode, num, tune_space)
