import numpy as np

from agent import Agent, Selection
from env import ConnectX
from play import train_agent
from plot import plot_hyperparameter


def run_tuning(selection, learning, opponent=None, hyperparam="alpha", lim=(0, 1), tune_num=10):
    """
    Tunes hyperparameters and plots results.
    :param opponent: Type of opponent. If None or invalid, selects random agent.
    :param lim: Lower and upper tune-space limit in format (low,high)
    :param hyperparam: Hyperparameter to tune
    :param selection: Agent selection mode
    :param learning: Agent learning mode
    :param tune_num: Number of hyperparameter values
    :param num: Number of iterations
    """

    # get hyperparameter tuning values
    tune_space = np.linspace(lim[0], lim[1], num=tune_num, endpoint=True)
    agents = []  # agents
    wins = []  # wins

    game = ConnectX()

    match opponent:
        case "brick":
            opponent = Agent(game, name="B", selection=Selection.BRICK)
        case "random":
            opponent = Agent(game, name="B", selection=Selection.RANDOM)
        case _:
            opponent = None

    for val in tune_space:
        agent0 = Agent(game, "A", selection=selection, learning=learning)
        agent0.set_hyperparameter(hyperparam, val)
        winners, win_history = train_agent(game, agent0, opponent,epochs=1000)

        agents.append(agent0)
        wins.append(win_history)

        game.reset()

    plot_hyperparameter(agents, hyperparam, tune_space, wins, selection.name, learning.name)
