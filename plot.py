import types
from os.path import isfile

from matplotlib import pyplot as plt
from os import getcwd as cwd


def avg_winrate(win_history, amark):
    """
    Returns win history for each name in the
    :param win_history: Win history.
    :param amark: Agent mark
    :returns Average winrate for agent, average losses (i.e opponent winrate), and average draw percentage.
    """
    winrate = []  # agent winrate over epochs
    lossrate = []  # agent lossrate (opponent winrate) over epochs
    drawrate = []  # draw percentage over epochs

    wins = 0  # agent win count
    losses = 0  # agent loss (opponent win) count
    draws = 0  # draw count

    marks = types.SimpleNamespace()
    marks.agent = amark
    marks.draw = "DRAW"

    epoch = 1
    for mark in win_history:
        match mark:
            case marks.agent:
                wins += 1
            case marks.draw:
                draws += 1
            case _:
                losses += 1

        epoch += 1
        winrate.append((wins / epoch) * 100)
        drawrate.append((draws / epoch) * 100)
        lossrate.append((losses / epoch) * 100)

    return winrate, lossrate, drawrate


def plot_average(win_history, agent, oparams=(None, None)):
    """
    Plots a graph for the win percentages of an agent over time.
    :param agent: Agent mark.
    :param oparams: Opponent parameters in form (Selection,Learning). If both None, assumes self-learning.
    :param win_history: Win per epoch.
    """
    # initialize plot
    fig = plt.figure(figsize=(10, 8))

    # get epoch count
    epochs = len(win_history)

    # get plot title and plot (file) name
    title = f"Average outcomes for RL-Agent (S={agent.selection.name}, L={agent.learning.name})"  # plot title
    fname = f"{agent.selection.name}-{agent.learning.name}_"  # plot (file) name
    if oparams != (None, None):
        fname += f"vs_{oparams[0]}-{oparams[1]}"
        title += f" versus opponent (S={oparams[0]}, L={oparams[1]})"
    else:
        fname += f"selfplay"
        title += " versus itself"

    # plot various rates
    winrate, lossrate, drawrate = avg_winrate(win_history, agent.name)
    plt.plot([0, epochs], [50, 50], c='grey', linestyle='--')
    plt.plot(winrate, label="Win rate")
    plt.plot(lossrate, label="Loss rate")
    plt.plot(drawrate, label="Draw rate")

    # display legend
    plt.legend(loc='upper right')

    # configure plot
    plt.xlabel("Epoch")
    plt.ylabel("Outcome rate (%)")
    plt.xlim(0, epochs)  # epoch count
    plt.ylim(0, 100)  # percentage
    plt.title(title)

    # save plot
    fcount = 0
    while isfile(f"{cwd()}/plots/{fname}_ATTEMPT{fcount}.png"):
        fcount += 1
    fname = f"{cwd()}/plots/{fname}_ATTEMPT{fcount}.png"
    fig.savefig(fname)

    # show plot
    # plt.show()
