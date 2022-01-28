from agent import Agent, Selection, Learning
from env import ConnectX
from play import *
from plot import plot_average


def main():
    game = ConnectX()
    agentM = Agent(game, "M", learning=Learning.MC, selection=Selection.EPSILON_GREEDY)
    agentQ = Agent(game, "Q", learning=Learning.Q, selection=Selection.EPSILON_GREEDY)
    agent0 = agentM
    agent1 = Agent(game, "R", selection=Selection.RANDOM)
    agent2 = Agent(game, "B", selection=Selection.BRICK)
    winners, win_history = train_agent(game, agent0, epochs=10000, render=False)
    plot_average(win_history, agent0, (agent1.selection.name, agent1.learning.name))
    return
    train_agent(game, agent0, agent1, epochs=10000, render=False, debug=False)
    train_agent(game, agent0, agent2, epochs=10000, render=False)
    while True:
        play_agent(agent0)

        while True:
            ans = input("Play again? (y/n) ")
            if ans.lower() not in ('n', 'y'):
                print("Invalid answer!")
            elif ans.lower == 'n':
                return


if __name__ == "__main__":
    main()
