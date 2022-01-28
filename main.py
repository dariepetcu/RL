from agent import Agent, Selection, Learning
from env import ConnectX
from play import *
from plot import plot_average
from tuning import run_tuning, tune_MC, tune_SARSA, tune_Q


def main():
    game = ConnectX()   # create playing environment with default (connect 4) settings
    agentM = Agent(game, "M", learning=Learning.MC, selection=Selection.EPSILON_GREEDY,
                   epsilon=0.11, alpha=0.13, gamma=0.56)    # create montecarlo agent
    agentQ = Agent(game, "Q", learning=Learning.Q, selection=Selection.EPSILON_GREEDY,
                   epsilon=0.11, alpha=0.13, gamma=0.99)    # create qlearning agent
    agentS = Agent(game, "S", learning=Learning.SARSA, selection=Selection.EPSILON_GREEDY,
                   epsilon=0.11, alpha=0.13, gamma=0.99)    # create sarsa agent

    agentR = Agent(game, "R", selection=Selection.RANDOM)   # create random-playin agent
    agentB = Agent(game, "B", selection=Selection.BRICK)    # create brick-laying agent
    RL_agents = [agentM, agentQ, agentS]
    dumb_agents = [agentR, agentB]

    for agent0 in RL_agents:
        for agent1 in dumb_agents:
            winners, win_history = train_agent(game, agent0, agent1, epochs=100000, render=False)
            plot_average(win_history, agent0, (agent1.selection.name, agent1.learning.name))

        # play_agent(agent0)


if __name__ == "__main__":
    main()
