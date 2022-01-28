import sys
from random import shuffle

from agent import Agent, Selection
from env import ConnectX


def self_play(game, agent, render=False):
    """
    Runs game with agent game-play.
    :param game: ConnectX environment
    :param agent: Agent that plays against itself
    :param render: If True, prints board state and other info at every turn. Prints nothing if False.
    :returns the winner of the game.
    """
    marks = ("A", "B")  # player names
    reward = 0

    # initial action for both sides
    for mark in marks:
        agent.name = mark
        col = agent.select_action()  # select new action
        success, reward = game.step(col, agent.name)  # play action

    # loop until a winner is decided
    while game.get_winner() is None:

        # switches agent "perspective"
        agent.name = marks[game.turn % 2]

        # select action at+1 based on st+1
        col = agent.select_action()

        # put piece
        success, reward = game.step(col, agent.name)

        # update estimates based on known values
        agent.update_estimates(reward)

        # guards to make sure things are going smoothly
        if not success:  # failure to make move
            print(f"Move {col} by Agent {agent.name} failed!\n"
                  f"Currently valid moves: {game.valid_moves()}\n"
                  f"Current state:")
            game.print_state()
            sys.exit()

    # ending updates
    for copy in (True, False):  # first
        agent.name = marks[game.turn % 2]
        reward = game.get_reward(agent.name)
        agent.update_estimates(reward)
        if render:
            game.print_state()
        if copy:  # makes a copy of the end state (to update winning agent estimations)
            game.copy_end_state()

    return game.get_winner()


def agent_play(game, agent0, agent1, render=False):
    """
    Runs game with two agents duking it out to the death in an intense game of Connect 4.
    :param game: ConnectX environment
    :param agent0: Agent 0
    :param agent1: Agent 1
    :param render: If True, prints board state and other info at every turn. Prints nothing if False.
    :returns the winner of the game.
    """
    agents = [agent0, agent1]
    agent0.name = "A"
    agent1.name = "B"
    shuffle(agents)
    reward = 0

    # initial action for both sides
    for agent in agents:
        col = agent.select_action()  # select new action
        success, reward = game.step(col, agent.name)  # play action

    # loop until a winner is decided
    while game.get_winner() is None:

        # switches agent "perspective"
        agent = agents[game.turn % 2]

        # select action at+1 based on st+1
        col = agent.select_action()

        # put piece
        success, reward = game.step(col, agent.name)

        # update estimates based on known values
        agent.update_estimates(reward)

        # guards to make sure things are going smoothly
        if not success:  # failure to make move
            print(f"Move {col} by Agent {agent.name} failed!\n"
                  f"Currently valid moves: {game.valid_moves()}\n"
                  f"Current state:")
            game.print_state()
            sys.exit()

        # ending updates
        for copy in (True, False):  # first
            agent = agents[game.turn % 2]
            reward = game.get_reward(agent.name)
            agent.update_estimates(reward)
            if render:
                game.print_state()
            if copy:  # makes a copy of the end state (to update winning agent estimations)
                game.copy_end_state()

        return game.get_winner()


def run(game, agent0, agent1=None, render=False):
    """
    Runs game with given agents.
    :param game: Game environment. Instance of class ConnectX.
    :param agent0: Required agent.
    :param agent1: Second (optional) agent. If None chosen, self-play is used.
    :param render: If True, prints board state and other info at every turn. Prints nothing if False.
    :returns The winner of the game.
    """
    if agent1 is None:
        return self_play(game, agent0, render)
    else:
        return agent_play(game, agent0, agent1, render)


def train_agent(epochs=1000, render=False):
    game = ConnectX()
    agent0 = Agent(game, "A")
    agent1 = Agent(game, "B", selection=Selection.BRICK)
    print("Initializing training....")
    winners = {}
    for i in range(epochs):
        win = run(game, agent0, agent1, render=render)
        if win in winners.keys():
            winners[win] += 1
        else:
            winners[win] = 1
        game.reset()
        print(f"{round(i * 100 / epochs, 1)}%", end="\r")
    print(f"WINS:")
    for key in winners.keys():
        print(f"  {key}: {winners[key]}")

    return agent0



def main():
    agent0 = train_agent(epochs=100000, render=False)
    print(len(agent0.Qpairs))
    i = 0
    for key in agent0.Qpairs.keys():
        if 0 not in agent0.Qpairs.get(key):
            print(f"{key}: {agent0.Qpairs.get(key)}")
    play_agent(agent0)


if __name__ == "__main__":
    main()
