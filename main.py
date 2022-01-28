import sys
from random import shuffle

from agent import Agent, Selection, Learning
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
    turns = {"A": 0, "B": 0}
    reward = 0

    # initial action for both sides
    for mark in marks:
        agent.name = mark
        agent.turns = turns[mark]

        col = agent.select_action()  # select new action
        success, reward = game.step(col, agent.name)  # play action
        if success:
            turns[agent.name] += 1
            agent.turns += 1

    # loop until a winner is decided
    while game.get_winner() is None:

        # switches agent "perspective"
        agent.name = marks[game.turn % 2]
        agent.turns = turns[agent.name]

        # select action at+1 based on st+1
        col = agent.select_action()

        # put piece
        success, reward = game.step(col, agent.name)
        if success:
            turns[agent.name] += 1
            agent.turns += 1

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
    # agent0.name = "A"
    # agent1.name = "B"
    agents = [agent0, agent1]
    shuffle(agents)
    reward = 0

    # initial action for both sides
    for agent in agents:
        col = agent.select_action()  # select new action
        success, reward = game.step(col, agent.name)  # play action
        if success:
            agent.turns += 1

    # loop until a winner is decided
    while game.get_winner() is None:

        # switches agent "perspective"
        agent = agents[game.turn % 2]

        # select action at+1 based on st+1
        col = agent.select_action()

        # put piece
        success, reward = game.step(col, agent.name)
        if success:
            agent.turns += 1

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
    if agent1 is not None:
        winners = {agent0.name: 0, agent1.name: 0, "DRAW": 0}
    else:
        winners = {"A": 0, "B": 0, "DRAW": 0}
    for i in range(epochs):
        win = run(game, agent0, agent1, render=render)
        winners[win] += 1
        game.reset()
        print(f"{round(i * 100 / epochs, 1)}%", end="\r")
    print(f"WINS:")
    for key in winners.keys():
        print(f"  {key}: {winners[key]}")

    return winners


def play_agent(agent0):
    game = ConnectX()
    while game.get_winner() is None:
        acol = agent0.select_action()
        game.step(acol, agent0.name)
        game.print_state()
        print(f"VALID MOVES: {game.valid_moves()}")
        while True and game.get_winner() is None:
            move = input("SELECT A COLUMN: ")
            try:
                move = int(move)
                success, reward = game.step(move, "P")
            except ValueError:
                success = False
            if not success:
                print("Invalid move!")
            else:
                break
    game.print_state()

def main():
    game = ConnectX()
    agent0 = Agent(game, "M", learning=Learning.MC, selection=Selection.EPSILON_GREEDY)
    agent1 = Agent(game, "R", selection=Selection.RANDOM)
    agent2 = Agent(game, "B", selection=Selection.BRICK)
    train_agent(game, agent0, agent1, epochs=100000, render=False)
    train_agent(game, agent0, agent2, epochs=100000, render=False)
    train_agent(game, agent0, epochs=100000, render=False)
    play_agent(agent0)


if __name__ == "__main__":
    main()
