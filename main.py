import sys
from random import shuffle, choice

from agent import Agent
from env import ConnectX


def self_play(game, agent, render=False):
    """
    Runs game with agent game-play.
    :param game: ConnectX environment
    :param agent: Agent that plays against itself
    :param render: If True, prints board state and other info at every turn. Prints nothing if False.
    :returns the winner of the game.
    """
    marks = ("A", "B") # player names
    reward = 0

    # initial action for both sides
    for mark in marks:
        if render:
            game.print_state()

        agent.name = mark
        col = agent.select_action()  # select new action
        success, reward = game.step(col, agent.name)  # play action  # starting config

    # loop until a winner is decided
    while game.get_winner() is None:

        if render:
            game.print_state()

        # switches agent "perspective"
        agent.name = marks[agent.turn % 2]

        # select action at+1 based on st+1
        col = agent.select_action()

        # update estimates based on known values
        agent.update_estimates(reward)

        # put piece
        success, reward = game.step(col, agent.name)

        # guards to make sure things are going smoothly
        if success:  # failure to make move
            print(f"Move {col} by Agent {agent.name} failed! Current state:")
            game.print_state()
            sys.exit()

    # ending updates
    for copy in (True, False):  # first
        agent.name = marks[agent.turn % 2]
        reward = game.get_reward(agent.name)
        agent.update_estimates(reward)
        if copy:  # makes a copy of the end state (to update winning agent estimations)
            game.copy_end_state()

    return game.get_winner()


def run(game, agent0, agent1=None, render=False):
    """
    Runs game with given agents.
    :param game: Game environment. Instance of class ConnectX.
    :param agent0: Required agent.
    :param agent1: Optional agent. If None chosen, game-play is used.
    If "random" selected, agent plays against a randomly selecting agent.
    :param render: If True, prints board state and other info at every turn. Prints nothing if False.
    :returns The winner of the game.
    """
    if agent1 is None:
        return self_play(game, agent0, render)

    agents = [agent0, agent1]
    shuffle(agents)  # shuffles agent order to make it random.
    while game.get_winner() is None:
        for ag in agents:
            if ag is "random":
                vmoves = game.valid_moves()
                game.add_piece(choice(vmoves), "R")
            else:
                game.add_piece(ag.play(), ag.name)
    return game.get_winner()


def train_agent(epochs=1000):
    game = ConnectX()
    agent0 = Agent(game, "A")
    agent1 = None
    winners = {}
    for _ in range(epochs):
        win = run(game, agent0, agent1)
        if win in winners.keys():
            winners[win] += 1
        else:
            winners[win] = 0
    print(f"WINS:")
    for key in winners.keys():
        print(f"  {key}: {winners[key]}")


def main():
    train_agent()


if __name__ == "__main__":
    main()
