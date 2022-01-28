import sys
from random import shuffle


def self_play(game, agent, render=False):
    """
    Runs game with agent game-play.
    :param game: ConnectX environment
    :param agent: Agent that plays against itself
    :param render: If True, prints board state and other info at every turn. Prints nothing if False.
    :returns the winner of the game.
    """
    aname = agent.name  # agent original name
    marks = (aname, "Z")  # player names

    # initial action for both sides
    for mark in marks:
        agent.name = mark
        col = agent.select_action()  # select new action
        success = game.add_piece(col, agent.name)  # play action

    # loop until a winner is decided
    while game.get_winner() is None:
        # switches agent "perspective"
        agent.name = marks[game.turn % 2]

        # gets reward rt
        reward = game.get_reward(agent.name)

        # select action at+1 based on st+1
        col = agent.select_action()

        # put piece
        success = game.add_piece(col, agent.name)

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
    for mark in marks:
        agent.name = mark
        reward = game.get_reward(agent.name)
        agent.update_estimates(reward)

    if render:
        game.print_state()

    agent.name = aname
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
    shuffle(agents)

    # initial action for both sides
    for agent in agents:
        col = agent.select_action()  # select new action
        success = game.add_piece(col, agent.name)  # play action

    # loop until a winner is decided
    while game.get_winner() is None:

        # switches agent "perspective"
        agent = agents[game.turn % 2]

        # gets reward rt
        reward = game.get_reward(agent.name)

        # select action at+1 based on st+1
        col = agent.select_action()

        # put piece
        success = game.add_piece(col, agent.name)

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
    for agent in agents:
        reward = game.get_reward(agent.name)
        agent.update_estimates(reward)

    if render:
        game.print_state()

    return game.get_winner()


def run(game, agent0, agent1=None, render=False):
    """
    Runs game with given agents.
    :param game: Game environment. Instance of class ConnectX.
    :param agent0: Required agent.
    :param agent1: Second (optional) agent. If None chosen, self-play is used.
    :param render: If True, prints board state and other info at every turn. Prints nothing if False.
    :returns The winner of the game
    """
    if agent1 is None:
        return self_play(game, agent0, render)
    else:
        return agent_play(game, agent0, agent1, render)


def train_agent(game, agent0, agent1=None, epochs=10000, render=False):
    """
    Trains the agent over the given number of epochs.
    :param game: Game environment
    :param agent0: Required agent.
    :param agent1: Second optional agent. If None, self-play is used.
    :param epochs: Number of epochs to train the agent for.
    :param render: If True, prints the final state of each game
    :return:
    """

    # pretty print stuff
    print(f"{agent0.name} vs ", end="")
    if agent1 is None:
        print("itself: ", end="")
    else:
        print(f"{agent1.name}: ", end="")
    print("Initialize training....")

    # record-keeping
    win_history = []  # win history
    # create dictionary keeping track of total winner stats
    if agent1 is not None:
        winners = {agent0.name: 0, agent1.name: 0, "DRAW": 0}
    else:
        winners = {agent0.name: 0, "Z": 0, "DRAW": 0}

    # run epochs
    for i in range(epochs):
        win = run(game, agent0, agent1, render=render) # play game
        winners[win] += 1 # update winner count
        win_history.append(win) # update winner history

        game.reset()  # reset for next round

        print(f"{round(i * 100 / epochs, 1)}%", end="\r")  # pretty-prints a loading percentage to show how much

    # print training results
    print(f"WINS:")
    for key in winners.keys():
        print(f"  {key}: {winners[key]}")

    return winners, win_history


def play_agent(agent0):
    """
    Lets the human play the agent in the terminal. Returns nothing.
    :param agent0: Agent to play against
    """
    game = agent0.game
    game.reset()

    while game.get_winner() is None:
        acol = agent0.select_action()
        game.add_piece(acol, agent0.name)
        game.print_state()
        print(f"VALID MOVES: {game.valid_moves()}")
        while True and game.get_winner() is None:
            move = input("SELECT A COLUMN: ")
            try:
                move = int(move)
                success, reward = game.add_piece(move, "P")
            except ValueError:
                success = False
            if not success:
                print("Invalid move!")
            else:
                break

    game.print_state()
