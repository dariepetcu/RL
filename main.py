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
    opponent = agent
    marks = ["A", "B"] # player names
    i = 0 # used to alternate between player "A" and "B"
    success, next_state, reward, done = False, game.get_state(), 0, False # starting config
    while game.get_winner() is None:
        agent.name = marks[i % 2]
        agent.update_estimations(reward)
        agent.select_action()
        i += 1



def run(game, agent0, agent1=None, render=False):
    """
    Runs game with given agents.
    :param agent0: Required agent.
    :param agent1: Optional agent. If None chosen, game-play is used.
    If "random" selected, agent plays against a randomly selecting agent.
    :param render: If True, prints board state and other info at every turn. Prints nothing if False.
    :returns The winner of the game.
    """
    if agent1 is None:
        self_play(game, agent0, render)

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
    agent = Agent(game, "A")



if __name__ == "__main__":
    main()
