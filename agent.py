import numpy as np


class Agent:
    """
    Agent class. Plays Connect 4 and learns from it.
    """

    def __init__(self, board, mark, columns=7, rows=6, goal=4):
        """
        Initializes an agent that can play Connect 4 with reinforcement learning enabled.
        :param board: Board state represented as a matrix of size rows x columns
        :param mark: Player number of agent, either "1" or "2"
        :param rows: Number of rows
        :param columns: Number of columns
        :param goal: Number of pieces in a row required to win
        """

        self.mark = mark
        self.columns = columns
        self.rows = rows
        self.goal = goal
        self.board = board
        self.current_sel = 0

    def lmaolol(self, obs, config):
        self.current_sel += 1
        self.current_sel %= 7
        return self.current_sel
