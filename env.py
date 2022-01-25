import sys

import numpy as np

class ConnectX:

    def __init__(self, dim=(7, 6), goal=4):
        """
        Initializes a connect X game with the given goal and dimensions.
        :param dim: (x,y) tuple of game dimensions
        :param goal: number of connected pieces required
        """
        self.dimx = dim[0] # sets x dimension
        self.dimy = dim[1] # sets y dimension
        self.goal = goal # sets goal
        self.board = np.empty(self.dimx, self.dimy) # initializes empty board
        self.turn = 0 # number of turns since start

    def valid_move(self, column):
        """
        Determines whether or not the move requested by the agent is valid.
        :param column: Column to which the agent wants to add a piece to.
        :returns True if valid, False if invalid.
        """
        if self.board[self.dimx][column] is None:
            return True
        else:
            return False

    def four_in_a_row(self, row, column):
        """
        Checks if the winning condition has been fulfilled.
        :param row:
        :param column:
        :return:
        """
        # Get a list of horizontal, vertical, and diagonal lines involving the piece that was just added
        lines = []

        # Horizontal & vertical
        lines.append(self.board[:, column])
        lines.append(self[row])

        # Diagonal
        w = self.shape[1] - 1
        h = self.shape[0] - 1
        distance_to_left = min([row, column]) * -1
        distance_to_right = min([h - row, w - column]) + 1
        lines.append([self[row + n, column + n] for n in range(distance_to_left, distance_to_right)])
        distance_to_left = min([row, w - column]) * -1
        distance_to_right = min([h - row, column]) + 1
        lines.append([self[row + n, column - n] for n in range(distance_to_left, distance_to_right)])

        # Split each list into chunks of four and check each to see if all values are a) the same and b) not zero
        fours = []
        for line in lines:
            for n in range(len(line) - 3):
                four = line[n:n + 4]
                if (four[0] == four).all() & (four[0] != 0):
                    return True
        return False



    def add_piece(self, column, player):
        """
        Attempts to add a piece to the board.
        :param column: Column to add a board to.
        :param player: Initial of the player adding a piece.
        :returns True if successful, False if failure, "WIN" if it was a winning move
        """
        if len(player) > 1:
            sys.exit(f"{player}: name too long!")

        if self.valid_move(column):
            for y in range(self.dimy):
                if self.board[column][y] is None:
                    self.board[column][y] = player
                    if self.is_win((column, y)):
                        return "WIN"
                    return

        return False