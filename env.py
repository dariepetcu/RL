import sys

import numpy as np



class ConnectX:

    def __init__(self, dim=(7, 6), goal=4):
        """
        Initializes a connect X game with the given goal and dimensions.
        :param dim: (x,y) tuple of game dimensions
        :param goal: number of connected pieces required
        """
        self.dimx = dim[0]  # sets x dimension
        self.dimy = dim[1]  # sets y dimension
        self.goal = goal  # sets goal
        self.board = np.empty(self.dimx, self.dimy)  # initializes empty board
        self.turn = 0  # number of turns since start

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

    def drop_piece(self, col, piece, config):
        next_grid = self.board.copy()
        for y in range(self.dimy):
            if next_grid[col][y] is None:
                next_grid[col][y] = None
        return next_grid

    # Returns True if dropping piece in column results in game win
    def check_winning_move(self, config, col, piece):
        # Convert the board to a 2D grid
        grid = np.asarray(self.board).reshape(config.rows, config.columns)
        next_grid = self.drop_piece( col, piece, config)
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[row, col:col + config.inarow])
                # print(config.inarow,"\n",window)
                if window.count(piece) == config.inarow:
                    return True
        # vertical
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(next_grid[row:row + config.inarow, col])
                if window.count(piece) == config.inarow:
                    return True
        # positive diagonal
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if window.count(piece) == config.inarow:
                    return True
        # negative diagonal
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if window.count(piece) == config.inarow:
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
                    if self.check_winning_move((column, y)):
                        return "WIN"
                    return

        return False
