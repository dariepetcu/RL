import sys

import numpy as np


class ConnectX:

    def __init__(self, dim=(7, 6), goal=4):
        """
        Initializes a connect X game with the given goal and dimensions.
        :param dim: (x,y) tuple of game dimensions
        :param goal: number of connected pieces required
        """
        self.moves = []  # all moves so far
        self.columns = dim[0]  # sets column count
        self.rows = dim[1]  # sets row count
        self.goal = goal  # sets goal
        self.board = [[None] * self.columns] * self.columns
        self.turn = 0  # number of turns since start
        self.winner = None  # player that won the game

    def valid_moves(self):
        """
        Returns list of legal moves.
        :returns list of legal/valid columns with space in them
        """
        legal = []
        for c in range(self.columns):
            if None in self.board[c]:
                legal.append(c)
        print(legal)
        return legal

    def drop_piece(self, col, piece):
        """
        Creates a board state with a new piece added
        :param col: Piece column
        :param piece: Piece mark
        :returns Row of the new piece
        """
        for row in range(self.rows):
            r = self.rows - row
            if self.board[col][r] is None:
                self.board[col][r] = piece
                return r

    def get_negative_diagonal(self, prow, pcol):
        """
        Get the negative diagonal line of the new piece
        :param prow: Piece row
        :param pcol: Piece column
        :returns List of pieces
        """
        nd_pieces = []  # positive diagonal pieces

        ndrow = prow
        ndcol = pcol

        while ndrow > 0 and ndcol > 0:
            ndrow -= 1
            ndcol -= 1

        while ndrow < self.rows and ndcol < self.columns:
            nd_pieces.append(self.board[ndcol][ndrow])
            ndcol += 1
            ndrow += 1

        return nd_pieces

    def get_positive_diagonal(self, prow, pcol):
        """
        Get the positive diagonal line of the new piece
        :param prow: Piece row
        :param pcol: Piece column
        :returns List of pieces
        """
        pd_pieces = []  # positive diagonal pieces

        pdrow = prow
        pdcol = pcol

        while pdrow < self.rows and pdcol > 0:
            pdrow += 1
            pdcol -= 1

        while pdrow > 0 and pdcol < self.columns:
            pd_pieces.append(self.board[pdcol][pdrow])
            pdrow -= 1
            pdcol += 1

        return pd_pieces

    def get_row(self, prow, pcol):
        """
        Gets the pieces in the piece row
        :param prow: Piece row
        :param pcol: Piece column
        :returns List of pieces in the row
        """

        row_pieces = []  # row pieces
        for c in range(self.columns):
            row_pieces.append(self.board[c][prow])

        return row_pieces

    # Returns True if dropping piece in column results in game win
    def check_win(self, prow, pcol, mark):
        """
        Checks if the player has won based on the new piece position
        :param prow: Row of new piece
        :param pcol: Column of new piece
        :param mark: Player mark
        :returns True if it is a winning move, False if it is not a winning move
        """

        col_pieces = self.board[pcol]
        win = [mark] * self.goal

        # check horizontal/vertical
        if win in col_pieces or win in self.get_row(prow, pcol):
            return True

        # check diagonals
        if win in self.get_negative_diagonal(prow, pcol) or win in self.get_positive_diagonal(prow, pcol):
            return True

        return False

    def add_piece(self, column, player):
        """
        Attempts to add a piece to the board.
        :param column: Column to add a board to.
        :param player: Mark of the mark adding a piece.
        :returns True
        """

        if self.winner is not None:
            return False

        if len(player) > 1:
            sys.exit(f"{player}: name too long!")
        if player == "0":
            print("Player name cannot be 0!")

        if column in self.valid_moves():
            prow = self.drop_piece(column, player)
            self.turn += 1
            if self.check_win(prow, column, player):
                self.winner = player
            self.moves.append(column)
            return True
        else:
            print(f"{column}: Invalid move!")
            return False

    def get_winner(self):
        """
        Getter. Returns game winner.
        :return: Game winner
        """
        return self.winner

    def get_moves(self):
        """
        Returns move history of all players.
        :returns Move history.
        """
        return self.moves

    def print_state(self):
        for r in range(self.columns):
            for c in range(self.rows):
                mark = self.board[c][r]

                if mark is None:
                    print(" 0 ", end='')
                else:
                    print(f" {mark} ", end='')
            print("")

        print(f"WINNER: {self.winner}")
