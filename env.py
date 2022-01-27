import sys


class ConnectX:

    def __init__(self, dim=(7, 6), goal=4):
        """
        Initializes a connect X game with the given goal and dimensions.
        :param dim: (x,y) tuple of game dimensions
        :param goal: number of connected pieces required
        """
        self._moves = []  # all _moves so far
        self.columns = dim[0]  # sets column count
        self.rows = dim[1]  # sets row count
        self.goal = goal  # sets goal
        self._board = ['0'] * self.columns * self.rows  # game board
        self._history = [self._board]  # state history
        self.turn = 0  # number of turns since start
        self._winner = None  # mark that won the game

    def reset(self):
        """
        Resets the game to the starting specifications.
        """

        self._board = ['0'] * self.columns * self.rows  # game board
        self.turn = 0  # number of turns since start
        self._winner = None  # mark that won the game
        self._moves = []
        self._history = [self._board]

    def valid_moves(self):
        """
        Returns list of legal _moves.
        :returns list of legal/valid columns with space in them
        """
        legal = []
        for c in range(self.columns):
            i = c
            while i < len(self._board):
                if self._board[i] == '0':
                    legal.append(c)
                    break
                i += self.columns

        return legal

    def _drop_piece(self, col, piece):
        """
        Updates a game state with a new piece added
        :param col: Piece column
        :param piece: Piece mark
        :returns Row of the new piece
        """
        i = col
        free = col  # highest-up free spot in column
        while i < len(self._board):
            if self._board[i] == '0':
                free = i
            i += self.columns

        self._board[free] = piece
        return free

    def _get_ndiag(self, pos):
        """
        Get the negative diagonal line of the new piece
        :param pos: Piece position on _board as index of game._board
        :returns List of pieces
        """
        i = pos
        step = self.columns + 1

        nd_pieces = []

        while i % self.columns != 0 and not i < self.columns:
            i -= step

        count = 0
        while i < len(self._board) and count < self.columns:
            nd_pieces.append(self._board[i])
            # check if the counter isn't "looping back" to the first row from the last
            if i % self.columns > (i + step) % self.columns:
                break
            else:
                i += step
                count += 1

        return nd_pieces

    def _get_pdiag(self, pos):
        """
        Get the positive diagonal line of the new piece
        :param pos: Piece position on _board as index of game._board
        :returns List of pieces
        """

        pd_pieces = []
        i = pos
        step = self.columns - 1

        while i % self.columns != 0 and not i < self.columns:
            i -= step

        count = 0
        while i < len(self._board) and count < self.columns:
            pd_pieces.append(self._board[i])

            # check if the counter isn't "looping back" to the last row from the first
            if i % self.columns < (i + step) % self.columns:
                break
            else:
                i += step
                count += 1

        return pd_pieces

    def _get_row(self, pos):
        """
        Gets the pieces in the piece row
        :param pos: Piece position on _board as index of game._board
        :returns List of pieces in the row
        """

        row_pieces = []  # row pieces
        i = pos

        while i % self.columns != 0:
            i -= 1

        start = i
        while i < len(self._board) and i - start < self.columns:
            row_pieces.append(self._board[i])
            i += 1

        return row_pieces

    def _get_col(self, pos):
        """
        Gets the pieces in the piece column
        :param pos: Piece position on _board as index of game._board
        :returns List of pieces in the columns
        """
        col_pieces = []
        i = pos % self.columns
        while i < len(self._board):
            col_pieces.append(self._board[i])
            i += self.columns
        return col_pieces

    # Returns True if dropping piece in column results in game win
    def _check_win(self, pos, mark):
        """
        Checks if the mark has won based on the new piece position
        :param pos: Piece position on _board as index of game._board
        :param mark: Player mark
        :returns True if it is a winning move, False if it is not a winning move
        """

        get_sublist = [self._get_row, self._get_col, self._get_ndiag, self._get_pdiag]
        for get in get_sublist:
            sublist = get(pos)
            i, j = 0, 0  # sublist counter and win counter
            while i < len(sublist) and j < self.goal:
                if sublist[i] == mark:
                    j += 1
                else:
                    j = 0
                i += 1
            if j == self.goal:
                return True

        return False

    def add_piece(self, column, mark):
        """
        Attempts to add a piece to the game.
        :param column: Column to add a game to.
        :param mark: Single-character mark of the mark adding a piece.
        :returns True
        """

        if self._winner is not None:
            return False

        if len(mark) > 1:
            sys.exit(f"{mark}: name too long!")
        if mark == "0":
            sys.exit("Player name cannot be 0!")

        if column in self.valid_moves():
            pos = self._drop_piece(column, mark)  # update _board
            self.turn += 1  # update turn count

            if self._check_win(pos, mark):  # check if this is a winning move
                self._winner = mark
            elif '0' not in self._board:  # check if its a draw
                self._winner = "DRAW"

            self._moves.append((mark, column))  # update move history
            self._history.append(self._board) # update state history
            return True
        else:
            print(f"{column}: Invalid move!")
            return False

    def get_reward(self, mark):
        """
        Returns reward for the game.
        :param mark: Player mark
        :returns 1 if the given mark won, -1 if they lost, otherwise 0.
        """
        if self._winner is None:
            return 0
        elif self._winner == mark:
            return 1
        else:
            return -1

    def step(self, col, mark):
        """
        Runs one step of the game for the given agent. Updates board and provides additional information.
        :param col: Agent's select column
        :param mark: Agent name
        :returns Move success, Result state, reward, and whether or not the game is done.
        """
        success = self.add_piece(mark, col)  # adds piece, updates winner, turn, board, returns move validity
        reward = self.get_reward(mark)  # mark "reward" for the state
        return success, reward

    def get_winner(self):
        """
        Getter. Returns game _winner.
        :return: Game _winner
        """
        return self._winner

    def get_move(self, traceback=0):
        """
        Getter. Returns a specific moe from the move history
        :returns Move at turn t (mark,move)
        """
        idx = -1 - traceback  # traceback of most recent state.
        if self.turn - idx < 0:
            idx = 0

        return self._moves[idx]

    def get_state(self, mark=None, traceback=0, string=True):
        """
        Gets state of the _board from the state history as a string or as a list
        :param mark: Player mark. If given, state replaces mark's mark with "1", and the adversary's mark with "2".
        Standardizes state representations for game-play.
        :param traceback: How many states ago the retrieved state is. traceback=0 is the most recent state.
        :param string: State representation. If True, returns string, else returns list.
        :returns State represented as string or list
        """
        idx = -1 - traceback  # traceback of most recent state.
        if self.turn + 1 - idx < 0:
            idx = 0

        board_state = self._history[idx]

        if mark is not None:
            state = []
            for p in board_state:
                if p == '0':
                    state.append("0")
                elif p == mark:
                    state.append("1")
                else:
                    state.append("2")
        else:
            state = board_state.copy()

        if string:
            return "".join(state)
        else:
            return state

    def _print_vdiv(self):
        """
        Pretty-prints vertical divider for the _board.
        """
        for _ in range(self.columns):
            print('+---', end='')
        print('+')

    def copy_end_state(self):
        """
        Creates a copy of the end state at the end of the history state.
        """
        self._history.append(self._history[-1])
        self._moves.append(self._moves[-1])
        self.turn += 1

    def print_state(self):
        """
        Pretty prints the current state of the _board, the turn number,
        most recent move and a _winner, if one is present.
        """
        i = 0
        print(f"TURN {self.turn}. COL: {self._moves[-1]}")

        self._print_vdiv()

        while i < len(self._board):
            mark = self._board[i]
            if mark == "0":
                print(f"|   ", end='')
            else:
                print(f"| {mark} ", end='')

            if i % self.columns == self.columns - 1:
                print("|")
                self._print_vdiv()

            i += 1

        if self._winner is not None:
            print(f"WINNER: {self._winner}")