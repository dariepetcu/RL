import numpy as np

class Board:
    def __init__(self, height = 6, width = 7, goal = 4):
        """
        Initializes the connect-X game object
        """
        self.height = height
        self.width = width
        self.goal = goal
        self.game = np.empty([self.height, self.width])

    def valid_move(self, column):
        if self.game[self.height][column] is None:
            return True
        else:
            return False
