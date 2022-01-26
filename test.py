import random

from env import ConnectX

game = ConnectX()

def move(mark):
    valid_moves = game.valid_moves()
    move = random.choice(valid_moves)
    game.add_piece(move, mark)
    game.print_state()

while game.winner is None:
    move("A")
    move("B")