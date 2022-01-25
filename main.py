import sys
import numpy as np
from kaggle_environments import evaluate, make, utils, agent


def main():
    env = make("connectx", debug=True)
    # help(env)
    env.render(mode="ipython", width=500, height=450)
    env.run(["negamax", "random"])
    env.render(mode="ipython", width=500, height=450)
    # env.reset()
    # env.run([my_agent, my_agent])
    # env.render(mode="ipython", width=500, height=450)
    # # Play as first position against random agent.
    # trainer = env.train([None, "random"])
    # # "None" represents which agent you'll manually play as (first or second player).
    # env.play([None, "negamax"], width=500, height=450)
    # observation = trainer.reset()
    #
    # while not env.done:
    #     my_action = my_agent(observation, env.configuration)
    #     print("My Action", my_action)
    #     observation, reward, done, info = trainer.step(my_action)
    #     env.render(mode="ipython", width=100, height=90, header=False, controls=False)
    #
    # env.render()
    # # Play as the first agent against default "random" agent.
    # env.run([my_agent, "random"])
    # env.render(mode="ipython", width=500, height=450)

# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

if __name__ == "__main__":
    main()
