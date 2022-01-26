import sys
import numpy as np
from kaggle_environments import evaluate, make, utils, agent

from agent import Agent


def main():
    """env = make("connectx", debug=True)
    # help(env)
    env.render(mode="ipython", width=500, height=450)
    env.run(["negamax", "random"])
    env.render(mode="ipython", width=500, height=450)
    env.play()"""
    # env.reset()

    env = make("connectx", {"rows": 6, "columns": 7, "inarow": 4}, debug=True)
    agent = Agent(env, 2)
    # Training agent in first position (mark 1) against the default random agent.
    trainer = env.train([None, agent.lmaolol])

    obs = trainer.reset()
    for _ in range(100):
        env.render()
        action = [c for c in range(len(obs.board)) if obs.board[c] == 0][0]  # Action for the agent being trained.
        obs, reward, done, info = trainer.step(action)
        print(env.)
        print(reward)
        if done:
            obs = trainer.reset()

    env.render(mode="ansi")
    # env.run([my_agent, my_agent])
    # env.render(mode="ipython", width=500, height=450)
    # # Play as first position against random agent.
    # trainer = env.train([None, "random"])
    # # "None" represents which agent you'll manually play as (first or second mark).
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

if __name__ == "__main__":
    main()
