from agent import Agent
from env import ConnectX


def train_agent(epochs=1000):
    game = ConnectX()
    agent = Agent(game, "A")
    for _ in range(epochs):
        game.run(agent)
        game.reset()
        agent.reset()


def main():
    train_agent()



if __name__ == "__main__":
    main()
