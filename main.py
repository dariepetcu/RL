from Agent import *
from Problem import Problem, Dist


def run_agent(dist_type, mode, k):
    env = Problem(k, dist_type=dist_type, verbose=False)
    agent = Agent(env, mode=mode)

    agent.run(verbose=True, max_steps=10)


def main():
    dist_type = Dist.GAUSS
    mode = Mode.UCB
    k = 100
    run_agent(dist_type, mode, k)


if __name__ == "__main__":
    main()
