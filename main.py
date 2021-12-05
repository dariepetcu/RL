from Agent import *
from Problem import Problem, Dist


def run_agent(dist_type, mode, k):
    """
    Runs the agent with the given parameters
    :param dist_type: Distribution type
    :param mode: Selection mode
    :param k: Number of arms
    """
    env = Problem(k, dist_type=dist_type, verbose=False)
    agent = Agent(env, mode=mode)

    agent.run(verbose=True, plot=True, max_steps=1000)


def main():
    dist_type = Dist.GAUSS
    mode = Mode.EPSILON_GREEDY
    k = 100
    run_agent(dist_type, mode, k)


if __name__ == "__main__":
    main()
