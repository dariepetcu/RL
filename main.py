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

    agent.run(verbose=True, max_steps=100)


def main():
    dist_type = Dist.BERNOULLI
    mode = Mode.ACTION_PREFERENCES
    k = 100
    run_agent(dist_type, mode, k)


if __name__ == "__main__":
    main()
