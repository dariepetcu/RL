import sys
from enum import Enum
from numpy import random


class Dist(Enum):
    GAUSS = 0
    BERNOULLI = 1


class Problem:
    def __init__(self, k, dist_type=Dist.GAUSS, verbose=False):
        """
        Initializes the multi-armed bandit problem object
        :param k: Number of arms
        :param dist_type: Reward distribution (Gaussian dist_type. by default)
        """
        self.arms = k  # number of arms
        self.dist_type = dist_type  # reward distribution type

        if dist_type == Dist.GAUSS:
            self.reward_dists = []  # reward distributions for each arm
            self.generate_reward_dists(verbose)
        elif dist_type == Dist.BERNOULLI:
            self.reward_dists = []  # reward probability for each arm
            self.generate_probabilities(verbose)
        else:
            # invalid distribution provided, exit
            sys.exit("Invalid distribution (dist_type = " + str(dist_type) + ") provided!")

    def generate_probabilities(self, verbose):
        """
        Generates reward probabilities for each arm. Used for Bernoulli distribution.
        """
        for a in range(self.arms):
            prob = random.uniform(0, 1)
            if verbose:
                print("ARM {}:\t{}".format(a, round(prob,3)))
            self.reward_dists.append(prob)

    def generate_reward_dists(self, verbose):
        """
        Generates reward distributions for each arm. Used for Gaussian distribution.
        """
        for a in range(self.arms):
            stdev = random.uniform(0, .4)
            mean = random.uniform(stdev, 1 - stdev)
            dist = random.normal(loc=mean, scale=stdev)
            if verbose:
                print("ARM {}:\t MEAN: {},\t STD: {}".format(a, round(mean, 3), round(stdev, 3)))
            self.reward_dists.append(dist)

    def pull_arm(self, a):
        """
        Performs a specific action
        :param a: Action to perform
        :returns Reward from the action
        """
        if None:
            return 0
        if self.dist_type == Dist.GAUSS:
            # Gaussian distribution
            reward = random.choice(self.reward_dists)
            if reward < 0:
                reward = 0
        else:
            # Bernoulli distribution
            reward = int(random.uniform(0, 1) < self.reward_dists[a])
        return reward
