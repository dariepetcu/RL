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
        :param verbose: If True, prints additional information about the problem
        """
        self.arms = k  # number of arms
        self.dist_type = dist_type  # reward distribution type
        self.verbose = verbose

        if dist_type == Dist.GAUSS:
            self.reward_dists = []  # reward distributions for each arm
            self.generate_reward_dists()
        elif dist_type == Dist.BERNOULLI:
            self.reward_dists = []  # reward probability for each arm
            self.generate_probabilities()
        else:
            # invalid distribution provided, exit
            sys.exit("Invalid distribution (dist_type = " + str(dist_type) + ") provided!")

    def get_max_values(self):
        """
        Returns the maximum value from an array of 1000 samples from the distribution of each arm.
        :returns Maximum generated value per arm (list of 1s if Bernoulli dist)
        """
        if self.dist_type == Dist.GAUSS:
            max_values = []
            for a in range(self.arms):
                mean, stdev = self.reward_dists[a][0], self.reward_dists[a][1]
                rewards = random.normal(loc=mean, scale=stdev, size=500)
                max_values.append(max(rewards))
        else:
            max_values = [1] * self.arms
        return max_values

    def generate_probabilities(self):
        """
        Generates reward probabilities for each arm. Used for Bernoulli distribution.
        """
        for a in range(self.arms):
            prob = random.uniform(0, 1)
            if self.verbose:
                print("ARM {}:\t{}".format(a, round(prob, 3)))
            self.reward_dists.append(prob)

    def generate_reward_dists(self):
        """
        Generates reward distributions for each arm. Used for Gaussian distribution.
        """
        for a in range(self.arms):
            stdev = random.uniform(0, .3)
            mean = random.uniform(0.5, 1 - stdev)
            if self.verbose:
                print("ARM {}:\t MEAN: {},\t STD: {}".format(a, round(mean, 3), round(stdev, 3)))
            self.reward_dists.append((mean, stdev))

    def pull_arm(self, a):
        """
        Performs a specific action
        :param a: Action to perform
        :returns Reward for the action
        """
        if None:
            return 0
        if self.dist_type == Dist.GAUSS:
            # Gaussian distribution
            mean, stdev = self.reward_dists[a][0], self.reward_dists[a][1]
            reward = random.normal(loc=mean, scale=stdev)
        else:
            # Bernoulli distribution
            reward = int(random.uniform(0, 1) < self.reward_dists[a])
        return reward
