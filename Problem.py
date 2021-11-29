import sys
from enum import Enum
from numpy import random


class Dist(Enum):
    GAUSS = 0,
    BERNOULLI = 1


class Problem:

    def __init__(self, k, dist=Dist.BERNOULLI):
        """
        Initializes the multi-armed bandit problem object
        :param k: Number of arms
        :param dist: Reward distribution (Bernoulli dist. by default)
        """
        if dist == Dist.GAUSS:
            self.stdevs = [] # standard deviation for each arm
            self.generate_stdevs()
        elif dist == Dist.BERNOULLI:
            self.stdevs = None # stdevs not needed for Bernoulli dist
        else:
            # invalid distribution provided, exit
            sys.exit("Invalid distribution (dist = " + str(dist) + ") provided!")

        self.actions = k  # number of actions
        self.probabilities = []  # probability for each arm i (k arms in total), mean reward for Gaussian
        self.time = 0  # current time step
        self.dist = dist  # reward distribution

        self.generate_probabilities()

    def generate_probabilities(self):
        """
        Generates probabilities for each arm
        """
        for _ in range(self.actions):
            self.probabilities.append(random.uniform(0, 1))

    def generate_stdevs(self):
        """
        Generates standard deviations for each arm
        """
        for _ in range(self.actions):
            self.stdevs.append(random.uniform(0, 1))

    def pull_arm(self, a):
        """
        Performs a specific action and increments time
        :param a: Action to perform
        :returns Reward from the action
        """
        if self.dist == Dist.GAUSS:
            # Gaussian distribution
            reward = random.normal(loc=self.probabilities[a], scale=self.stdevs[a])
            if reward < 0:
                reward = 0
        else:
            # Bernoulli distribution
            reward = int(random.uniform(0, 1) < self.probabilities[a])
        self.time += 1 # increment time
        return reward
