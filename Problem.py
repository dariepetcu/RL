import sys
from enum import Enum
from numpy import random


class Dist(Enum):
    GAUSS = 0,
    BERNOULLI = 1


class Problem:

    def __init__(self, k, dist):
        if dist != Dist.GAUSS and dist != Dist.BERNOULLI:
            sys.exit("Invalid distribution (dist = " + dist + ") provided!")

        self.actions = k  # number of actions
        self.probabilities = []  # probability for each arm i (k arms in total)
        self.time = 0  # current time step
        self.dist = dist  # reward distribution

        self.generate_probabilities()

    def generate_probabilities(self):
        """
        Generates probabilities for each arm
        """
        for _ in range(self.actions):
            self.probabilities.append(random.uniform(0, 1))

    def pull_arm(self, a):
        """
        Performs a specific action and increments time
        :param a: Action to perform
        :returns Reward from the action
        """
        if self.dist == Dist.GAUSS:
            # Gaussian distribution
            reward = random.normal(loc=self.probabilities[a])
        else:
            # Bernoulli distribution
            reward = random.uniform(0, 1) < self.probabilities[a]
        self.time += 1
        return reward