import sys
from enum import Enum

import numpy as np
from numpy import random

from numpy import argmax


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

        if self.verbose:
            self.print_arms()

    def generate_reward_dists(self):
        """
        Generates reward distributions for each arm. Used for Gaussian distribution.
        """
        for a in range(self.arms):
            stdev = 0.2  # random.uniform(0, .3)
            mean = random.uniform(.3, 1)
            self.reward_dists.append((mean, stdev))

    def generate_probabilities(self):
        """
        Generates reward probabilities for each arm. Used for Bernoulli distribution.
        """
        for a in range(self.arms):
            prob = random.uniform(0, 1)
            self.reward_dists.append(prob)

    def best_action(self):

        if self.dist_type == Dist.GAUSS:
            best_actions = 0
            best_reward = self.reward_dists[0][0]

            for arm in range(1, self.arms):
                reward = self.reward_dists[arm][0]
                if reward > best_reward:  # new highest utility arm
                    best_actions = [arm]
                    best_reward = reward
                elif reward == best_reward:  # arm with same highest utility
                    best_actions.append(arm)
        else:  # bernoulli dist
            best_option = np.argmax(self.reward_dists)

        return best_option

    def print_arms(self):
        if self.dist_type == (Dist.GAUSS):
            for i in range(self.arms):
                a = self.reward_dists[i]
                mean, stdev = a
                print("ARM {}:\t MEAN: {},\t STD: {}".format(i, round(mean, 3), round(stdev, 3)))
        else:
            for i in range(self.arms):
                a = self.reward_dists[i]
                try:  # convert gauss dist into bernoulli dist
                    prob = a[0]
                except:
                    prob = a
                print("ARM {}:\t{}".format(i, round(prob, 3)))

    def pull_arm(self, a):
        """
        Performs a specific action
        :param a: Action to perform
        :returns Reward for the action and whether the selected action is the best action
        """
        if None:
            return 0
        if self.dist_type == Dist.GAUSS:
            # Gaussian distribution
            mean, stdev = self.reward_dists[a][0], self.reward_dists[a][1]
            reward = random.normal(loc=mean, scale=stdev)
            # limit reward to the 0 or 1
            if reward < 0:
                reward = 0
            elif reward > 1:
                reward = 1
        else:
            # Bernoulli distribution
            try:  # convert gauss dist into bernoulli dist
                prob = a[0]
            except:
                prob = a
            reward = int(random.uniform(0, 1) < prob)

        is_best = 1 if a == self.best_action() else 0
        return reward, is_best
