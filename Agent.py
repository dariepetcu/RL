import math
import random
import sys
import numpy as np
from enum import Enum


class Mode(Enum):
    GREEDY = 0
    EPSILON_GREEDY = 1
    OPTIMISTIC = 2
    SOFTMAX = 3
    UCB = 4
    ACTION_PREFERENCES = 5


class Agent:
    def __init__(self, env, mode=Mode.GREEDY, epsilon=0.5, ucb_c = 0.5):
        self.env = env
        self.mode = mode
        self.estimations = []
        self.uncertainties = [0] * env.arms
        self.initialize_estimations()
        self.step = 0 # time step
        # hyperparameters:
        self.epsilon = epsilon
        self.ucb_c = ucb_c


    def run(self,verbose=False,max_steps = 100):
        """
        :param verbose: Prints verbose code
        :param max_steps: Max number of epochs
        Runs the agent on the problem.
        """
        for self.step in range(max_steps):
            arm, reward = self.choose_action()
            # update parameters
            self.update_estimations(arm, reward, self.step)
            if verbose:
                print("Step:", self.step, "; Pulling arm", arm, "; Reward:", round(reward, 3))

    # Q(a) = 0 unless optimistic initial values
    def initialize_estimations(self):
        if self.mode == Mode.OPTIMISTIC:
            # initialize values as more than max possible reward
            self.estimations = [self.env.optimistic_reward] * self.env.arms
            pass
        else:
            # initialize values at a specific value
            self.estimations = [0] * self.env.arms
            pass

    # incremental sample-average update
    def update_estimations(self, arm, reward, step):
        if step > 0:
            self.estimations[arm] += (reward - self.estimations[arm]) / step
        self.uncertainties[arm] += 1

    def choose_action(self):
        """
        Calls appropriate action selection method based on mode.
        :returns the selected arm and the resultant reward
        """
        if Mode.GREEDY:
            selected_arm = self.greedy()
        elif Mode.EPSILON_GREEDY:
            selected_arm = self.epsilon_greedy()
        elif Mode.OPTIMISTIC:
            # greedy action selection used for opt init vals
            selected_arm = self.greedy()
        elif Mode.SOFTMAX:
            selected_arm = self.softmax()
        elif Mode.UCB:
            selected_arm = self.ucb()
        elif Mode.ACTION_PREFERENCES:
            selected_arm = self.action_pref()
        else:
            sys.exit("Invalid selection mode selected!")

        return selected_arm, self.env.pull_arm(selected_arm)


    def greedy(self):
        best_action = np.argmax(self.estimations)
        # implement eps greedy here
        return best_action

    def epsilon_greedy(self):
        eps_action = random.uniform(0, 1)
        if eps_action < self.epsilon:
            return self.greedy()
        else:
            return random.randint(0, self.env.arms - 1)

    def softmax(self):
        best_action = random.Random()
        # implement eps greedy here
        return best_action

    def ucb(self):
        best_action = random.Random()

        # implement eps greedy here
        return best_action

    def action_pref(self):
        best_action = random.Random()
        # implement eps greedy here
        return best_action