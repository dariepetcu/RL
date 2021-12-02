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


# Arm selection based on Softmax probability
def categorical_draw(pi):
    z = random.random()
    cum_prob = 0.0

    for i in range(len(pi)):
        prob = pi[i]
        cum_prob += prob

        if cum_prob > z:
            return i
    return len(pi) - 1


class Agent:
    def __init__(self, env, mode=Mode.GREEDY, epsilon=0.5, ucb_c = 0.5):
        self.env = env
        self.mode = mode
        self.step = 1  # time step

        self.estimations = [0] * env.arms
        self.uncertainties = [1] * env.arms
        self.H = [1 / env.arms] * env.arms
        self.pi = [1 / env.arms] * env.arms
        self.optimistic_values = env.get_max_values()

        # hyperparameters:
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.alpha = alpha


    def run(self,verbose=False,max_steps = 100):
        """
        :param verbose: Prints verbose code
        :param max_steps: Max number of epochs
        Runs the agent on the problem.
        """
        for self.step in range(max_steps):
            arm, reward = self.choose_action()
            # update parameters
            self.update_parameters(arm, reward)
            if verbose:
                print("Step:", self.step, "; Pulling arm", arm, "; Reward:", round(reward, 3))

    def choose_action(self):
        """
        Calls appropriate action selection method based on mode.
        :returns the selected arm and the resultant reward
        """
        match self.mode:
            case Mode.GREEDY:
                selected_arm = self.greedy()
            case Mode.EPSILON_GREEDY:
                selected_arm = self.epsilon_greedy()
            case Mode.OPTIMISTIC:
                # greedy action selection used for opt init vals
                selected_arm = self.greedy()
            case Mode.SOFTMAX:
                selected_arm = self.softmax()
            case Mode.UCB:
                selected_arm = self.ucb()
            case Mode.ACTION_PREFERENCES:
                selected_arm = self.action_pref()
            case _:
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

    def ucb(self):
        best_reward = 0
        best_action = None
        for arm in range(self.env.arms):
            reward = self.estimations[arm] + self.ucb_c * math.sqrt((np.log(self.step) / self.uncertainties[arm]))
            if best_reward > reward:
                best_action = arm
                best_reward = reward
        return best_action

    def softmax(self):
        # Pick arm based on cumulative probability
        best_action = categorical_draw(self.pi)
        return best_action

    def action_pref(self):
        # select action
        best_action = categorical_draw(self.H)
        return best_action

    # update the parameters that are relevant for the chosen algorithm
    def update_parameters(self, arm, reward):
        self.update_estimations(arm, reward)
        match self.mode:
            case Mode.UCB:
                self.update_uncertainties(arm)
            case Mode.SOFTMAX:
                self.update_pi(arm)
            case Mode.ACTION_PREFERENCES:
                self.update_preferences(arm, reward)
                self.update_pi(arm)

    # Q(a) = 0 unless optimistic initial values
    def initialize_estimations(self):
        if self.mode == Mode.OPTIMISTIC:
            # initialize values as more than max possible reward
            self.estimations = self.optimistic_values
            pass
        else:
            # initialize values at a specific value
            self.estimations = [0] * self.env.arms
            pass

    # incremental sample-average update
    def update_estimations(self, arm, reward):
        if self.step > 0:
            self.estimations[arm] += (reward - self.estimations[arm]) / self.step

    def update_uncertainties(self, selected_arm):
        self.uncertainties[selected_arm] += 1

    def update_preferences(self, selected_arm, reward):
        for arm in range(self.env.arms):
            regret = (self.optimistic_values[arm] - reward)
            if arm == selected_arm:
                self.H[arm] += self.alpha * regret * (1 - self.pi[arm])
            else:
                self.H[arm] -= self.alpha * regret * self.pi[arm]

    def update_pi(self, selected_arm):
        total = 0
        for arm in range(self.env.arms):
            total += math.exp((self.estimations[arm] / self.tau) if self.mode == Mode.SOFTMAX else (self.H[arm]))

        self.pi[selected_arm] = math.exp((self.estimations[selected_arm] / self.tau) if self.mode == Mode.SOFTMAX else
                                         (self.H[selected_arm])) / total
