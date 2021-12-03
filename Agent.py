import math
import random
import sys
import numpy as np
from enum import Enum

# enum to store algorithm type
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

    # return the randomly selected arm
    for i in range(len(pi)):
        prob = pi[i]
        cum_prob += prob

        if cum_prob > z:
            return i
    return len(pi) - 1


class Agent:
    def __init__(self, env, mode=Mode.GREEDY, epsilon=0.5, ucb_c=0.5, alpha=0.5, tau=0.5):
        """
        Agent that solves the multi armed bandit problem
        :param env: the multi-armed bandit
        :param mode: selected algorithm
        :param epsilon: for eps-greedy
        :param ucb_c: c hyperparameter for UCB
        :param alpha: alpha hyperparameter for action preference
        :param tau: hyperparameter for softmax
        """
        self.env = env
        self.mode = mode
        # time step, starts at 3 for UCB to avoid subunitary ln
        self.step = 3 if mode == Mode.UCB else 1

        # Qt(a) initialized to 0
        self.estimations = [0] * env.arms
        # Na(t) starts at 1 to avoid division by 0 in UCB
        self.uncertainties = [1] * env.arms
        # Ht(a) and PIt(a) initialized to equal complementary probabilities
        self.H = [1 / env.arms] * env.arms
        self.pi = [1 / env.arms] * env.arms
        # optimistic values retrieved from Problem.py distributions
        self.optimistic_values = env.get_max_values()

        # hyperparameters:
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.alpha = alpha
        self.tau = tau

        self.initialize_estimations()

    def run(self, verbose=False, max_steps=100):
        """
        :param verbose: Prints verbose code
        :param max_steps: Max number of epochs
        Runs the agent on the problem.
        """
        for self.step in range(self.step, max_steps):
            arm, reward = self.choose_action()
            # update parameters
            self.update_parameters(arm, reward)
            if verbose:
                print("Step:", self.step, "; Pulling arm", arm, "; Reward:", round(reward, 3))
        print("Process complete!")

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
        """
        Selects action for greedy algorithm
        :returns selected action
        """
        best_action = np.argmax(self.estimations)
        return best_action

    def epsilon_greedy(self):
        """
        Selects action for epsilon greedy algorithm
        :returns selected action
        """
        eps_action = random.uniform(0, 1)
        if eps_action < self.epsilon:
            return self.greedy()
        else:
            return random.randint(0, self.env.arms - 1)

    def ucb(self):
        """
        Selects action for UCB algorithm
        :returns selected action
        """
        best_reward = 0
        best_action = 0
        for arm in range(self.env.arms):
            reward = self.estimations[arm] + self.ucb_c * math.sqrt((np.log(self.step) / self.uncertainties[arm]))
            if best_reward > reward:
                best_action = arm
                best_reward = reward
        return best_action

    def softmax(self):
        """
        Selects action for softmax algorithm
        :returns selected action
        """
        best_action = categorical_draw(self.pi)
        return best_action

    def action_pref(self):
        """
        Selects action for action preference algorithm
        :returns selected action
        """
        best_action = categorical_draw(self.H)
        return best_action

    def update_parameters(self, arm, reward):
        """
        Calls appropriate methods for parameter updating, depending on selected algorithm
        :param arm: selected arm
        :param reward: obtained reward
        """
        self.update_estimations(arm, reward)
        match self.mode:
            case Mode.UCB:
                self.update_uncertainties(arm)
            case Mode.SOFTMAX:
                self.update_pi(arm)
            case Mode.ACTION_PREFERENCES:
                self.update_preferences(arm, reward)
                self.update_pi(arm)

    def initialize_estimations(self):
        """
        Initializes the estimations either optimistically or at a fixed value
        """
        if self.mode == Mode.OPTIMISTIC:
            # initialize values as more than max possible reward
            self.estimations = self.optimistic_values
        else:
            # initialize values at a specific value
            self.estimations = [0] * self.env.arms

    def update_estimations(self, arm, reward):
        """
        incremental sample-average update. updates the utility estimate of the selected arm based on the resultant
        reward
        :param arm: Selected arm
        :param reward: Resultant reward
        """
        if self.step > 0:
            self.estimations[arm] += (reward - self.estimations[arm]) / self.step

    def update_uncertainties(self, selected_arm):
        """
        Increments the uncertainties for the UCB algorithm
        :param selected_arm: Selected arm
        """
        self.uncertainties[selected_arm] += 1

    def update_preferences(self, selected_arm, reward):
        """
        Updates the action preferences of all arms for action preferences
        :param selected_arm: Selected arm
        """
        for arm in range(self.env.arms):
            regret = (self.optimistic_values[arm] - reward)
            if arm == selected_arm:
                self.H[arm] += self.alpha * regret * (1 - self.pi[arm])
            else:
                self.H[arm] -= self.alpha * regret * self.pi[arm]

    def update_pi(self, selected_arm):
        """
        Updates the pi value of all arms for softmax/action preferences
        :param selected_arm: Selected arm
        """
        total = 0
        for arm in range(self.env.arms):
            total += math.exp((self.estimations[arm] / self.tau) if self.mode == Mode.SOFTMAX else (self.H[arm]))

        self.pi[selected_arm] = math.exp((self.estimations[selected_arm] / self.tau) if self.mode == Mode.SOFTMAX else
                                         (self.H[selected_arm])) / total
