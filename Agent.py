import math
import random
import sys
import numpy as np
from enum import Enum


class Mode(Enum):
    """
    Enum to represent selection mode/algorithm
    """
    GREEDY = 0
    EPSILON_GREEDY = 1
    OPTIMISTIC = 2
    SOFTMAX = 3
    UCB = 4
    ACTION_PREFERENCES = 5


def categorical_draw(pi):
    """
    Arm selection based on pi probability. Algorithm taken from
    https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-softmax-algorithm-e1fa4cb0c422
    :param pi:
    :return:
    """
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
    def __init__(self, env, mode=Mode.GREEDY, epsilon=0.38, ucb_c=0.38, alpha=0.9, tau=0.12):
        """
        Agent that solves the multi armed bandit problem
        :param env: the multi-armed bandit
        :param mode: selected algorithm
        :param epsilon: epsilon hyperparameter for eps-greedy
        :param ucb_c: c hyperparameter for UCB
        :param alpha: alpha hyperparameter for action preference
        :param tau: hyperparameter for softmax
        """

        self.env = env
        self.mode = mode

        # time step, starts at 3 for UCB to avoid subunitary ln
        self.step = 3 if mode == Mode.UCB else 1
        self.average_rewards = []
        self.accuracy = []

        # Qt(a) given values in a later function
        self.estimations = None

        # Na(t) starts at 1 to avoid division by 0 in UCB
        self.uncertainties = [1] * env.arms

        # Ht(a) and PIt(a) initialized to equal complementary probabilities
        self.H = [1 / env.arms] * env.arms
        self.pi = [1 / env.arms] * env.arms

        # hyperparameters:
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.alpha = alpha
        self.tau = tau

        self.initialize_estimations()

    def run(self, verbose=False, max_steps=1000):
        """
        :param verbose: Prints verbose code
        :param max_steps: Max number of epochs
        Runs the agent on the problem.
        """
        total_reward = 0
        counter_selected_best = 0
        max_steps += self.step  # adjust for initial step value
        for self.step in range(self.step, max_steps):
            arm, reward, is_best = self.choose_action()
            total_reward += reward
            counter_selected_best += is_best
            self.average_rewards.append(round(total_reward / self.step, 3))
            self.accuracy.append(round(counter_selected_best / self.step, 3))
            # update parameters
            self.update_parameters(arm, reward)
            if verbose:
                print("Step:", self.step, "; Pulling arm", arm, "; Reward:", round(reward, 3),
                      "; current average reward:", round(total_reward / self.step, 3))
        # print("Process complete!")

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
        reward, is_best = self.env.pull_arm(selected_arm)
        return selected_arm, reward, is_best

    def greedy(self):
        """
        Selects action for greedy algorithm. If multiple arms have the same highest estimated utility, one is randomly
        chosen.
        :returns selected action
        """
        # best_actions = [random.randint(0,self.env.arms - 1)]
        # best_reward = self.estimations[best_actions[0]]
        best_actions = [0]
        best_reward = self.estimations[best_actions[0]]

        for arm in range(self.env.arms):
            reward = self.estimations[arm]
            if reward > best_reward:  # new highest utility arm
                best_actions = [arm]
                best_reward = reward
            elif reward == best_reward:  # arm with same highest utility
                best_actions.append(arm)

        return random.choice(best_actions)

    def epsilon_greedy(self):
        """
        Selects action for epsilon greedy algorithm
        :returns selected action
        """
        eps_action = random.uniform(0, 1)
        if eps_action > self.epsilon:
            return self.greedy()
        else:
            return random.randint(0, self.env.arms - 1)

    def ucb(self):
        """
        Selects action for UCB algorithm
        :returns selected action
        """

        # initialize starting values
        rand_arm = random.randint(0, self.env.arms - 1)
        best_reward = self.estimations[rand_arm] + self.ucb_c * math.sqrt(
            np.log(self.step) / self.uncertainties[rand_arm])
        best_actions = [rand_arm]

        # iterate to find actions with highest rewards
        for arm in range(self.env.arms):
            reward = self.estimations[arm] + self.ucb_c * math.sqrt(np.log(self.step) / self.uncertainties[arm])
            if round(reward, 1) > round(best_reward, 1):
                best_actions = [arm]
                best_reward = reward
            elif round(reward, 1) == round(best_reward, 1):
                best_actions.append(arm)

        # randomly return one of the highest-reward actions
        return random.choice(best_actions)

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
        best_action = categorical_draw(self.pi)
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
        if self.mode == Mode.OPTIMISTIC or self.mode == Mode.UCB:  # initialize values as more than max possible reward
            self.estimations = [1] * self.env.arms
        else:  # initialize values at a specific value
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
        :param reward: Reward from selected action
        :param selected_arm: Selected arm
        """
        for arm in range(self.env.arms):
            regret = (reward - self.average_rewards[-1])
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
            if self.mode == Mode.SOFTMAX:  # if softmax algorithm
                total += math.exp(self.estimations[arm] / self.tau)
            else:  # if action preferences
                total += math.exp(self.H[arm])

        if self.mode == Mode.SOFTMAX:  # if softmax algorithm
            self.pi[selected_arm] = math.exp(self.estimations[selected_arm] / self.tau) / total
        else:  # if action preferences algorithm
            self.pi[selected_arm] = math.exp(self.H[selected_arm]) / total
