import sys
from enum import Enum
import random


class Mode(Enum):
    Q = 1
    SARSA = 2
    MC = 3


class Expl(Enum):
    GREEDY = 0
    EPSILON_GREEDY = 1
    OPTIMISTIC = 2
    SOFTMAX = 3
    UCB = 4
    ACTION_PREFERENCES = 5


class Agent:
    """
    Agent class. Plays Connect 4 and learns from it.
    """

    def __init__(self, game, name, mode=Mode.Q, exploration=Expl.EPSILON_GREEDY, alpha=0.5, gamma=0.9,
                 epsilon=0.38, depth=42):
        """
        Initializes an agent that can play Connect 4 with reinforcement learning enabled.
        :param game: Board state represented as a matrix of size rows x columns
        :param mark: Player number of agent, either "1" or "2"
        :param rows: Number of rows
        :param columns: Number of columns
        :param goal: Number of pieces in a row required to win
        """

        self.game = game
        self.current_sel = 0
        self.name = name
        self.Qpairs = {}
        self.state_history = []
        self.action_history = []

        self.mode = mode
        self.exploration = exploration
        self.G = []

        self.alpha = alpha
        self.gamma = gamma
        self.gammaMC = 0.99
        self.epsilon = epsilon
        # length of eligibility trace
        self.depth = depth
        self.decay = 0.8

    def reset(self):
        self.action_history.clear()
        self.state_history.clear()
        self.G = []

    def brick_layer(self):
        self.current_sel += 1
        self.current_sel %= 7
        return self.current_sel

    def select_action(self):
        match self.exploration:
            case Expl.GREEDY:
                action = self.greedy()
            case Expl.EPSILON_GREEDY:
                action = self.epsilon_greedy()
            case _:
                sys.exit(f"{self.exploration}: Invalid action selection method!")
            # case Expl.OPTIMISTIC:
            #     # greedy action selection used for opt init vals
            #     action = self.greedy()
            # case Expl.SOFTMAX:
            #     action = self.softmax()
            # case Expl.UCB:
            #     action = self.ucb()
            # case Expl.ACTION_PREFERENCES:
            #     action = self.action_pref()
            # case _:
            #     sys.exit("Invalid selection mode selected!")
        return action

    def update_estimates(self, reward):
        match self.mode:
            case Mode.Q:
                self.QLearn(reward)
            case Mode.SARSA:
                self.update_SARSA(reward)
            # case Mode.MC:
            #     self.montecarlo(reward)

    def Q(self, state, action):
        return self.Qpairs.get(state, [0] * self.game.columns)[action]

    # def montecarlo(self, reward):
    #     self.G = [0] * self.game.turn
    #     self.G[self.game.turn] = 0
    #     for i in range(self.game.turn, 0, -1):
    #         self.G[i] += pow(self.gammaMC, i) * reward

    # def eligibility_trace(self, reward):
    #     for idx in range(self.depth):
    #         if self.mode == Mode.Q:
    #             pass
    #         elif self.mode == Mode.SARSA:
    #             pass

    def QLearn(self, reward):
        state, action, next_state, next_action = self.get_sa_pairs()
        updated_Q = self.Q(state, action)
        updated_Q += self.alpha * (reward + self.gamma * max(self.Qpairs.get(next_state, [0] * self.game.columns)) -
                                   updated_Q)
        self.Qpairs.get(state, [0] * self.game.columns)[action] = updated_Q

    def update_SARSA(self, reward):
        state, action, next_state, next_action = self.get_sa_pairs()
        for action in self.game.valid_moves():
            updated_Q = self.Q(state, action)
            updated_Q += self.alpha * (reward + self.gamma * self.Q(next_state, next_action) - self.Q(state, action))

    def greedy(self):
        state = self.game.get_state(mark=self.name)
        best_actions = [0]
        best_estimate = self.Qpairs.get(state, [0] * self.game.columns)[0]

        for action in self.game.valid_moves():
            estimate = self.Q(state, action)
            if estimate > best_estimate:  # new best estimated action
                best_actions = [action]
                best_estimate = estimate
            elif estimate == best_estimate:  # action with same highest utility
                best_actions.append(action)

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
            return random.randint(0, self.game.columns - 1)

    def get_sa_pairs(self):
        """
        returns S,A,S,A sequence
        :return: s_t, a_t, s_{t+1}, a_{t+1}
        """
        return self.game.get_state(2, mark=self.name), self.game.get_move(2)[1], \
               self.game.get_state(mark=self.name), self.game.get_move()[1]
