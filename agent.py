import sys
from enum import Enum
import random


class Learning(Enum):
    Q = 1
    SARSA = 2
    MC = 3


class Selection(Enum):
    GREEDY = 0
    EPSILON_GREEDY = 1
    OPTIMISTIC = 2
    SOFTMAX = 3
    BRICK = 4
    RANDOM = 5


class Agent:
    """
    Agent class. Plays Connect 4 and learns from it.
    """

    def __init__(self, game, name, learning=Learning.Q, selection=Selection.EPSILON_GREEDY, alpha=0.5, gamma=0.9,
                 epsilon=0.38):
        """
        Initializes an agent that can play Connect 4 with reinforcement learning enabled.
        :param game: Board state represented as a matrix of size rows x columns
        :param name: Agent name.
        :param learning: Learning mode.
        :param selection: Selection setting.
        :param alpha: Learning rate.
        :param gamma: Discount parameter.
        :param epsilon: Epsilon for epsilon-greedy.
        """

        self.game = game  # game environment
        self.current_sel = 0  # bricklaying parameter.
        self.name = name  # agent name.
        self.Qpairs = {}  # state-action dictionary
        self.turns = 0  # number of pieces the agent played successfully

        self.learning = learning  # learning mode
        self.selection = selection  # learning mode
        self.G = []

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.gammaMC = 0.99  # discount parameter for monte carlo
        self.epsilon = epsilon  # epsilon for epsilon-greedy

        # length of eligibility trace
        self.depth = self.game.rows * self.game.columns  #
        self.decay = 0.8

    def reset(self):
        self.G = []

    def brick_layer(self):
        self.current_sel += 1
        while self.current_sel not in self.game.valid_moves():
            self.current_sel += 1
            self.current_sel %= self.game.columns

        return self.current_sel % self.game.columns

    def random_agent(self):
        return random.choice(self.game.valid_moves())

    def select_action(self):
        match self.selection:
            case Selection.GREEDY:
                action = self.greedy()
            case Selection.EPSILON_GREEDY:
                action = self.epsilon_greedy()
            case Selection.BRICK:
                action = self.brick_layer()
            case Selection.RANDOM:
                action = self.random_agent()
            case _:
                sys.exit(f"{self.selection}: Invalid action learning method!")
            # case Selection.OPTIMISTIC:
            #     # greedy action learning used for opt init vals
            #     action = self.greedy()
            # case Selection.SOFTMAX:
            #     action = self.softmax()
            # case Selection.UCB:
            #     action = self.ucb()
            # case Selection.ACTION_PREFERENCES:
            #     action = self.action_pref()
            # case _:
            #     sys.exit("Invalid learning learning selected!")
        return action

    def update_estimates(self, reward):
        if self.selection in [Selection.BRICK, Selection.RANDOM]:
            return
        match self.learning:
            case Learning.Q:
                self.QLearn(reward)
            case Learning.SARSA:
                self.update_SARSA(reward)
            case _:
                return
            # case Learning.MC:
            #     self.montecarlo(reward)

    def Q(self, state, action):
        return self.Qpairs.get(state, [0] * self.game.columns)[action]

    def renew_Q(self, state, action, updated_Q):
        new_entry = self.Qpairs.get(state, [0] * self.game.columns)
        new_entry[action] = updated_Q
        self.Qpairs[state] = new_entry

    # def montecarlo(self, reward):
    #     self.G = [0] * self.game.turn
    #     self.G[self.game.turn] = 0
    #     for i in range(self.game.turn, 0, -1):
    #         self.G[i] += pow(self.gammaMC, i) * reward

    # def eligibility_trace(self, reward):
    #     for idx in range(self.depth):
    #         if self.learning == Learning.Q:
    #             pass
    #         elif self.learning == Learning.SARSA:
    #             pass

    def QLearn(self, reward):
        state, action, next_state, next_action = self.get_sa_pairs()
        updated_Q = self.Q(state, action)
        updated_Q += self.alpha * (reward + self.gamma * max(self.Qpairs.get(next_state, [0] * self.game.columns)) -
                                   updated_Q)
        self.renew_Q(state, action, updated_Q)

    def update_SARSA(self, reward):
        state, action, next_state, next_action = self.get_sa_pairs()
        for action in self.game.valid_moves():
            updated_Q = self.Q(state, action)
            updated_Q += self.alpha * (reward + self.gamma * self.Q(next_state, next_action) - self.Q(state, action))
            self.renew_Q(state, action, updated_Q)

    def greedy(self):
        state = self.game.get_state(mark=self.name)
        best_actions = []
        best_estimate = self.Qpairs.get(state, [0] * self.game.columns)[0]
        valid_moves = self.game.valid_moves()

        for action in valid_moves:
            estimate = self.Q(state, action)
            if estimate > best_estimate:  # new best estimated action
                best_actions = [action]
                best_estimate = estimate
            elif estimate == best_estimate:  # action with same highest utility
                best_actions.append(action)

        for action in best_actions:
            if action not in valid_moves:
                best_actions.remove(action)

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
            return random.choice(self.game.valid_moves())

    def get_sa_pairs(self):
        """
        returns S,A,S,A sequence
        :return: s_t, a_t, s_{t+1}, a_{t+1}
        """
        state = self.game.get_state(2, mark=self.name)
        action = self.game.get_move(2)[1]
        next_state = self.game.get_state(mark=self.name)
        next_action = self.game.get_move()[1]
        return state, action, next_state, next_action
