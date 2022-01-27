from enum import Enum

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

    def __init__(self, game, name, mode = Mode.Q, alpha = 0.5, gamma = 0.9, depth = 42):
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
        self.alpha = alpha
        self.gamma = gamma
        self.gammaMC = 0.99
        # length of eligibility trace
        self.depth = depth
        self.decay = 0.8
        self.mode = mode
        self.G = []

    def select_action(self):
        match self.exploration:
            case Expl.GREEDY:
                selected_arm = self.greedy()
            case Expl.EPSILON_GREEDY:
                selected_arm = self.epsilon_greedy()
            case Expl.OPTIMISTIC:
                # greedy action selection used for opt init vals
                selected_arm = self.greedy()
            case Expl.SOFTMAX:
                selected_arm = self.softmax()
            case Expl.UCB:
                selected_arm = self.ucb()
            case Expl.ACTION_PREFERENCES:
                selected_arm = self.action_pref()
            case _:
                sys.exit("Invalid selection mode selected!")
        reward, is_best = self.env.pull_arm(selected_arm)
        return selected_arm, reward, is_best

    def update_estimates(self, reward):
        match self.mode:
            case Mode.Q:
                self.QLearn(reward)
            case Mode.SARSA:
                self.sarsa(reward)
            case Mode.MC:
                self.montecarlo(reward)

    def reset(self):
        self.action_history.clear()
        self.state_history.clear()
        self.G = []

    def brick_layer(self, board):
        self.current_sel += 1
        self.current_sel %= 7
        return self.current_sel

    def Q(self, state, action):
        return self.Qpairs.get(state, [0] * self.game.columns)[action]

    def montecarlo(self):
        self.G = [0] * self.game.turn
        self.G[self.game.turn] = 0
        for i in range(self.game.turn, 0, -1):
            self.G[i] += pow(self.gammaMC, i) * self.get_reward()

    def eligibility_trace(self, reward):
        for idx in range(self.depth):
            if self.mode == Mode.Q:
                pass
            elif self.mode == Mode.SARSA:
                pass

    def QLearn(self, reward):

    def update_SARSA(self, time):
        next_state = self.game.get_state(mark = self.name)
z        stateNew = state
        actionNew = 0
        for action in self.game.valid_moves():
            self.Q(state, action) = self.Q(state, action) + self.alpha *\
                                         (reward + self.gamma * self.Q(stateNew, actionNew) - self.Q(state, action))