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

    def __init__(self, game, name, learning=Learning.Q, selection=Selection.EPSILON_GREEDY, alpha=0.99, gamma=0.99,
                 epsilon=0.38):
        """
        Initializes an agent that can play Connect 4 with reinforcement learning enabled.
        :param game: Board object, containing state represented as a matrix of size rows x columns
        :param name: Agent name
        :param learning: Learning mode
        :param selection: Selection algorithm
        :param alpha: Learning rate hyperparameter
        :param gamma: Discount hyperparameter
        :param epsilon: Epsilon hyperparameter for epsilon-greedy
        """

        self.game = game  # game environment
        self.current_sel = 0  # parameter used by bricklayer agent.
        self.name = name  # agent name.
        self.Qpairs = {}  # state-action dictionary
        self.turns = 0  # number of pieces the agent played successfully

        self.learning = learning  # learning mode
        self.selection = selection  # learning mode
        self.G = []

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon for epsilon-greedy

    def reset(self):
        """
        Method used to reset information that is not relevant outside
        the scope of the current episode
        """
        self.G = []

    def brick_layer(self):
        """
        Method implementing the brick layering agent, which chooses the column to the right
        of the one chosen in the previous round
        :return: the column it chooses for piece placing
        """
        self.current_sel += 1
        # prevent selection from going outside board or being invalid
        while self.current_sel not in self.game.valid_moves():
            self.current_sel += 1
            self.current_sel %= self.game.columns

        return self.current_sel % self.game.columns

    def random_agent(self):
        """
        Method implementing the random agent, which chooses a random valid move
        :return: the chosen move
        """
        return random.choice(self.game.valid_moves())

    def select_action(self):
        """
        Action selection method called by the trainer to ask agent what its intention is
        :return: the chosen action
        """
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
                sys.exit(f"{self.selection}: Invalid action selection method!")
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
            #     sys.exit("Invalid learning selected!")
        return action

    def update_estimates(self, reward):
        """
        Method called by the trainer to tell agent it can update its estimation
        based on the newly received information (reward and new state)
        :param reward: reward for reaching s_{t+1}, is zero unless s_{t+1} is final state
        """
        if self.selection in [Selection.BRICK, Selection.RANDOM]:
            return
        match self.learning:
            case Learning.Q:
                self.QLearn(reward)
            case Learning.SARSA:
                self.update_SARSA(reward)
            case Learning.MC:
                if self.game.get_winner() is not None:
                    self.montecarlo(reward)
            case _:
                return

    def Q(self, state, action):
        """
        Method used to fetch value estimates from the dict object that stores them
        :params state, action: pair whose Q-estimate must be fetched
        :return: fetched Q-value for passed state, action.
        """
        return self.Qpairs.get(state, [random.random()] * self.game.columns)[action]

    def renew_Q(self, state, action, updated_Q):
        """
        Method to edit value estimates in the dictionary
        :params state, action: used to target the value estimate to be updated
        :param updated_Q: the updated value estimate
        """
        new_entry = self.Qpairs.get(state, [random.random()] * self.game.columns)
        new_entry[action] = updated_Q
        self.Qpairs[state] = new_entry

    def montecarlo(self, reward):
        """
        Only called after game is finished. Computes discounted reward for each state
        :param reward: used for computing G[state]
        """
        turns, history = self.game.get_player_history(self.name)
        self.G = [0] * turns
        # start from turns-1 because G[turns] = 0
        for i in range(turns - 1, 0, -1):
            self.G[i] = pow(self.gamma, turns - i - 1) * reward
        self.mc_update(turns, history)

    def mc_update(self, turns, history):
        """
        Once discounted reward G[t] has been calculated for current match, this method
        updates the value estimates based on it
        :param turns: the total number of moves from current match
        :param history: list of all state, action pairs from current match
        """
        for i in range(turns):
            state = history[i][0]
            action = history[i][1]
            updated_Q = self.Q(state, action)
            updated_Q += self.alpha * (self.G[i] - updated_Q)
            self.renew_Q(state, action, updated_Q)

    def QLearn(self, reward):
        """
        Method to update value estimate of current state, action when using QLearning
        :param reward: received for reaching s_{t+1}
        """
        state, action, next_state, next_action = self.get_sa_pairs()
        updated_Q = self.Q(state, action)
        updated_Q += self.alpha * self.TD_error(state, action, reward, next_state, next_action)
        self.renew_Q(state, action, updated_Q)

    def update_SARSA(self, reward):
        """
        Method to update value estimate of current state, action when using SARSA
        :param reward: received for reaching s_{t+1}
        """
        state, action, next_state, next_action = self.get_sa_pairs()
        updated_Q = self.Q(state, action)
        updated_Q += self.alpha * self.TD_error(state, action, reward, next_state, next_action)
        self.renew_Q(state, action, updated_Q)

    def TD_error(self, state, action, reward, next_state, next_action):
        """
        Method to compute the TD-Learning according to the selected TD algorithm (on/off policy)
        :param state: current state
        :param action: action to go from s to s_{t+1}
        :param reward: reward for getting to s_{t+1}
        :param next_state: predicted state s_{t+1}
        :param next_action: predicted action a_{t+1}
        :return: TD-error value based on value estimates and hyperparameters
        """
        if self.game.get_winner is not None:
            # Q value of final state is 0, so TD-target = reward
            return reward - self.Q(state, action)
        match self.learning:
            case Learning.Q:
                return (reward + self.gamma * max(self.Qpairs.get(next_state, [random.random()] * self.game.columns)) -
                                   self.Q(state, action))
            case Learning.SARSA:
                return reward + self.gamma * self.Q(next_state, next_action) - self.Q(state, action)

    def greedy(self):
        """
        Action selection method. Chooses valid action with best value estimation
        :return: chosen action
        """
        state = self.game.get_state(mark=self.name)
        best_actions = []
        valid_moves = self.game.valid_moves()
        # initialize best move as random valid one
        best_actions.append(random.choice(valid_moves))
        best_estimate = self.Qpairs.get(state, [random.random()] * self.game.columns)[best_actions[0]]

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
        :return: s_t, a_t, s_{t+1}, a_{t+1} sequence
        """
        turns, history = self.game.get_player_history(self.name, n=2)
        state, action = history[0]
        next_state, next_action = history[1]
        #print(f"ACTION: {action}\nOLD: {state}")
        #print(f"NEW: {next_state}")
        return state, action, next_state, next_action

    def set_hyperparameter(self, hyper, value):
        """
        Setter. Sets the value of a provided hypeparameter. Used for fine-tuning.
        :param hyper: Hyperparameter whose value needs to be set
        :param value: New value
        """

        match hyper:  # matches correct hyperparameter
            case "alpha":
                self.alpha = value
            case "epsilon":
                self.epsilon = value
            case "gamma":
                self.gamma = value
            case _:
                print(f"{hyper}: Invalid hyperparameter!")