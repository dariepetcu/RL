import random
import Problem

class Agent:

    def __init__(self, envmt):
        self.envmt = envmt
        self.selection_mode = "greedy"
        self.action_history = []
        self.rewards_history = []
        self.epsilon = 0.5

    # call appropriate action selection method based on selection_mode
    def choose_action(self):
        pass

    def estimate_action(self, time, action):
        rewards, selection = 0
        for step in time:
            match_action = self.action_history[step] == action
            rewards += self.rewards_history[step] * match_action
            selection += match_action

        if selection == 0:
            return 0
        else:
            return rewards / selection

    def greedy(self):
        best_action = max([(self.estimate_action(self.envmt.time, a), a) for a in range(self.envmt.actions)])
        # implement eps greedy here
        return best_action

    def epsilon_greedy(self):
        eps_action = random.uniform(0, 1)
        if eps_action < self.epsilon:
            return self.greedy()
        else:
            # TODO: k or k-1? cross check against Problem.py implementation
            return random.randint(0, self.envmt.k - 1)

    def optim_init_vals(self):
        best_action = random.Random()
        # implement eps greedy here
        return best_action

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

