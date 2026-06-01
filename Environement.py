import random
import numpy as np
from Action import Action


DELTA = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}

PERPENDICULAR = {
    Action.UP: (Action.LEFT, Action.RIGHT),
    Action.DOWN: (Action.LEFT, Action.RIGHT),
    Action.LEFT: (Action.UP, Action.DOWN),
    Action.RIGHT: (Action.UP, Action.DOWN),
}


class Environement:
    def __init__(self, rows=4, cols=4, state=(0, 0), step_cost=0.0, slip=0.0):
        self.rows = rows
        self.cols = cols
        self.state = state
        self.start = state
        self.board = np.zeros((rows, cols))
        self.step_cost = step_cost
        self.slip = slip

    def reset(self):
        self.state = self.start

    def set_reward(self, cell, value):
        self.board[cell] = value

    def step(self, state, action: Action):
        """Apply one action deterministically (no slip). Walls reflect."""
        dr, dc = DELTA[action]
        r, c = state
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        new_state = (nr, nc)
        reward = self.board[new_state]
        if reward == 0:
            reward = -self.step_cost
        return new_state, reward

    def move(self, state, action: Action):
        """Stochastic move used at runtime. Sample outcome by slip prob."""
        if self.slip <= 0:
            return self.step(state, action)
        left, right = PERPENDICULAR[action]
        p_intended = 1.0 - self.slip
        p_side = self.slip / 2.0
        r = random.random()
        if r < p_intended:
            chosen = action
        elif r < p_intended + p_side:
            chosen = left
        else:
            chosen = right
        return self.step(state, chosen)

    def transitions(self, state, action):
        """List of (prob, new_state, reward). Used by DP."""
        if self.slip <= 0:
            ns, rw = self.step(state, action)
            return [(1.0, ns, rw)]
        left, right = PERPENDICULAR[action]
        p_intended = 1.0 - self.slip
        p_side = self.slip / 2.0
        outs = []
        for prob, a in [(p_intended, action), (p_side, left), (p_side, right)]:
            ns, rw = self.step(state, a)
            outs.append((prob, ns, rw))
        return outs

    def get_actions(self, state):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    def end_of_game(self, state):
        return self.board[state] != 0

    def __call__(self, state, action):
        return self.move(state, action)
