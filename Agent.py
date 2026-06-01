import random
import numpy as np
from Action import Action
from Environement import Environement as Env


class Random_Agent:
    def __init__(self, env) -> None:
        self.env : Env = env
        self.Reward = 0

    def get_action(self, state):
        actions = self.env.get_actions(state)
        return random.choice(actions)

    def add_reward(self, reward):
        self.Reward += reward

    def __call__(self, state):
        return self.get_action(state)


class AI_Agent:
    def __init__(self, env) -> None:
        self.env : Env = env
        self.rows = env.rows
        self.cols = env.cols
        self.Reward = 0
        self.Policy = np.full((self.rows, self.cols), 3)
        self.Value = np.zeros((self.rows, self.cols))
        self.gamma = 0.9
        self.accuracy = 0.0001

    def get_action(self, state):
        return Action(self.Policy[state])

    def add_reward(self, reward):
        self.Reward += reward

    def q_value(self, state, action):
        q = 0.0
        for prob, ns, rw in self.env.transitions(state, action):
            v_next = 0.0 if self.env.end_of_game(ns) else self.Value[ns]
            q += prob * (rw + self.gamma * v_next)
        return q

    def best_action_value(self, state):
        actions = self.env.get_actions(state)
        return max(((a, self.q_value(state, a)) for a in actions), key=lambda x: x[1])

    def _sweep(self, update_fn, callback=None):
        """Gauss-Seidel sweep: update V in place. Converges ~2x faster than Jacobi."""
        iteration = 0
        while True:
            delta = 0.0
            for row in range(self.rows):
                for col in range(self.cols):
                    s = (row, col)
                    if self.env.end_of_game(s):
                        continue
                    old = self.Value[s]
                    self.Value[s] = update_fn(s)
                    delta = max(delta, abs(self.Value[s] - old))
            iteration += 1
            if callback:
                stop = callback(iteration, delta, self.Value, self.Policy)
                if stop:
                    return
            if delta < self.accuracy:
                return

    def policy_eval(self, callback=None):
        self._sweep(lambda s: self.q_value(s, Action(self.Policy[s])), callback)

    def Policy_improv(self):
        stable = True
        for row in range(self.rows):
            for col in range(self.cols):
                s = (row, col)
                if self.env.end_of_game(s):
                    continue
                old = self.Policy[s]
                best_a, _ = self.best_action_value(s)
                self.Policy[s] = best_a.value
                if old != self.Policy[s]:
                    stable = False
        return stable

    def Policy_Iteration(self, callback=None):
        while True:
            self.policy_eval(callback=callback)
            if self.Policy_improv():
                break

    def Value_Iteration(self, callback=None):
        def update(s):
            best_a, best_q = self.best_action_value(s)
            self.Policy[s] = best_a.value
            return best_q
        self._sweep(update, callback)

    def __call__(self, state):
        return self.get_action(state)
