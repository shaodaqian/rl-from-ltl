"""Core algorithm for Sample Efficient LTL learning
"""
import numpy as np
from itertools import product
from .mdp import MDP
import os
import importlib
import multiprocessing
import collections
import random

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T
# from torch.utils.tensorboard import SummaryWriter
# from torch.autograd import Variable

if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt

if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact

Transition = collections.namedtuple("Transition", ("state", "action", "next_state", "reward"))


class LearningAlgo:
    """
    This class is the implementation of our core algorithms.

    Attributes
    ----------
    shape : The shape of the product MDP.

    Parameters
    ----------
    mdp : The MDP model of the environment.

    auto : The automaton obtained from the LTL specification.

    prism: The PRISM model instance with the LTL task defined

    discount : The discount factor.

    U : The upper bound for Max reward.

    eva_frequency: The frequency of evaluating the current policy
    """

    def __init__(self, mdp, auto, prism, U=0.1, discount=0.99, eva_frequency=10000):
        self.mdp = mdp
        self.auto = auto
        self.prism = prism
        self.discount = discount
        self.U = U  # We can also explicitly define a function of discount
        self.shape = auto.shape + mdp.shape + (len(mdp.A) + auto.shape[1],)

        # Create the action matrix
        self.A = np.empty(self.shape[:-1], dtype=np.object)
        for i, q, r, c in self.states():
            self.A[i, q, r, c] = mdp.allowed_actions((r, c)) + [len(mdp.A) + e_a for e_a in auto.eps[q]]

        self.memory = ExperienceReplay(100000)
        self.evaluation_frequency = eva_frequency

    def states(self):
        """
        Iterates through all product states
        """
        n_mdps, n_qs, n_rows, n_cols, n_actions = self.shape
        for i, q, r, c in product(range(n_mdps), range(n_qs), range(n_rows), range(n_cols)):
            yield i, q, r, c

    def random_state(self):
        """
        Generates a random product state.
        """
        n_mdps, n_qs, n_rows, n_cols, n_actions = self.shape
        mdp_state = np.random.randint(n_rows), np.random.randint(n_cols)
        return (np.random.randint(n_pairs), np.random.randint(n_qs)) + mdp_state

    # Actions = ['U', 'D', 'R', 'L']

    def prism_evaluation(self, policy):
        """
        Evaluates the current policy using PRISM
        """
        prob = self.prism.build_model(self.mdp, self.auto, policy)
        return prob



    def step(self, state, action):
        """
        Performs a step in the environment from a state taking an action
        """
        i, q, r, c = state
        experiences = []
        if action < len(self.mdp.A):  # MDP actions
            mdp_states, probs = self.mdp.get_transition_prob((r, c), self.mdp.A[action])  # MDP transition
            next_state = mdp_states[np.random.choice(len(mdp_states), p=probs)]
            q_ = self.auto.delta[q][self.mdp.label[(r, c)]]  # auto transition
            reward = self.U if self.auto.acc[q][self.mdp.label[(r, c)]][i] else 0
            experiences.append((state, action, (i, q_,) + next_state, reward))

            return (i, q_,) + next_state, experiences
        else:  # epsilon-actions
            reward = self.U if self.auto.acc[q][self.mdp.label[r, c]][i] else 0
            experiences.append((state, action, (i, action - len(self.mdp.A), r, c), reward))
            return (i, action - len(self.mdp.A), r, c), experiences

    def counterfact_step(self, state, action, k, counterfactual):
        """
        Performs a step in the environment with counterfactual imagining
        """
        i, q, r, c = state
        experiences = []
        if action < len(self.mdp.A):  # MDP actions
            mdp_states, probs = self.mdp.get_transition_prob((r, c), self.mdp.A[action])  # MDP transition
            next_state = mdp_states[np.random.choice(len(mdp_states), p=probs)]
            q_ = self.auto.delta[q][self.mdp.label[(r, c)]]  # auto transition
            reward = self.U if self.auto.acc[q][self.mdp.label[(r, c)]][i] else 0
            if reward:
                reward = self.U * (k + 1) / (self.K + 1)
                next_k = k + 1 if k < self.K - 1 else k
            else:
                next_k = k

            if counterfactual:
                reachable_states = set()
                for auto in range(self.shape[1]):
                    accept = self.U if self.auto.acc[auto][self.mdp.label[(r, c)]][i] else 0
                    if accept:
                        reward_ = self.U * (k + 1) / (self.K + 1)
                    else:
                        reward_ = 0
                    next_auto = self.auto.delta[auto][self.mdp.label[(r, c)]]
                    exp = (i, auto, r, c)
                    exp_ = (i, next_auto,) + next_state
                    experiences.append((exp, action, exp_, reward_))

            else:
                experiences.append((state, action, (i, q_,) + next_state, reward))
            return (i, q_,) + next_state, next_k, experiences
        else:  # epsilon-actions
            reward = self.U * (k + 1) / self.K if self.auto.acc[q][self.mdp.label[r, c]][i] else 0

            experiences.append((state, action, (i, action - len(self.mdp.A), r, c), reward))
            return (i, action - len(self.mdp.A), r, c), k, experiences

    def efficient_ltl_learning(self, start, T, trails, K, counterfactual=True):
        """
        The algorithm for sample efficient RL from LTL
        """
        T = T if T else np.prod(self.shape[:-1])
        trails = trails if trails else 100000
        self.K = K if K else 0

        print("U,K,discount", self.U, self.K, self.discount)
        Q = np.zeros(self.shape)
        for i, q, r, c in self.states():
            for a in self.A[i, q, r, c]:
                Q[i, q, r, c, a] = 2 * self.U

        epsilon = 0.1
        alpha = 0.1
        print("Q shape", Q.shape)
        probs = []
        for i in range(trails):
            state = (0, self.auto.q0) + (start if start else self.mdp.random_state())
            k, non_accept = 0, 0
            if (i * T) % self.evaluation_frequency == 0:
                policy = np.argmax(Q[:, :, :, :, :], axis=4)
                prob = self.prism_evaluation(policy)
                probs.append(prob)
                print(i, prob)

            # each episode loop
            for t in range(T):
                if np.random.rand() < epsilon or np.max(Q[state]) == 0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])
                i1, q1, r1, c1 = state
                next_state, next_k, experiences = self.counterfact_step(state, action, k, counterfactual)

                if self.auto.acc[q1][self.mdp.label[(r1, c1)]][i1]:
                    non_accept = 0
                else:
                    non_accept += 1
                if non_accept > 300:
                    break

                for exp, action, exp_, reward in experiences:
                    gamma = (1 - reward) if reward else self.discount
                    Q[exp][action] += alpha * (reward + gamma * np.max(Q[exp_]) - Q[exp][action])
                state = next_state
                k = next_k

        return Q, probs


    def csrl_ql(self, start, T, trails):
        """
        The algorithm for the approach by Bozkurt et al.
        """
        T = T if T else np.prod(self.shape[:-1])
        trails = trails if trails else 100000

        Q = np.zeros(self.shape)
        for i, q, r, c in self.states():
            for a in self.A[i, q, r, c]:
                Q[i, q, r, c, a] = 2 * self.U

        epsilon = 0.1
        alpha = 0.1
        print("Q shape", Q.shape)
        probs = []
        for i in range(trails):
            state = (0, self.auto.q0) + (start if start else self.mdp.random_state())
            k, non_accept = 0, 0
            if (i * T) % self.evaluation_frequency == 0:
                policy = np.argmax(Q[:, :, :, :, :], axis=4)
                prob = self.prism_evaluation(policy)
                probs.append(prob)
                print(i, prob)

            # each episode loop
            for t in range(T):
                if np.random.rand() < epsilon or np.max(Q[state]) == 0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])
                i1, q1, r1, c1 = state
                next_state, experiences = self.step(state, action)

                if self.auto.acc[q1][self.mdp.label[(r1, c1)]][i1]:
                    non_accept = 0
                else:
                    non_accept += 1
                if non_accept > 300:
                    break

                for exp, action, exp_, reward in experiences:
                    gamma = (1 - reward) if reward else self.discount
                    Q[exp][action] += alpha * (reward + gamma * np.max(Q[exp_]) - Q[exp][action])
                state = next_state

        return Q, probs


    def lcrl_ql(self, start, T, trails):
        """
        The algorithm for the approach by Hasanbeig et al.
        """
        T = T if T else np.prod(self.shape[:-1])
        trails = trails if trails else 100000

        Q = np.zeros(self.shape)
        for i, q, r, c in self.states():
            for a in self.A[i, q, r, c]:
                Q[i, q, r, c, a] = 0

        epsilon = 0.1
        alpha = 0.1
        print("Q shape", Q.shape)
        probs = []
        for i in range(trails):
            state = (0, self.auto.q0) + (start if start else self.mdp.random_state())
            k, non_accept = 0, 0
            if (i * T) % self.evaluation_frequency == 0:
                policy = np.argmax(Q[:, :, :, :, :], axis=4)
                prob = self.prism_evaluation(policy)
                probs.append(prob)
                print(i, prob)

            # each episode loop
            for t in range(T):
                if np.random.rand() < epsilon or np.max(Q[state]) == 0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])
                i1, q1, r1, c1 = state
                next_state, experiences = self.step(state, action)

                if self.auto.acc[q1][self.mdp.label[(r1, c1)]][i1]:
                    non_accept = 0
                else:
                    non_accept += 1
                if non_accept > 300:
                    break

                for exp, action, exp_, reward in experiences:
                    gamma = self.discount if reward else 1
                    Q[exp][action] += alpha * (reward + gamma * np.max(Q[exp_]) - Q[exp][action])

                state = next_state

        return Q, probs

    def omega_regular_rl(self, start, T, trails, zeta=0.99):
        """
        The algorithm for the approach by Hahn et al.
        """
        T = T if T else np.prod(self.shape[:-1])
        trails = trails if trails else 100000

        Q = np.zeros(self.shape)
        for i, q, r, c in self.states():
            for a in self.A[i, q, r, c]:
                Q[i, q, r, c, a] = 0

        epsilon = 0.1
        alpha = 0.1
        print("Q shape", Q.shape)
        probs = []
        for i in range(trails):
            state = (0, self.auto.q0) + (start if start else self.mdp.random_state())
            k, non_accept = 0, 0
            if (i * T) % self.evaluation_frequency == 0:
                policy = np.argmax(Q[:, :, :, :, :], axis=4)
                prob = self.prism_evaluation(policy)
                probs.append(prob)
                print(i, prob)

            # each episode loop
            for t in range(T):
                if np.random.rand() < epsilon or np.max(Q[state]) == 0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])
                i1, q1, r1, c1 = state
                next_state, experiences = self.step(state, action)

                if self.auto.acc[q1][self.mdp.label[(r1, c1)]][i1]:
                    non_accept = 0
                else:
                    non_accept += 1
                if non_accept > 300:
                    break

                for exp, action, exp_, reward in experiences:
                    if reward:
                        if np.random.rand() < zeta:
                            reward = 0
                        else:
                            reward = 1
                    Q[exp][action] += alpha * (reward + np.max(Q[exp_]) - Q[exp][action])

                    if reward:
                        break

                state = next_state

        return Q, probs


    def plot(self, value=None, policy=None, iq=None, **kwargs):
        """
        Plots the MDP environment with optionally the value and policy on top
        """

        if iq:
            val = value[iq] if value is not None else None
            pol = policy[iq] if policy is not None else None
            self.mdp.plot(val, pol, **kwargs)
        else:
            # A helper function for the sliders
            def plot_value(i, q):
                val = value[i, q] if value is not None else None
                pol = policy[i, q] if policy is not None else None
                self.mdp.plot(val, pol, **kwargs)

            i = IntSlider(value=0, min=0, max=self.shape[0] - 1)
            q = IntSlider(value=self.auto.q0, min=0, max=self.shape[1] - 1)
            interact(plot_value, i=i, q=q)


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # if memory isn't full, add a new experience
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

