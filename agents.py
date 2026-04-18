"""
MovieLens RL — Bandit & Q-Learning Agents
==========================================
Adapted from the Cybersecurity Guardian bandit_core.py and qlearning_agent.py.
The core math is identical — only the arm/state definitions change.
"""

import numpy as np
from data_loader import ARMS, N_ARMS, STATE_NAMES, N_STATES

N_FEATURES = 7   # context vector dimensionality
ALPHA       = 1.0 # LinUCB exploration coefficient


# ---------------------------------------------------------------------------
# LinUCB Agent
# ---------------------------------------------------------------------------

class LinUCBAgent:
    """
    LinUCB (Disjoint model) contextual bandit for genre recommendation.

    Each arm a (genre) maintains:
        A_a : d×d matrix (initialized to identity)
        b_a : d-vector  (initialized to zeros)
        θ_a = A_a^{-1} b_a  — learned weight vector

    UCB score: θ_a^T x + α * sqrt(x^T A_a^{-1} x)
    """
    def __init__(self, n_arms=N_ARMS, n_features=N_FEATURES, alpha=ALPHA):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features)    for _ in range(n_arms)]
        self.total_reward = 0.0
        self.n_updates = 0

    def select_arm(self, context):
        x = context.reshape(-1)
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
        chosen = int(np.argmax(scores))
        return chosen, scores

    def update(self, arm, context, reward):
        x = context.reshape(-1)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.total_reward += reward
        self.n_updates += 1

    def get_theta(self):
        """Return learned weight matrix (n_arms × n_features)."""
        weights = np.zeros((self.n_arms, self.n_features))
        for a in range(self.n_arms):
            weights[a] = np.linalg.inv(self.A[a]) @ self.b[a]
        return weights


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-Learning with ε-greedy exploration.
    States = 6 (age_group × gender), Arms = 6 genres.

    Q(s,a) ← Q(s,a) + α [r − Q(s,a)]   (γ=0, bandit setting)
    """
    def __init__(self, alpha=0.05, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.9999, seed=42):
        self.alpha         = alpha
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)

        # Optimistic init at 0.5 encourages early exploration
        self.Q             = np.full((N_STATES, N_ARMS), 0.5)
        self.visit_counts  = np.zeros((N_STATES, N_ARMS), dtype=int)
        self.epsilon_log   = []
        self.total_reward  = 0.0

    def select_arm(self, state, context=None):
        if self.rng.random() < self.epsilon:
            chosen = int(self.rng.integers(0, N_ARMS))
        else:
            chosen = int(np.argmax(self.Q[state]))
        self.epsilon_log.append(self.epsilon)
        return chosen

    def update(self, state, arm, reward):
        td_error = reward - self.Q[state, arm]
        self.Q[state, arm] += self.alpha * td_error
        self.visit_counts[state, arm] += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_reward += reward


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class RandomAgent:
    """Selects a random genre each time — pure exploration baseline."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.total_reward = 0.0

    def select_arm(self, context=None, state=None):
        return int(self.rng.integers(0, N_ARMS))

    def update(self, arm, context, reward):
        self.total_reward += reward


class PopularityAgent:
    """
    Always recommends Drama (arm=2) — the most-rated genre.
    Represents a naive popularity-based baseline (no personalization).
    """
    def __init__(self):
        self.total_reward = 0.0

    def select_arm(self, context=None, state=None):
        return 2  # Drama — highest volume in dataset

    def update(self, arm, context, reward):
        self.total_reward += reward
