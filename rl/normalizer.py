"""
rl/normalizer.py

Welford online running mean/variance normaliser for observations.
Thread-safe enough for single-process use.
"""

from __future__ import annotations

import numpy as np


class RunningNormalizer:
    """Welford incremental mean/variance tracker.

    Usage:
        norm = RunningNormalizer(obs_dim)
        norm.update(obs_batch)        # np.ndarray [N, dim]
        normalized = norm.normalize(obs)
    """

    def __init__(self, shape: int, clip: float = 5.0, epsilon: float = 1e-8) -> None:
        self.shape = shape
        self.clip = clip
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Update running stats with a batch of observations.

        x: shape (N, dim) or (dim,)
        """
        x = np.atleast_2d(x).astype(np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        new_var = m2 / total

        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return x normalised to approx zero mean, unit variance, clipped."""
        x = x.astype(np.float64)
        normed = (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = int(d["count"])
