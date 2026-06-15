"""
rl/buffer.py

Rollout buffer for on-policy PPO training.
Stores a fixed-length trajectory, then computes advantages via
Generalized Advantage Estimation (GAE).
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-length rollout buffer with GAE computation.

    Args:
        n_steps:    Rollout length (transitions per update).
        obs_dim:    Observation dimension.
        action_dim: Action dimension.
        device:     Torch device for returned tensors.
    """

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs      = np.zeros((n_steps, obs_dim),  dtype=np.float32)
        self.actions  = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards  = np.zeros(n_steps, dtype=np.float32)
        self.dones    = np.zeros(n_steps, dtype=np.float32)
        self.values   = np.zeros(n_steps, dtype=np.float32)

        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)

        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.obs[self._ptr]       = obs
        self.actions[self._ptr]   = action
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr]   = reward
        self.dones[self._ptr]     = float(done)
        self.values[self._ptr]    = value
        self._ptr += 1

    @property
    def full(self) -> bool:
        return self._ptr >= self.n_steps

    def reset(self) -> None:
        self._ptr = 0

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> None:
        """Compute GAE advantages and returns (in-place).

        δ_t = r_t + γ·(1−d_t)·V_{t+1} − V_t
        A_t = δ_t + γ·λ·(1−d_t)·A_{t+1}
        returns = advantages + values
        """
        advantages = np.zeros(self.n_steps + 1, dtype=np.float32)
        values_ext = np.append(self.values, last_value)  # V[t+1] at boundary

        for t in reversed(range(self.n_steps)):
            not_done = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * not_done * values_ext[t + 1]
                - values_ext[t]
            )
            advantages[t] = delta + gamma * lam * not_done * advantages[t + 1]

        self.advantages = advantages[:self.n_steps]
        self.returns    = self.advantages + self.values

        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std  = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_minibatches(
        self, minibatch_size: int
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Yield shuffled minibatches as (obs, actions, log_probs, advantages, returns)."""
        indices = np.random.permutation(self.n_steps)
        obs_t      = torch.as_tensor(self.obs,      device=self.device)
        actions_t  = torch.as_tensor(self.actions,  device=self.device)
        log_probs_t = torch.as_tensor(self.log_probs, device=self.device)
        adv_t      = torch.as_tensor(self.advantages, device=self.device)
        returns_t  = torch.as_tensor(self.returns,   device=self.device)

        for start in range(0, self.n_steps, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield (
                obs_t[idx],
                actions_t[idx],
                log_probs_t[idx],
                adv_t[idx],
                returns_t[idx],
            )
