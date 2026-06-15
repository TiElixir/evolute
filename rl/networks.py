"""
rl/networks.py

ActorCritic neural network for PPO.
- Shared MLP trunk or separate actor/critic trunks.
- Gaussian actor head with state-independent log_std.
- tanh action squashing with corrected log-prob.
- Auto-detects CUDA.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def _make_mlp(in_dim: int, hidden: int, n_layers: int = 2) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden), nn.Tanh()]
        dim = hidden
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Shared-trunk MLP actor-critic for continuous action spaces.

    Args:
        obs_dim:    Observation vector dimension.
        action_dim: Number of continuous actions.
        hidden:     Hidden layer width (default 256).
        device:     torch device string or Device object.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 256,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Shared trunk
        self.trunk = _make_mlp(obs_dim, hidden, n_layers=2)

        # Actor head — outputs mean of Gaussian
        self.actor_head = nn.Linear(hidden, action_dim)

        # State-independent log_std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head — outputs scalar value
        self.critic_head = nn.Linear(hidden, 1)

        self._init_weights()
        self.to(device)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller gain for output layers
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def _trunk_forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def get_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy distribution.

        Returns:
            action:   Tanh-squashed sample, shape (batch, action_dim).
            log_prob: Log probability with tanh correction, shape (batch,).
            value:    State value estimate, shape (batch,).
        """
        features = self._trunk_forward(obs)
        mean = self.actor_head(features)
        std = self.log_std.exp().expand_as(mean)

        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        # tanh-corrected log-prob:  log π(a) = log π(raw) - Σ log(1 - tanh²(raw))
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        log_prob = log_prob_raw - correction

        value = self.critic_head(features).squeeze(-1)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy of given (obs, action) pairs.

        action should be in tanh-squashed space (as returned by get_action).

        Returns:
            log_prob: shape (batch,)
            entropy:  shape (batch,)
            value:    shape (batch,)
        """
        features = self._trunk_forward(obs)
        mean = self.actor_head(features)
        std = self.log_std.exp().expand_as(mean)

        dist = Normal(mean, std)

        # Un-squash action to raw pre-tanh value
        # Clamp to prevent atanh infinity at ±1
        action_clamped = action.clamp(-1 + 1e-6, 1 - 1e-6)
        raw_action = torch.atanh(action_clamped)

        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1.0 - action_clamped.pow(2) + 1e-6).sum(dim=-1)
        log_prob = log_prob_raw - correction

        # Entropy of the pre-tanh Gaussian (approximate; tanh reduces it)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic_head(features).squeeze(-1)
        return log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return only the value estimate (used at rollout boundary)."""
        features = self._trunk_forward(obs)
        return self.critic_head(features).squeeze(-1)
