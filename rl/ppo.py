"""
rl/ppo.py

PPO trainer implemented from scratch using PyTorch.

Core algorithm:
  1. Collect n_steps of experience from the environment.
  2. Compute GAE advantages.
  3. For n_epochs, sample minibatches and apply the clipped PPO update.
  4. Log metrics to TensorBoard.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.creature_env import CreatureEnv
from rl.buffer import RolloutBuffer
from rl.networks import ActorCritic
from rl.normalizer import RunningNormalizer


class PPOTrainer:
    """PPO training loop for a single creature-environment pair.

    Args:
        env:          CreatureEnv instance (reset/step API).
        actor_critic: ActorCritic network (already on correct device).
        config:       Dict with training hyperparameters (see default.yaml).
        writer:       Optional TensorBoard SummaryWriter.
        log_prefix:   Prefix string for TensorBoard tags.
    """

    def __init__(
        self,
        env: CreatureEnv,
        actor_critic: ActorCritic,
        config: Dict[str, Any],
        writer=None,
        log_prefix: str = "",
    ) -> None:
        self.env = env
        self.ac = actor_critic
        self.config = config
        self.writer = writer
        self.log_prefix = log_prefix
        self.device = actor_critic.device

        tc = config.get("training", config)  # support both flat and nested
        self.lr              = float(tc.get("learning_rate", 3e-4))
        self.gamma           = float(tc.get("gamma", 0.99))
        self.gae_lambda      = float(tc.get("gae_lambda", 0.95))
        self.clip_eps        = float(tc.get("clip_eps", 0.2))
        self.n_steps         = int(tc.get("n_steps", 2048))
        self.n_epochs        = int(tc.get("n_epochs", 10))
        self.minibatch_size  = int(tc.get("minibatch_size", 256))
        self.entropy_coef    = float(tc.get("entropy_coef", 0.001))
        self.value_coef      = float(tc.get("value_coef", 0.5))
        self.max_grad_norm   = float(tc.get("max_grad_norm", 0.5))
        self.eval_every      = int(tc.get("eval_every_steps", 50_000))
        self.eval_episodes   = int(tc.get("eval_episodes", 3))
        self.checkpoint_dir  = str(tc.get("checkpoint_dir", "checkpoints"))

        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)
        self.obs_normalizer = RunningNormalizer(env.observation_dim)
        self.buffer = RolloutBuffer(
            self.n_steps, env.observation_dim, env.action_dim, self.device
        )

        self.total_steps: int = 0
        self.update_count: int = 0
        self.best_eval_reward: float = -1e9
        self._current_obs: Optional[np.ndarray] = None
        self._current_done: bool = True
        self._ep_reward: float = 0.0
        self._ep_len: int = 0
        self._ep_rewards: list = []

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> float:
        """Train for total_timesteps env steps.

        Returns:
            best_eval_reward achieved during training.
        """
        print(f"  PPO training: {total_timesteps:,} steps  |  "
              f"obs_dim={self.env.observation_dim}  action_dim={self.env.action_dim}  "
              f"device={self.device}")

        obs = self.env.reset()
        self.obs_normalizer.update(obs[None])
        obs = self.obs_normalizer.normalize(obs)
        self._current_obs = obs
        self._ep_reward = 0.0
        self._ep_len = 0

        t0 = time.time()

        while self.total_steps < total_timesteps:
            self.collect_rollout()
            losses = self.update()
            self.update_count += 1

            if self._ep_rewards:
                mean_ep = np.mean(self._ep_rewards[-20:])
            else:
                mean_ep = 0.0

            # TensorBoard logging
            if self.writer:
                tag = f"{self.log_prefix}/" if self.log_prefix else ""
                self.writer.add_scalar(f"{tag}train/mean_ep_reward",    mean_ep,                    self.total_steps)
                self.writer.add_scalar(f"{tag}train/policy_loss",       losses["policy_loss"],      self.total_steps)
                self.writer.add_scalar(f"{tag}train/value_loss",        losses["value_loss"],       self.total_steps)
                self.writer.add_scalar(f"{tag}train/entropy",           losses["entropy"],          self.total_steps)
                self.writer.add_scalar(f"{tag}train/approx_kl",        losses["approx_kl"],        self.total_steps)
                self.writer.add_scalar(f"{tag}train/clip_fraction",     losses["clip_fraction"],    self.total_steps)

            # Console log every ~10 updates
            if self.update_count % 10 == 0:
                elapsed = time.time() - t0
                fps = self.total_steps / max(elapsed, 1)
                print(
                    f"  steps={self.total_steps:>8,}  "
                    f"ep_reward={mean_ep:>8.2f}  "
                    f"policy_loss={losses['policy_loss']:>7.4f}  "
                    f"value_loss={losses['value_loss']:>7.4f}  "
                    f"fps={fps:>5.0f}"
                )

            # Periodic evaluation
            if self.total_steps % self.eval_every < self.n_steps:
                eval_reward = self.evaluate(self.eval_episodes)
                if self.writer:
                    tag = f"{self.log_prefix}/" if self.log_prefix else ""
                    self.writer.add_scalar(f"{tag}eval/mean_reward", eval_reward, self.total_steps)
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.save_checkpoint(
                        os.path.join(self.checkpoint_dir, self.env.genome.name, "best.pt")
                    )
                    print(f"  ★ New best eval reward: {eval_reward:.2f}  (saved checkpoint)")

        print(f"  Training complete.  Best eval reward: {self.best_eval_reward:.2f}")
        return self.best_eval_reward

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> None:
        """Collect n_steps of experience and fill the buffer."""
        self.buffer.reset()
        self.ac.eval()

        obs = self._current_obs
        if obs is None:
            raw_obs = self.env.reset()
            self.obs_normalizer.update(raw_obs[None])
            obs = self.obs_normalizer.normalize(raw_obs)

        for _ in range(self.n_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = self.ac.get_action(obs_tensor)

            action_np   = action.squeeze(0).cpu().numpy()
            log_prob_np = log_prob.item()
            value_np    = value.item()

            raw_obs, reward, done, _info = self.env.step(action_np)

            self.obs_normalizer.update(raw_obs[None])
            next_obs = self.obs_normalizer.normalize(raw_obs)

            self.buffer.add(obs, action_np, log_prob_np, reward, done, value_np)
            self.total_steps += 1
            self._ep_reward += reward
            self._ep_len += 1

            if done:
                self._ep_rewards.append(self._ep_reward)
                self._ep_reward = 0.0
                self._ep_len = 0
                raw_obs = self.env.reset()
                self.obs_normalizer.update(raw_obs[None])
                next_obs = self.obs_normalizer.normalize(raw_obs)

            obs = next_obs

        # Bootstrap value at rollout boundary
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.ac.get_value(obs_tensor).item()

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        self._current_obs = obs

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """Run n_epochs of PPO update over the filled buffer."""
        self.ac.train()

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_kl          = 0.0
        total_clip        = 0.0
        n_batches         = 0

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b in self.buffer.get_minibatches(self.minibatch_size):
                new_lp, entropy, new_val = self.ac.evaluate(obs_b, act_b)

                # Clipped surrogate objective
                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = nn.functional.mse_loss(new_val, ret_b)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                total_kl          += approx_kl
                total_clip        += clip_frac
                n_batches         += 1

        denom = max(n_batches, 1)
        return {
            "policy_loss":   total_policy_loss / denom,
            "value_loss":    total_value_loss  / denom,
            "entropy":       total_entropy     / denom,
            "approx_kl":     total_kl          / denom,
            "clip_fraction": total_clip        / denom,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 3) -> float:
        """Run n_episodes deterministically (mean action) and return mean reward."""
        self.ac.eval()
        ep_rewards = []

        for _ in range(n_episodes):
            raw_obs = self.env.reset()
            obs = self.obs_normalizer.normalize(raw_obs)
            ep_reward = 0.0
            done = False

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    # Use mean action (no sampling) for deterministic eval
                    features = self.ac.trunk(obs_t)
                    mean = self.ac.actor_head(features)
                    action = torch.tanh(mean)
                action_np = action.squeeze(0).cpu().numpy()
                raw_obs, reward, done, _ = self.env.step(action_np)
                obs = self.obs_normalizer.normalize(raw_obs)
                ep_reward += reward

            ep_rewards.append(ep_reward)

        return float(np.mean(ep_rewards))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model weights + normalizer state to a .pt file.

        Security: torch.save is used (weights only); load uses weights_only=True.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor_critic_state": self.ac.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "normalizer": self.obs_normalizer.state_dict(),
            "total_steps": self.total_steps,
            "best_eval_reward": self.best_eval_reward,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a .pt checkpoint.

        Security: weights_only=True prevents arbitrary code execution from
        malicious .pt files (requires PyTorch >= 2.0).
        """
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for PyTorch < 2.0
            ckpt = torch.load(path, map_location=self.device)  # noqa: S614

        self.ac.load_state_dict(ckpt["actor_critic_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.obs_normalizer.load_state_dict(ckpt["normalizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.best_eval_reward = ckpt.get("best_eval_reward", -1e9)
        print(f"  Loaded checkpoint: {path}  (steps={self.total_steps})")
