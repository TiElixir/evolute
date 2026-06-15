"""
rl/train.py

CLI entry point for training a single creature genome with PPO.

Usage:
    python -m rl.train --genome creature/presets/biped.json --timesteps 1000000
    python -m rl.train --genome creature/presets/biped.json --timesteps 200000 --render
    python -m rl.train --genome creature/presets/biped.json --checkpoint checkpoints/biped_v1/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a creature with PPO")
    parser.add_argument("--genome",      required=True, help="Path to genome JSON file")
    parser.add_argument("--config",      default="config/default.yaml", help="YAML config path")
    parser.add_argument("--timesteps",   type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--device",      default=None,  help="cuda / cpu (auto-detected if omitted)")
    parser.add_argument("--render",      action="store_true", help="Show render window during eval")
    parser.add_argument("--checkpoint",  default=None,  help="Resume from this checkpoint .pt file")
    parser.add_argument("--log-dir",     default="logs", help="TensorBoard log directory")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # CUDA check
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"  Using device: {device}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    config_path = args.config
    config: dict = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"  [warn] Config not found at '{config_path}', using defaults.")

    tc = config.get("training", {})
    total_timesteps = args.timesteps or int(tc.get("total_timesteps", 1_000_000))

    # ------------------------------------------------------------------ #
    # Load genome
    # ------------------------------------------------------------------ #
    from creature.morphology import Genome
    genome_path = args.genome
    if not os.path.isfile(genome_path):
        print(f"  [error] Genome file not found: {genome_path}")
        sys.exit(1)
    genome = Genome.load(genome_path)
    print(f"  Genome: '{genome.name}'  ({len(genome.bones)} bones, "
          f"{len(genome.get_motorized_joints())} motorized joints)")

    # ------------------------------------------------------------------ #
    # Build environment
    # ------------------------------------------------------------------ #
    from environment.creature_env import CreatureEnv
    env = CreatureEnv(genome, config=config, render=args.render)
    print(f"  obs_dim={env.observation_dim}  action_dim={env.action_dim}")

    # ------------------------------------------------------------------ #
    # Build network
    # ------------------------------------------------------------------ #
    from rl.networks import ActorCritic
    ac = ActorCritic(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden=256,
        device=device,
    )
    total_params = sum(p.numel() for p in ac.parameters())
    print(f"  Network parameters: {total_params:,}")

    # ------------------------------------------------------------------ #
    # TensorBoard writer
    # ------------------------------------------------------------------ #
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(args.log_dir, genome.name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"  TensorBoard logs → {log_dir}")
    except ImportError:
        print("  [warn] TensorBoard not available; logging disabled.")

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    from rl.ppo import PPOTrainer
    trainer = PPOTrainer(env, ac, config, writer=writer, log_prefix=genome.name)

    # Resume from checkpoint if provided
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)
        else:
            print(f"  [warn] Checkpoint not found: {args.checkpoint}")

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    print(f"\n  Starting training for {total_timesteps:,} timesteps ...\n")
    best_reward = trainer.train(total_timesteps)

    # Final checkpoint
    final_path = os.path.join(tc.get("checkpoint_dir", "checkpoints"), genome.name, "final.pt")
    trainer.save_checkpoint(final_path)
    print(f"\n  Final checkpoint saved → {final_path}")
    print(f"  Best eval reward: {best_reward:.2f}")

    if writer:
        writer.close()

    env.close()

    # ------------------------------------------------------------------ #
    # Optional: show replay with render
    # ------------------------------------------------------------------ #
    if args.render:
        print("\n  Opening render window for final evaluation ...")
        env_render = CreatureEnv(genome, config=config, render=True)
        env_render.reset()
        import numpy as np
        obs = env_render.reset()
        from rl.normalizer import RunningNormalizer
        norm = RunningNormalizer(env_render.observation_dim)
        norm.load_state_dict(trainer.obs_normalizer.state_dict())
        done = False
        while not done:
            obs_t = torch.as_tensor(norm.normalize(obs), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                features = ac.trunk(obs_t)
                mean = ac.actor_head(features)
                action = torch.tanh(mean)
            obs, _, done, _ = env_render.step(action.squeeze(0).cpu().numpy())
            if env_render._renderer and env_render._renderer.poll_quit():
                break
        env_render.close()


if __name__ == "__main__":
    main()
