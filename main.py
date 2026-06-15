"""
main.py — CLI dispatcher for Creature Evolution RL

Subcommands:
  editor  — open the bone/joint GUI editor
  train   — train a single creature genome with PPO
  evolve  — run the evolutionary outer loop
  replay  — replay a trained creature checkpoint

Usage:
  python main.py editor
  python main.py editor --genome creature/presets/biped.json

  python main.py train --genome creature/presets/biped.json --timesteps 1000000
  python main.py train --genome creature/presets/biped.json --timesteps 200000 --render

  python main.py evolve --config config/default.yaml --generations 5 --pop-size 4
  python main.py evolve --per-timesteps 50000

  python main.py replay --checkpoint checkpoints/biped_v1/best.pt \\
                        --genome creature/presets/biped.json
"""

import argparse
import os
import sys

import torch


def _cuda_info() -> None:
    print("=" * 60)
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)


def cmd_editor(args: argparse.Namespace) -> None:
    from editor.creature_editor import main as editor_main
    # Override sys.argv so the editor's own argparse sees --genome
    sys.argv = ["editor"]
    if args.genome:
        sys.argv += ["--genome", args.genome]
    editor_main()


def cmd_train(args: argparse.Namespace) -> None:
    _cuda_info()
    # Delegate to rl/train.py's main()
    sys.argv = ["train", "--genome", args.genome]
    if args.timesteps:
        sys.argv += ["--timesteps", str(args.timesteps)]
    if args.render:
        sys.argv += ["--render"]
    if args.checkpoint:
        sys.argv += ["--checkpoint", args.checkpoint]
    if args.config:
        sys.argv += ["--config", args.config]
    if args.device:
        sys.argv += ["--device", args.device]
    from rl.train import main as train_main
    train_main()


def cmd_evolve(args: argparse.Namespace) -> None:
    _cuda_info()
    import yaml
    config: dict = {}
    config_path = args.config or "config/default.yaml"
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    from evolution.evolve import run_evolution
    run_evolution(config, args)


def cmd_race(args: argparse.Namespace) -> None:
    _cuda_info()
    import yaml
    config: dict = {}
    config_path = args.config or "config/default.yaml"
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    from evolution.race import run_race
    run_race(config, args)


def cmd_replay(args: argparse.Namespace) -> None:
    """Load a checkpoint and run a rendered evaluation episode."""
    _cuda_info()
    import yaml
    import numpy as np

    if not args.genome:
        print("[error] --genome is required for replay")
        sys.exit(1)
    if not args.checkpoint:
        print("[error] --checkpoint is required for replay")
        sys.exit(1)

    config: dict = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    from creature.morphology import Genome
    genome = Genome.load(args.genome)

    from environment.creature_env import CreatureEnv
    env = CreatureEnv(genome, config=config, render=True)

    from rl.networks import ActorCritic
    ac = ActorCritic(env.observation_dim, env.action_dim, hidden=256, device=device)

    from rl.ppo import PPOTrainer
    trainer = PPOTrainer(env, ac, config)
    trainer.load_checkpoint(args.checkpoint)

    print(f"\n  Replaying '{genome.name}'  (checkpoint: {args.checkpoint})")
    print("  Press ESC or close window to quit.\n")

    n_episodes = getattr(args, "episodes", 3)
    for ep in range(n_episodes):
        raw_obs = env.reset()
        obs = trainer.obs_normalizer.normalize(raw_obs)
        done = False
        ep_reward = 0.0
        step = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                features = ac.trunk(obs_t)
                mean = ac.actor_head(features)
                action = torch.tanh(mean)
            raw_obs, reward, done, _ = env.step(action.squeeze(0).cpu().numpy())
            obs = trainer.obs_normalizer.normalize(raw_obs)
            ep_reward += reward
            step += 1
            if env._renderer and env._renderer.poll_quit():
                env.close()
                return
        print(f"  Episode {ep+1}: steps={step}  reward={ep_reward:.2f}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Creature Evolution RL — Main CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- editor ---
    p_editor = sub.add_parser("editor", help="Open the creature genome editor")
    p_editor.add_argument("--genome", default="creature/presets/biped.json")

    # --- train ---
    p_train = sub.add_parser("train", help="Train a creature genome with PPO")
    p_train.add_argument("--genome",      required=True)
    p_train.add_argument("--config",      default="config/default.yaml")
    p_train.add_argument("--timesteps",   type=int, default=None)
    p_train.add_argument("--device",      default=None)
    p_train.add_argument("--render",      action="store_true")
    p_train.add_argument("--checkpoint",  default=None)

    # --- evolve ---
    p_evolve = sub.add_parser("evolve", help="Run the evolutionary outer loop")
    p_evolve.add_argument("--config",         default="config/default.yaml")
    p_evolve.add_argument("--generations",    type=int, default=None)
    p_evolve.add_argument("--pop-size",       type=int, default=None)
    p_evolve.add_argument("--per-timesteps",  type=int, default=None)

    # --- replay ---
    p_replay = sub.add_parser("replay", help="Replay a trained creature")
    p_replay.add_argument("--checkpoint",  required=True)
    p_replay.add_argument("--genome",      required=True)
    p_replay.add_argument("--config",      default="config/default.yaml")
    p_replay.add_argument("--device",      default=None)
    p_replay.add_argument("--episodes",    type=int, default=3)

    # --- race ---
    p_race = sub.add_parser("race", help="Run the visual evolutionary race mode")
    p_race.add_argument("--genome", default="creature/presets/biped.json")
    p_race.add_argument("--config", default="config/default.yaml")

    args = parser.parse_args()

    dispatch = {
        "editor": cmd_editor,
        "train":  cmd_train,
        "evolve": cmd_evolve,
        "replay": cmd_replay,
        "race": cmd_race,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
