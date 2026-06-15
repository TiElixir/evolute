"""
evolution/evolve.py

Main evolutionary outer loop.

For each generation:
  - Train each genome's PPO policy for per_genome_timesteps steps.
  - Use final eval reward as fitness.
  - Produce next generation via selection + mutation + crossover.
  - Log stats and save best genome per generation.

Usage:
    python -m evolution.evolve --config config/default.yaml
    python -m evolution.evolve --config config/default.yaml --generations 5 --pop-size 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log_generation_stats(gen: int, population) -> None:
    """Print a table of genome → fitness for the current generation."""
    sorted_inds = population.sorted_by_fitness()
    print(f"\n{'='*70}")
    print(f"  Generation {gen} Summary")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'Genome':<35} {'Fitness':>10}")
    print(f"  {'-'*5} {'-'*35} {'-'*10}")
    for rank, (genome, fitness) in enumerate(sorted_inds, 1):
        fit_str = f"{fitness:.2f}" if fitness is not None else "N/A"
        print(f"  {rank:<5} {genome.name:<35} {fit_str:>10}")
    print(f"\n  Mean fitness: {population.mean_fitness():.2f}")
    best_g, best_f = sorted_inds[0]
    print(f"  Best genome: '{best_g.name}'  fitness={best_f if best_f is not None else 'N/A'}")
    print(f"{'='*70}\n")


def save_best_genome(gen: int, population, output_dir: str = "checkpoints/evolution") -> None:
    """Save the best genome of this generation as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    best_genome, best_fitness = population.sorted_by_fitness()[0]
    best_genome.metadata["fitness"] = best_fitness
    path = os.path.join(output_dir, f"gen_{gen:03d}_best.json")
    best_genome.save(path)
    print(f"  Best genome saved → {path}")


def run_evolution(config: dict, args) -> None:
    """Main evolution loop."""
    ev_cfg = config.get("evolution", {})
    pop_size        = args.pop_size or int(ev_cfg.get("population_size", 8))
    n_generations   = args.generations or int(ev_cfg.get("n_generations", 10))
    per_ts          = args.per_timesteps or int(ev_cfg.get("per_genome_timesteps", 50_000))
    elitism         = int(ev_cfg.get("elitism", 2))
    mutation_rate   = float(ev_cfg.get("mutation_rate", 0.4))
    crossover_rate  = float(ev_cfg.get("crossover_rate", 0.3))
    tournament_k    = int(ev_cfg.get("tournament_k", 3))
    seed_paths      = ev_cfg.get("seed_genomes", ["creature/presets/biped.json"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  CUDA: {torch.cuda.is_available()}  |  Device: {device}")
    print(f"  Population: {pop_size}  |  Generations: {n_generations}  |  "
          f"Per-genome steps: {per_ts:,}\n")

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join("logs", "evolution")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"  TensorBoard logs → {log_dir}")
    except ImportError:
        pass

    # Load seed genomes
    from creature.morphology import Genome
    seed_genomes: List[Genome] = []
    for sp in seed_paths:
        if os.path.isfile(sp):
            seed_genomes.append(Genome.load(sp))
        else:
            print(f"  [warn] Seed genome not found: {sp}")
    if not seed_genomes:
        print("  [error] No seed genomes loaded. Aborting.")
        sys.exit(1)

    # Initialise population
    from evolution.population import Population
    population = Population(size=pop_size, seed_genomes=seed_genomes)

    from environment.creature_env import CreatureEnv
    from rl.networks import ActorCritic
    from rl.ppo import PPOTrainer

    for gen in range(n_generations):
        print(f"\n{'#'*70}")
        print(f"  GENERATION {gen}")
        print(f"{'#'*70}")
        gen_start = time.time()

        for idx in range(pop_size):
            genome = population.get_genome(idx)
            print(f"\n  [{idx+1}/{pop_size}] Training '{genome.name}'  "
                  f"({len(genome.bones)} bones, "
                  f"{len(genome.get_motorized_joints())} motors)")

            try:
                env = CreatureEnv(genome, config=config, render=False)
                ac = ActorCritic(
                    obs_dim=env.observation_dim,
                    action_dim=env.action_dim,
                    hidden=256,
                    device=device,
                )
                trainer = PPOTrainer(
                    env, ac, config, writer=writer,
                    log_prefix=f"gen{gen}/{genome.name[:20]}",
                )
                fitness = trainer.train(total_timesteps=per_ts)
                env.close()
            except Exception as e:
                print(f"  [error] Training failed for '{genome.name}': {e}")
                fitness = -1.0

            population.set_fitness(idx, fitness)
            genome.metadata["fitness"] = fitness

            if writer:
                writer.add_scalar(f"evolution/genome_fitness", fitness, gen * pop_size + idx)

        # Log generation stats
        log_generation_stats(gen, population)
        save_best_genome(gen, population)

        # Log aggregate stats to TensorBoard
        if writer:
            best_f = population.best()[1] or 0.0
            mean_f = population.mean_fitness()
            writer.add_scalar("evolution/best_fitness",  best_f, gen)
            writer.add_scalar("evolution/mean_fitness",  mean_f, gen)

        print(f"  Generation {gen} completed in {time.time() - gen_start:.1f}s")

        # Evolve to next generation (except after last)
        if gen < n_generations - 1:
            population = population.next_generation(
                generation=gen + 1,
                elitism=elitism,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                tournament_k=tournament_k,
            )

    if writer:
        writer.close()
    print("\n  Evolution complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolutionary outer loop")
    parser.add_argument("--config",         default="config/default.yaml")
    parser.add_argument("--generations",    type=int, default=None)
    parser.add_argument("--pop-size",       type=int, default=None)
    parser.add_argument("--per-timesteps",  type=int, default=None, help="PPO steps per genome")
    args = parser.parse_args()

    config: dict = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    run_evolution(config, args)


if __name__ == "__main__":
    main()
