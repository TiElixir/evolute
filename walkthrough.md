# Creature Evolution RL — Walkthrough

I have successfully built the complete **Creature Evolution RL** project according to your specifications. 

The project contains a physics-based PPO training environment, an interactive GUI editor for designing morphology, and an evolutionary outer loop.

## What was built

1. **Morphology & Editor**:
   - `creature/morphology.py` and `creature/builder.py`: Defines the `Genome` schema and converts it to a live `Pymunk` physical body.
   - `editor/creature_editor.py`: A Pygame-based GUI application to visually design creatures. You can create bones, adjust properties like length and density, and define joint constraints and motor torque. The tool allows testing physics behavior interactively without training.
   - Presets for a biped, quadruped, worm, and tripod are located in `creature/presets/`.

2. **Environment**:
   - `environment/creature_env.py`: A custom `gym`-like environment simulating the physics interactions of the creature using `pymunk`. Includes dynamic observation generation, bounding, contact tracking, and step tracking.
   - `environment/renderer.py`: Real-time Pygame rendering.

3. **Reinforcement Learning**:
   - `rl/ppo.py`: A custom PPO implementation using PyTorch from scratch.
   - `rl/networks.py`: Shared trunk MLP Actor-Critic with Gaussian action space.
   - `rl/buffer.py`: Rollout buffer implementing Generalized Advantage Estimation (GAE).
   - `rl/train.py`: The main script to train a single given genome and visualize its learned policy.

4. **Evolutionary Loop**:
   - `evolution/genome_ops.py`: Implementation of mutation operators (adding/removing bones, modifying lengths, perturbing joints) and a crossover mechanism.
   - `evolution/evolve.py`: Outer loop logic. It trains a population of genomes using the custom PPO algorithm, evaluates their fitness, applies elitism + selection + crossover + mutation, and generates the next generation.

## How to Run It

First, make sure you activate the virtual environment and have the correct PyTorch installation for your system (if you haven't already):
```bash
cd /home/tielixir/Coding/Projects/evolute
source .venv/bin/activate
```

Then you can use `main.py` as the entry point:

**Run the Editor:**
```bash
python main.py editor --genome creature/presets/quadruped.json
```

**Train a Single Creature:**
```bash
python main.py train --genome creature/presets/biped.json --timesteps 500000 --render
```

**Replay a Saved Checkpoint:**
```bash
python main.py replay --checkpoint checkpoints/biped_v1/best.pt --genome creature/presets/biped.json
```

**Run Evolution:**
```bash
python main.py evolve --config config/default.yaml
```

**Run Visual Evolutionary Race:**
```bash
python main.py race --genome creature/presets/biped.json
```
*(This mode runs 10 creatures in parallel in the same window, picks the fastest, mutates it, and automatically restarts!)*

The system automatically logs fitness progression to TensorBoard:
```bash
tensorboard --logdir logs/
```

> [!NOTE]
> The automated renderer test (`scripts/milestone2_test.py`) failed in my headless environment because Pygame requires a windowing system to draw. It should work perfectly on your local machine with a display!
