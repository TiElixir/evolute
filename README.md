# Creature Evolution RL

A 2D simulation where procedurally-defined **creatures** — rigid bones connected by motorized joints — learn to walk, crawl, and hop via **reinforcement learning** (PPO from scratch in PyTorch). An outer **evolutionary loop** mutates creature morphology across generations, using RL-trained fitness as the selection signal. A **Pygame GUI editor** lets you visually create and edit creature genomes.

---

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

### 2. Verify CUDA

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 3. Run the Editor

```bash
python main.py editor
python main.py editor --genome creature/presets/quadruped.json
```

### 4. Train a Creature

```bash
python main.py train --genome creature/presets/biped.json --timesteps 1000000
python main.py train --genome creature/presets/biped.json --timesteps 200000 --render
```

### 5. Replay a Checkpoint

```bash
python main.py replay \
  --checkpoint checkpoints/biped_v1/best.pt \
  --genome creature/presets/biped.json
```

### 6. Run Evolution

```bash
# Full run
python main.py evolve --config config/default.yaml

# Quick test (4 creatures, 3 generations, 50k steps each)
python main.py evolve --pop-size 4 --generations 3 --per-timesteps 50000
```

### 7. View TensorBoard Logs

```bash
tensorboard --logdir logs/
```

---

## Project Structure

```
evolute/
├── main.py                 # CLI dispatcher
├── requirements.txt
├── config/
│   └── default.yaml        # All hyperparameters
├── creature/
│   ├── morphology.py       # Genome, Bone, Joint dataclasses
│   ├── builder.py          # Pymunk physics builder + Creature class
│   └── presets/
│       ├── biped.json
│       ├── quadruped.json
│       ├── worm.json
│       └── tripod.json
├── environment/
│   ├── creature_env.py     # Gym-like environment
│   └── renderer.py         # Pygame renderer
├── editor/
│   └── creature_editor.py  # Interactive GUI editor
├── rl/
│   ├── networks.py         # ActorCritic neural network
│   ├── buffer.py           # Rollout buffer with GAE
│   ├── normalizer.py       # Welford running normalizer
│   ├── ppo.py              # PPO trainer
│   └── train.py            # Training CLI
├── evolution/
│   ├── genome_ops.py       # Mutation & crossover operators
│   ├── population.py       # Population management
│   └── evolve.py           # Evolution main loop
├── scripts/
│   ├── milestone1_test.py  # Physics build test (no rendering)
│   └── milestone2_test.py  # Renderer + physics sanity test
├── checkpoints/            # Created at runtime
└── logs/                   # TensorBoard logs
```

---

## Genome Format

Creatures are defined as human-editable JSON files. See any file in `creature/presets/` for examples.

Key fields:
- **`bones`**: Rigid body segments. One root bone (`"parent": null`), all others reference a parent by ID.
- **`joints`**: Connect pairs of bones. `is_motorized: true` joints become RL action dimensions.
- **`attach_point`**: Offset (in metres) from the parent bone's local origin where this bone attaches.
- **`angle_limit_deg`**: `[min, max]` rotation range for the joint.

---

## Editor Controls

| Action | Input |
|--------|-------|
| **Select bone/joint** | Left-click |
| **Start Add Bone** | Click "Add Bone" button, then click a parent bone and drag |
| **Edit properties** | Click bone/joint → use sliders in left panel |
| **Delete bone** | Select bone → "Delete Bone" button → confirm with Y |
| **Mirror limb** | Select bone → "Mirror" button |
| **Test Drive** | "Test Drive" button — toggles live physics |
| **Save genome** | "Save" button → type filename |
| **Load genome** | "Load" button → type filename (from `creature/saved/`) |
| **Pan camera** | Left/Right arrow keys |

---

## Observation & Action Spaces

**Observation** (flat `np.float32` vector, size dynamic per genome):
- Per bone (6 values): relative position to torso (x, y), relative angle, velocity (vx, vy), angular velocity
- Global (4 values): torso height, torso angle, torso velocity (vx, vy)
- Per motorized joint (1 value): current angle normalized to limit range
- Per foot/leaf bone (1 value): ground contact flag

**Action**: One continuous value in `[-1, 1]` per motorized joint (after `tanh` squashing).

---

## Reward Function (configurable in `config/default.yaml`)

```python
reward = (
    forward_velocity_x              # move right
    - 0.001 * sum(action²)         # energy penalty
    - 0.5  * abs(torso_angle)      # stay upright
    + 0.05                          # alive bonus
)
if fallen: reward -= 1.0; done = True
```

---

## PPO Hyperparameters (defaults in `config/default.yaml`)

| Parameter | Default |
|-----------|---------|
| `learning_rate` | `3e-4` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_eps` | `0.2` |
| `n_steps` | `2048` |
| `n_epochs` | `10` |
| `minibatch_size` | `256` |
| `entropy_coef` | `0.001` |

---

## Security Notes

- Genome files are loaded with Python's stdlib `json` (no `eval`/`pickle`).
- All genome data is validated by `validate_genome()` before use.
- Editor file save/load sanitizes filenames with `os.path.basename()` and enforces a hardcoded safe directory.
- PyTorch checkpoints use `weights_only=True` to prevent arbitrary code execution.

---

## Performance Tips

- **GPU training**: CUDA is used automatically if available; set `--device cpu` to force CPU.
- **Quick iteration**: Use `--per-timesteps 50000` for fast evolution test runs.
- **Parallel evolution** (stretch goal): The per-genome training loops are independent and can be parallelised with `multiprocessing.Pool` — each worker uses its own pymunk space, and GPU tensors are moved back to CPU for inter-process communication.
- **TensorBoard**: Monitor training progress with `tensorboard --logdir logs/`.
