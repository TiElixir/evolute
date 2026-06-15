"""
scripts/milestone1_test.py

Milestone 1 validation: load each preset genome, build it in a Pymunk space,
and print all body positions. No rendering — just physics object creation.

Run from project root:
    python scripts/milestone1_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymunk

from creature.morphology import Genome, validate_genome
from creature.builder import build_creature, PIXELS_PER_METER

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "..", "creature", "presets")


def test_preset(preset_name: str) -> None:
    path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
    print(f"\n{'='*60}")
    print(f"  Loading preset: {preset_name}")
    print(f"{'='*60}")

    genome = Genome.load(path)
    print(f"  Genome: '{genome.name}'")
    print(f"  Bones ({len(genome.bones)}): {[b.id for b in genome.bones]}")
    print(f"  Joints ({len(genome.joints)}): {[j.id for j in genome.joints]}")
    print(f"  Motorized joints: {len(genome.get_motorized_joints())}")

    # Build physics space
    space = pymunk.Space()
    space.gravity = (0, -900)

    # Add static ground
    ground = pymunk.Segment(space.static_body, (-500, 50), (5000, 50), 5)
    ground.friction = 1.0
    space.add(ground)

    creature = build_creature(space, genome, position=(300, 300))

    print(f"\n  Body positions (pixels):")
    for bid, body in creature.bodies.items():
        pos = body.position
        print(f"    [{bid}]: ({pos.x:.1f}, {pos.y:.1f})  angle={body.angle:.3f} rad")

    print(f"\n  Foot bones: {creature.foot_ids}")
    print(f"  Observation dim: {creature.observation_dim}")
    print(f"  Action dim:      {creature.action_dim}")

    # Sanity: step physics 10 frames
    dt = 1.0 / 60.0
    for _ in range(10):
        space.step(dt)

    print(f"\n  After 10 physics steps:")
    torso_pos = creature.get_torso_position()
    print(f"    Torso: ({torso_pos[0]:.1f}, {torso_pos[1]:.1f})")
    print(f"    Fallen: {creature.is_fallen(ground_y=50)}")
    print(f"  ✓ Preset '{preset_name}' OK")


def main() -> None:
    presets = ["biped", "quadruped", "worm", "tripod"]
    print("Milestone 1: Morphology + Builder Test")
    for preset in presets:
        test_preset(preset)
    print(f"\n{'='*60}")
    print("  All presets passed Milestone 1!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
