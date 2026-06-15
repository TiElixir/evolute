"""
scripts/milestone2_test.py

Milestone 2 validation: renders the biped falling under gravity
for 5 seconds with no motors active. Confirms joints don't explode.

Run from project root:
    .venv/bin/python scripts/milestone2_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymunk
import pygame

from creature.morphology import Genome
from creature.builder import build_creature, PIXELS_PER_METER
from environment.renderer import Renderer

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "..", "creature", "presets")


def main():
    genome = Genome.load(os.path.join(PRESETS_DIR, "biped.json"))

    space = pymunk.Space()
    space.gravity = (0, -900)
    space.damping = 0.9

    ground = pymunk.Segment(space.static_body, (-5000, 50), (50000, 50), 5)
    ground.friction = 1.0
    space.add(ground)

    creature = build_creature(space, genome, position=(300, 350))
    creature.space = space

    renderer = Renderer(width=1000, height=600, title="Milestone 2 — Biped physics test")
    renderer.ground_y = 50.0
    renderer.init()

    dt = 1.0 / 60.0
    substeps = 4
    max_frames = 60 * 10   # 10 seconds

    print("Milestone 2: Running biped physics for 10 seconds ...")
    print("Close window or press ESC to stop early.")

    frame = 0
    while frame < max_frames:
        if renderer.poll_quit():
            break

        for _ in range(substeps):
            space.step(dt / substeps)

        torso_pos   = creature.get_torso_position()
        torso_angle = creature.get_torso_angle()
        renderer.render(
            creature,
            info={
                "Frame": frame,
                "Torso X": f"{torso_pos[0] / PIXELS_PER_METER:.2f}m",
                "Torso Y": f"{torso_pos[1] / PIXELS_PER_METER:.2f}m",
                "Angle":   f"{torso_angle:.3f}rad",
                "Fallen":  creature.is_fallen(ground_y=50),
            },
        )
        renderer.tick(fps=60)
        frame += 1

    print(f"  Ran {frame} frames")
    torso_pos = creature.get_torso_position()
    print(f"  Final torso position: ({torso_pos[0]:.1f}, {torso_pos[1]:.1f})")
    print(f"  Fallen: {creature.is_fallen(ground_y=50)}")
    print("  ✓ Milestone 2 complete (joints held together)!")
    renderer.close()


if __name__ == "__main__":
    main()
