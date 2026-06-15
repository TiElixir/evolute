"""
environment/creature_env.py

Gym-like environment wrapper for a single creature.

API:
    env = CreatureEnv(genome, config)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    env.render()
    env.close()

Observation and action dimensions are derived dynamically from the genome.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pymunk

from creature.builder import Creature, build_creature, PIXELS_PER_METER
from creature.morphology import Genome


# ---------------------------------------------------------------------------
# Default config (can be overridden from YAML)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "physics": {
        "gravity": -900,
        "dt": 1.0 / 60.0,
        "substeps": 4,
        "ground_y": 50,
    },
    "reward_weights": {
        "forward_velocity": 1.0,
        "energy_penalty": 0.001,
        "upright_penalty": 0.5,
        "alive_bonus": 0.05,
        "fall_penalty": 1.0,
    },
    "max_steps": 1000,
    "spawn_x": 300,
    "spawn_height_above_ground": 250,
}


class CreatureEnv:
    """Single-creature physics environment.

    Args:
        genome:     The creature genome describing morphology.
        config:     Config dict (merged over DEFAULT_CONFIG).
        render:     If True, a Renderer is created and shown on each step.
    """

    def __init__(
        self,
        genome: Genome,
        config: Optional[Dict[str, Any]] = None,
        render: bool = False,
    ) -> None:
        self.genome = genome
        self.config = _deep_merge(DEFAULT_CONFIG, config or {})
        self._render_mode = render

        # Derived dims — computed once from genome
        self._obs_dim: Optional[int] = None
        self._action_dim: Optional[int] = None

        # Runtime state
        self.space: Optional[pymunk.Space] = None
        self.creature: Optional[Creature] = None
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._prev_torso_x: float = 0.0
        self._ground_segment: Optional[pymunk.Shape] = None
        self._ground_y: float = float(self.config["physics"]["ground_y"])
        self._renderer = None

        # Foot contact tracking via collision handlers
        self._foot_in_contact: Dict[str, bool] = {}

        # Pre-compute dims from a throw-away build
        self._compute_dims()

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self) -> np.ndarray:
        """Reset simulation, return initial observation."""
        self._teardown()
        self._build_space()
        self._step_count = 0
        self._episode_reward = 0.0
        torso_x, _ = self.creature.get_torso_position()
        self._prev_torso_x = torso_x
        obs = self.creature.get_observation()
        if self._render_mode:
            self._ensure_renderer()
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply action, step physics, return (obs, reward, done, info)."""
        assert self.creature is not None, "Call reset() before step()."

        # Apply tanh-bounded action to motors
        action = np.tanh(action)
        self.creature.apply_action(action)

        # Step physics (multiple substeps for stability)
        physics = self.config["physics"]
        dt = physics["dt"]
        substeps = physics["substeps"]
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.space.step(sub_dt)

        self._update_foot_contacts()

        # Observation
        obs = self.creature.get_observation()

        # Reward
        reward, done = self._compute_reward(action)
        self._episode_reward += reward
        self._step_count += 1

        # Update prev torso x
        torso_x, _ = self.creature.get_torso_position()
        self._prev_torso_x = torso_x

        # Max steps
        if self._step_count >= self.config["max_steps"]:
            done = True

        info = {
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "torso_x": torso_x,
            "fallen": self.creature.is_fallen(self._ground_y),
        }

        if self._render_mode and self._renderer is not None:
            self._renderer.render(
                self.creature,
                info={
                    "Step": self._step_count,
                    "Reward": f"{reward:.3f}",
                    "Episode": f"{self._episode_reward:.1f}",
                    "X": f"{torso_x / PIXELS_PER_METER:.2f}m",
                },
                camera_follow=True,
            )
            self._renderer.tick(fps=60)

        return obs, reward, done, info

    def render(self) -> None:
        """Force a render frame (call after step if render=False at init)."""
        if self._renderer is None:
            self._ensure_renderer()
        if self.creature:
            torso_x, _ = self.creature.get_torso_position()
            self._renderer.render(
                self.creature,
                info={"Step": self._step_count, "X": f"{torso_x / PIXELS_PER_METER:.2f}m"},
            )
            self._renderer.tick()

    def close(self) -> None:
        """Release all resources."""
        self._teardown()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_dims(self) -> None:
        """Build a throw-away space to determine obs/action dims."""
        space = pymunk.Space()
        creature = build_creature(space, self.genome, position=(300, 300))
        self._obs_dim = creature.observation_dim
        self._action_dim = creature.action_dim
        # Clean up
        for body in list(space.bodies):
            space.remove(body)
        for shape in list(space.shapes):
            space.remove(shape)
        for constraint in list(space.constraints):
            space.remove(constraint)

    def _build_space(self) -> None:
        """Create pymunk space, ground, and creature."""
        self.space = pymunk.Space()
        self.space.gravity = (0, self.config["physics"]["gravity"])
        self.space.damping = 0.9

        # Static ground segment
        gy = self._ground_y
        seg = pymunk.Segment(self.space.static_body, (-5000, gy), (50000, gy), 5)
        seg.friction = 1.0
        seg.elasticity = 0.0
        self._ground_segment = seg
        self.space.add(seg)

        # Spawn creature
        spawn_x = self.config.get("spawn_x", 300)
        spawn_h = self.config.get("spawn_height_above_ground", 250)
        spawn_y = self._ground_y + spawn_h
        self.creature = build_creature(self.space, self.genome, position=(spawn_x, spawn_y))
        self.creature.space = self.space

        # Set up foot contact detection
        self._foot_in_contact = {fid: False for fid in self.creature.foot_ids}
        self._setup_collision_handlers()

    def _setup_collision_handlers(self) -> None:
        """Register collision handlers for foot bones."""
        GROUND_CATEGORY = 0b01
        FOOT_CATEGORY = 0b10

        # Tag ground segment
        self._ground_segment.filter = pymunk.ShapeFilter(categories=GROUND_CATEGORY)

        # Tag foot shapes
        for fid in self.creature.foot_ids:
            shape = self.creature.shapes[fid]
            shape.filter = pymunk.ShapeFilter(categories=FOOT_CATEGORY)

        # Collision handler: foot ↔ ground
        handler = self.space.add_collision_handler(0, 0)  # catch-all

        env_ref = self

        def begin(arbiter, space, data):
            shapes = arbiter.shapes
            for s in shapes:
                for fid in env_ref.creature.foot_ids:
                    if s is env_ref.creature.shapes.get(fid):
                        env_ref._foot_in_contact[fid] = True
            return True

        def separate(arbiter, space, data):
            shapes = arbiter.shapes
            for s in shapes:
                for fid in env_ref.creature.foot_ids:
                    if s is env_ref.creature.shapes.get(fid):
                        env_ref._foot_in_contact[fid] = False
            return True

        handler.begin = begin
        handler.separate = separate

    def _update_foot_contacts(self) -> None:
        """Sync foot contact flags into the creature observation."""
        if self.creature is None:
            return
        for fid in self.creature.foot_ids:
            in_contact = self._foot_in_contact.get(fid, False)
            self.creature.set_foot_contact(fid, in_contact)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool]:
        """Compute per-step reward and done flag."""
        w = self.config["reward_weights"]
        torso_x, _ = self.creature.get_torso_position()
        torso_vx, _ = self.creature.get_torso_velocity()
        torso_angle = self.creature.get_torso_angle()

        forward_vel = torso_vx / PIXELS_PER_METER  # metres/s
        energy = float(np.sum(action ** 2))
        upright_penalty = abs(torso_angle)

        reward = (
            w["forward_velocity"] * forward_vel
            - w["energy_penalty"] * energy
            - w["upright_penalty"] * upright_penalty
            + w["alive_bonus"]
        )

        done = False
        fallen = self.creature.is_fallen(self._ground_y)
        if fallen:
            reward -= w["fall_penalty"]
            done = True

        return float(reward), done

    def _ensure_renderer(self) -> None:
        if self._renderer is None:
            from environment.renderer import Renderer
            self._renderer = Renderer(width=1000, height=600, title=f"RL — {self.genome.name}")
            self._renderer.ground_y = self._ground_y
            self._renderer.init()

    def _teardown(self) -> None:
        """Clear physics objects."""
        if self.space is not None:
            # Remove all constraints, shapes, bodies
            for c in list(self.space.constraints):
                self.space.remove(c)
            for s in list(self.space.shapes):
                self.space.remove(s)
            for b in list(self.space.bodies):
                if b is not self.space.static_body:
                    self.space.remove(b)
            self.space = None
        self.creature = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
