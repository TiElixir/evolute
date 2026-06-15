"""
creature/builder.py

Builds a live Pymunk physics body from a validated Genome.

The builder:
  - Walks the bone tree recursively from root.
  - Places each bone's body at the correct world position based on
    parent attach points + accumulated rotation.
  - Creates PivotJoint + RotaryLimitJoint (+ SimpleMotor if motorized)
    for each joint in the genome.
  - Returns a Creature object with named references to all physics objects
    and helper methods for RL (get_observation, apply_action, is_fallen).

Physics units:
  - Positions are in Pymunk pixels (scaled by PIXELS_PER_METER).
  - Genome lengths/widths are specified in "metres" and scaled here.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pymunk

from creature.morphology import (
    Bone,
    Genome,
    Joint,
    build_bone_tree,
    get_bone_depth_order,
    get_leaf_bones,
    get_root_bone,
)

# Scale factor: 1 genome-metre = PIXELS_PER_METER pymunk units
PIXELS_PER_METER: float = 100.0

# Fall detection: torso angle beyond this threshold (radians) = fallen
FALL_ANGLE_THRESHOLD: float = math.radians(75)

# Fall detection: torso COM below this height above ground = fallen
FALL_HEIGHT_THRESHOLD: float = 30.0   # pixels


def _scale(v: float) -> float:
    """Convert genome metres to pymunk pixels."""
    return v * PIXELS_PER_METER


def _make_box_vertices(length: float, width: float) -> List[Tuple[float, float]]:
    """
    Return vertices of a box shape centred at local origin.
    The bone's "long axis" is along Y (vertical when angle=0).
    length = total height, width = total width.
    """
    hw = _scale(width) / 2.0
    hl = _scale(length) / 2.0
    return [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]


# ---------------------------------------------------------------------------
# Creature — container for all physics objects
# ---------------------------------------------------------------------------


class Creature:
    """Holds all pymunk objects for a single creature instance.

    Attributes:
        genome: the source Genome.
        bodies: dict bone_id -> pymunk.Body
        shapes: dict bone_id -> pymunk.Shape
        joints: dict joint_id -> list of pymunk.Constraint
        motors: dict joint_id -> pymunk.SimpleMotor (only motorized joints)
        foot_ids: list of bone ids considered feet (leaf bones)
        bone_order: depth-first order of bone ids (for obs construction)
        space: the pymunk.Space this creature lives in (set by build_creature)
    """

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self.bodies: Dict[str, pymunk.Body] = {}
        self.shapes: Dict[str, pymunk.Shape] = {}
        self.constraints: Dict[str, List[pymunk.Constraint]] = {}  # joint_id -> constraints
        self.motors: Dict[str, pymunk.SimpleMotor] = {}
        self.foot_ids: List[str] = []
        self.bone_order: List[str] = []
        self.space: Optional[pymunk.Space] = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> np.ndarray:
        """Build the flat observation vector for RL.

        Structure per bone (in bone_order):
          [rel_x, rel_y, rel_angle, vel_x, vel_y, ang_vel]   (6 per bone)
        Global:
          [torso_height, torso_angle, torso_vx, torso_vy]     (4)
        Per motorized joint:
          [current_joint_angle_normalized]                      (1 per)
        Per foot bone:
          [contact_flag 0/1]                                    (1 per)
        """
        torso_body = self.bodies[self.genome.bones[0].id]
        torso_pos = torso_body.position
        torso_angle = torso_body.angle
        cos_t = math.cos(-torso_angle)
        sin_t = math.sin(-torso_angle)

        obs_parts: List[float] = []

        # Per-bone features
        for bid in self.bone_order:
            body = self.bodies[bid]
            # Position relative to torso, rotated into torso's frame
            dx = body.position.x - torso_pos.x
            dy = body.position.y - torso_pos.y
            rel_x = cos_t * dx - sin_t * dy
            rel_y = sin_t * dx + cos_t * dy
            rel_angle = body.angle - torso_angle
            # Wrap to [-pi, pi]
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
            vx, vy = body.velocity
            ang_vel = body.angular_velocity
            obs_parts += [rel_x / PIXELS_PER_METER,
                          rel_y / PIXELS_PER_METER,
                          rel_angle,
                          vx / PIXELS_PER_METER,
                          vy / PIXELS_PER_METER,
                          ang_vel]

        # Global torso features
        ground_y = self.space.static_body.position.y if self.space else 0.0
        torso_height = torso_pos.y - ground_y  # may be negative if fallen
        tvx, tvy = torso_body.velocity
        obs_parts += [
            torso_height / PIXELS_PER_METER,
            torso_angle,
            tvx / PIXELS_PER_METER,
            tvy / PIXELS_PER_METER,
        ]

        # Motorized joint angles (normalized to [-1, 1] within limit range)
        for jnt in self.genome.get_motorized_joints():
            ba = self.bodies[jnt.bone_a]
            bb = self.bodies[jnt.bone_b]
            rel = bb.angle - ba.angle
            rel = (rel + math.pi) % (2 * math.pi) - math.pi
            lo = math.radians(jnt.angle_limit_deg[0])
            hi = math.radians(jnt.angle_limit_deg[1])
            mid = (lo + hi) / 2.0
            half_range = max((hi - lo) / 2.0, 1e-6)
            normalized = (rel - mid) / half_range
            obs_parts.append(float(np.clip(normalized, -1.0, 1.0)))

        # Foot contact flags — set externally by env step via set_foot_contact
        obs_parts += [float(self._foot_contacts.get(fid, 0)) for fid in self.foot_ids]

        return np.array(obs_parts, dtype=np.float32)

    def reset_foot_contacts(self) -> None:
        self._foot_contacts: Dict[str, int] = {fid: 0 for fid in self.foot_ids}

    def set_foot_contact(self, bone_id: str, in_contact: bool) -> None:
        self._foot_contacts[bone_id] = 1 if in_contact else 0

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def apply_action(self, action: np.ndarray) -> None:
        """Apply continuous actions in [-1, 1] to motorized joints.

        action is expected to be of length == len(motorized_joints),
        ordered as genome.joints (filtered to is_motorized).
        Each value is scaled to the joint's max_motor_torque / a reference rate.
        """
        motorized = self.genome.get_motorized_joints()
        assert len(action) == len(motorized), (
            f"Action dim mismatch: got {len(action)}, expected {len(motorized)}"
        )
        for i, jnt in enumerate(motorized):
            motor = self.motors.get(jnt.id)
            if motor is None:
                continue
            # Scale action to angular rate; torque is set separately via max_force
            max_rate = 10.0  # rad/s max angular rate
            motor.rate = float(action[i]) * max_rate

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_torso_position(self) -> Tuple[float, float]:
        body = self.bodies[self.genome.bones[0].id]
        return (body.position.x, body.position.y)

    def get_torso_angle(self) -> float:
        return self.bodies[self.genome.bones[0].id].angle

    def get_torso_velocity(self) -> Tuple[float, float]:
        v = self.bodies[self.genome.bones[0].id].velocity
        return (v.x, v.y)

    def is_fallen(self, ground_y: float = 0.0) -> bool:
        torso = self.bodies[self.genome.bones[0].id]
        if abs(torso.angle) > FALL_ANGLE_THRESHOLD:
            return True
        if torso.position.y - ground_y < FALL_HEIGHT_THRESHOLD:
            return True
        return False

    @property
    def observation_dim(self) -> int:
        n_bones = len(self.genome.bones)
        n_motorized = len(self.genome.get_motorized_joints())
        n_feet = len(self.foot_ids)
        return n_bones * 6 + 4 + n_motorized + n_feet

    @property
    def action_dim(self) -> int:
        return len(self.genome.get_motorized_joints())


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_creature(
    space: pymunk.Space,
    genome: Genome,
    position: Tuple[float, float] = (200.0, 400.0),
) -> Creature:
    """Build a Pymunk physics body from a Genome and add it to `space`.

    Args:
        space: The pymunk.Space to add bodies/shapes/constraints to.
        genome: A validated Genome describing the creature.
        position: World-space (x, y) position for the root bone's COM in pixels.

    Returns:
        A Creature instance with all physics objects populated.
    """
    creature = Creature(genome)
    creature.space = space
    creature._foot_contacts = {}

    children = build_bone_tree(genome)
    root = get_root_bone(genome)
    creature.bone_order = get_bone_depth_order(genome)
    creature.foot_ids = get_leaf_bones(genome)
    creature.reset_foot_contacts()

    # Map bone_id -> world position of its attachment point on the parent
    # (i.e. where the joint connecting parent→this bone lives in world space).
    # For the root, this is `position`.
    world_attach: Dict[str, Tuple[float, float]] = {root.id: position}
    # World angle accumulated from root
    world_angle: Dict[str, float] = {root.id: 0.0}

    # BFS: process bones in depth order so parents are always built first
    queue: List[str] = [root.id]
    while queue:
        bid = queue.pop(0)
        bone = genome.get_bone_by_id(bid)
        attach_world = world_attach[bid]
        angle = world_angle[bid]

        # The bone's COM is at the attachment point for now (simplified —
        # we treat the local anchor as (0, half_length) in parent frame).
        # For the root, COM = position.
        com_x = attach_world[0]
        com_y = attach_world[1]

        body, shape = _create_bone_body(bone, (com_x, com_y), angle)
        space.add(body, shape)
        creature.bodies[bid] = body
        creature.shapes[bid] = shape

        # Pre-compute world attach points for children
        for child_id in children[bid]:
            child_bone = genome.get_bone_by_id(child_id)
            # child_bone.attach_point is the offset in the *parent's* local frame
            # (in metres), pointing from parent COM to the child's COM.
            ap = child_bone.attach_point
            # Rotate by parent's world angle
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            offset_x = cos_a * _scale(ap[0]) - sin_a * _scale(ap[1])
            offset_y = sin_a * _scale(ap[0]) + cos_a * _scale(ap[1])
            child_world_x = com_x + offset_x
            child_world_y = com_y + offset_y
            world_attach[child_id] = (child_world_x, child_world_y)
            world_angle[child_id] = angle  # child starts at same angle

            queue.append(child_id)

    # Create joints/constraints
    for jnt in genome.joints:
        constraints = _create_joint_constraints(jnt, creature)
        for c in constraints:
            space.add(c)
        creature.constraints[jnt.id] = constraints
        # Extract motor if present
        for c in constraints:
            if isinstance(c, pymunk.SimpleMotor):
                creature.motors[jnt.id] = c

    return creature


def _create_bone_body(
    bone: Bone,
    world_pos: Tuple[float, float],
    angle: float,
) -> Tuple[pymunk.Body, pymunk.Shape]:
    """Create a pymunk Body + Shape for a single bone."""
    vertices = _make_box_vertices(bone.length, bone.width)

    # Compute mass and moment from density and area
    area = _scale(bone.length) * _scale(bone.width)
    mass = bone.density * area * 0.0001  # scale down to reasonable physics mass
    mass = max(mass, 0.1)               # minimum mass

    moment = pymunk.moment_for_poly(mass, vertices)
    body = pymunk.Body(mass, moment)
    body.position = world_pos
    body.angle = angle

    shape = pymunk.Poly(body, vertices)
    shape.friction = 0.9
    shape.elasticity = 0.1

    return body, shape


def _create_joint_constraints(
    jnt: Joint,
    creature: Creature,
) -> List[pymunk.Constraint]:
    """Create the set of pymunk constraints for a joint."""
    body_a = creature.bodies[jnt.bone_a]
    body_b = creature.bodies[jnt.bone_b]

    # Anchor points in local space (metres -> pixels)
    anchor_a = (_scale(jnt.anchor_a[0]), _scale(jnt.anchor_a[1]))
    anchor_b = (_scale(jnt.anchor_b[0]), _scale(jnt.anchor_b[1]))

    constraints: List[pymunk.Constraint] = []

    # 1. PivotJoint — keeps the two bones connected at the anchor
    pivot = pymunk.PivotJoint(body_a, body_b, anchor_a, anchor_b)
    pivot.error_bias = pow(1.0 - 0.1, 60)   # fast correction
    pivot.max_bias = 1000.0
    constraints.append(pivot)

    # 2. RotaryLimitJoint — enforces angle limits
    lo = math.radians(jnt.angle_limit_deg[0])
    hi = math.radians(jnt.angle_limit_deg[1])
    rot_limit = pymunk.RotaryLimitJoint(body_a, body_b, lo, hi)
    rot_limit.max_force = jnt.max_motor_torque * PIXELS_PER_METER
    constraints.append(rot_limit)

    # 3. SimpleMotor — only for motorized joints
    if jnt.is_motorized:
        motor = pymunk.SimpleMotor(body_a, body_b, 0.0)  # rate=0 initially
        motor.max_force = jnt.max_motor_torque * PIXELS_PER_METER
        constraints.append(motor)

    # Disable collision between connected bones
    for c in constraints:
        c.collide_bodies = False

    return constraints
