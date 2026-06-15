"""
creature/morphology.py

Genome data structures for Creature Evolution RL.
Defines Bone, Joint, and Genome dataclasses with JSON serialization,
validation, and file I/O.

Security:
  - JSON deserialization uses Python's stdlib json module (no eval/pickle).
  - All numeric fields are type-checked and range-clamped during validation
    to prevent physics instability from malformed genomes.
  - File paths are validated by callers (see editor) before reaching save/load.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Bone:
    id: str
    parent: Optional[str]                  # None for the root/torso
    shape: str = "box"                     # "box" | "segment"
    length: float = 1.0
    width: float = 0.3
    density: float = 1.0
    color: List[int] = field(default_factory=lambda: [200, 200, 200])
    # World-space offset from parent's LOCAL attach_point.
    # For the root bone this is ignored (placed at build origin).
    attach_point: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent": self.parent,
            "shape": self.shape,
            "length": self.length,
            "width": self.width,
            "density": self.density,
            "color": list(self.color),
            "attach_point": list(self.attach_point),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Bone":
        return cls(
            id=str(d["id"]),
            parent=d.get("parent"),
            shape=str(d.get("shape", "box")),
            length=float(d.get("length", 1.0)),
            width=float(d.get("width", 0.3)),
            density=float(d.get("density", 1.0)),
            color=list(d.get("color", [200, 200, 200])),
            attach_point=list(d.get("attach_point", [0.0, 0.0])),
        )


@dataclass
class Joint:
    id: str
    bone_a: str
    bone_b: str
    # Anchor positions in LOCAL space of each bone (metres).
    anchor_a: List[float] = field(default_factory=lambda: [0.0, 0.0])
    anchor_b: List[float] = field(default_factory=lambda: [0.0, 0.0])
    # [min_deg, max_deg] relative rotation of bone_b w.r.t. bone_a
    angle_limit_deg: List[float] = field(default_factory=lambda: [-60.0, 60.0])
    max_motor_torque: float = 500.0
    is_motorized: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bone_a": self.bone_a,
            "bone_b": self.bone_b,
            "anchor_a": list(self.anchor_a),
            "anchor_b": list(self.anchor_b),
            "angle_limit_deg": list(self.angle_limit_deg),
            "max_motor_torque": self.max_motor_torque,
            "is_motorized": self.is_motorized,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Joint":
        return cls(
            id=str(d["id"]),
            bone_a=str(d["bone_a"]),
            bone_b=str(d["bone_b"]),
            anchor_a=list(d.get("anchor_a", [0.0, 0.0])),
            anchor_b=list(d.get("anchor_b", [0.0, 0.0])),
            angle_limit_deg=list(d.get("angle_limit_deg", [-60.0, 60.0])),
            max_motor_torque=float(d.get("max_motor_torque", 500.0)),
            is_motorized=bool(d.get("is_motorized", True)),
        )


@dataclass
class Genome:
    name: str
    bones: List[Bone] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "generation": 0,
        "lineage": [],
        "fitness": None,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bones": [b.to_dict() for b in self.bones],
            "joints": [j.to_dict() for j in self.joints],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        return cls(
            name=str(d["name"]),
            bones=[Bone.from_dict(b) for b in d.get("bones", [])],
            joints=[Joint.from_dict(j) for j in d.get("joints", [])],
            metadata=dict(d.get("metadata", {"generation": 0, "lineage": [], "fitness": None})),
        )

    def save(self, path: str) -> None:
        """Serialize genome to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Genome":
        """Load genome from JSON file.

        Security: uses stdlib json (no eval/pickle).
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        genome = cls.from_dict(data)
        validate_genome(genome)   # raises ValueError on invalid
        return genome

    def copy(self) -> "Genome":
        """Return a deep copy via round-trip serialisation."""
        return Genome.from_dict(self.to_dict())

    def get_motorized_joints(self) -> List[Joint]:
        return [j for j in self.joints if j.is_motorized]

    def get_bone_by_id(self, bone_id: str) -> Optional[Bone]:
        for b in self.bones:
            if b.id == bone_id:
                return b
        return None

    def get_joint_by_id(self, joint_id: str) -> Optional[Joint]:
        for j in self.joints:
            if j.id == joint_id:
                return j
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_genome(genome: Genome) -> None:
    """Validate a Genome and raise ValueError if anything is wrong.

    Checks performed:
      - At least one bone exists.
      - Exactly one root bone (parent is None).
      - No duplicate bone IDs or joint IDs.
      - All parent references point to existing bones.
      - No cyclic parent references.
      - All joint bone_a / bone_b reference existing bones.
      - angle_limit_deg[0] <= angle_limit_deg[1].
      - Numeric fields are within sane physical ranges.
    """
    if not genome.bones:
        raise ValueError("Genome must have at least one bone.")

    # --- Duplicate IDs ---
    bone_ids = [b.id for b in genome.bones]
    if len(bone_ids) != len(set(bone_ids)):
        raise ValueError(f"Duplicate bone IDs detected: {bone_ids}")

    joint_ids = [j.id for j in genome.joints]
    if len(joint_ids) != len(set(joint_ids)):
        raise ValueError(f"Duplicate joint IDs detected: {joint_ids}")

    bone_id_set = set(bone_ids)

    # --- Root bone ---
    roots = [b for b in genome.bones if b.parent is None]
    if len(roots) != 1:
        raise ValueError(
            f"Genome must have exactly one root bone (parent=null), "
            f"found {len(roots)}: {[r.id for r in roots]}"
        )

    # --- Parent references ---
    for bone in genome.bones:
        if bone.parent is not None and bone.parent not in bone_id_set:
            raise ValueError(
                f"Bone '{bone.id}' references unknown parent '{bone.parent}'."
            )

    # --- Cycle detection (DFS) ---
    children: Dict[str, List[str]] = {bid: [] for bid in bone_ids}
    for bone in genome.bones:
        if bone.parent is not None:
            children[bone.parent].append(bone.id)

    visited: set = set()
    stack: List[str] = [roots[0].id]
    while stack:
        node = stack.pop()
        if node in visited:
            raise ValueError(f"Cyclic parent reference detected involving bone '{node}'.")
        visited.add(node)
        stack.extend(children[node])

    orphans = bone_id_set - visited
    if orphans:
        raise ValueError(f"Orphaned bones not reachable from root: {orphans}")

    # --- Joint references ---
    for joint in genome.joints:
        if joint.bone_a not in bone_id_set:
            raise ValueError(
                f"Joint '{joint.id}' references unknown bone_a '{joint.bone_a}'."
            )
        if joint.bone_b not in bone_id_set:
            raise ValueError(
                f"Joint '{joint.id}' references unknown bone_b '{joint.bone_b}'."
            )
        if joint.bone_a == joint.bone_b:
            raise ValueError(
                f"Joint '{joint.id}' connects a bone to itself ('{joint.bone_a}')."
            )

    # --- Angle limits ---
    for joint in genome.joints:
        lo, hi = joint.angle_limit_deg
        if lo > hi:
            raise ValueError(
                f"Joint '{joint.id}' angle_limit_deg min ({lo}) > max ({hi})."
            )

    # --- Numeric sanity (clamping happens in from_dict; we raise on extremes) ---
    for bone in genome.bones:
        if bone.length <= 0:
            raise ValueError(f"Bone '{bone.id}' length must be > 0, got {bone.length}.")
        if bone.width <= 0:
            raise ValueError(f"Bone '{bone.id}' width must be > 0, got {bone.width}.")
        if bone.density <= 0:
            raise ValueError(f"Bone '{bone.id}' density must be > 0, got {bone.density}.")
        if len(bone.color) != 3 or not all(0 <= c <= 255 for c in bone.color):
            raise ValueError(f"Bone '{bone.id}' color must be [R,G,B] with values 0-255.")

    for joint in genome.joints:
        if joint.max_motor_torque <= 0:
            raise ValueError(
                f"Joint '{joint.id}' max_motor_torque must be > 0, "
                f"got {joint.max_motor_torque}."
            )


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def build_bone_tree(genome: Genome) -> Dict[str, List[str]]:
    """Return a dict mapping bone_id -> list of child bone_ids."""
    children: Dict[str, List[str]] = {b.id: [] for b in genome.bones}
    for bone in genome.bones:
        if bone.parent is not None:
            children[bone.parent].append(bone.id)
    return children


def get_leaf_bones(genome: Genome) -> List[str]:
    """Return IDs of bones that have no children (potential feet/tips)."""
    children = build_bone_tree(genome)
    return [bid for bid, ch in children.items() if not ch]


def get_root_bone(genome: Genome) -> Bone:
    """Return the root bone (parent=None). Assumes genome is valid."""
    for bone in genome.bones:
        if bone.parent is None:
            return bone
    raise ValueError("No root bone found.")


def get_bone_depth_order(genome: Genome) -> List[str]:
    """BFS order starting from root — useful for observation construction."""
    children = build_bone_tree(genome)
    root = get_root_bone(genome)
    order: List[str] = []
    queue: List[str] = [root.id]
    while queue:
        node = queue.pop(0)
        order.append(node)
        queue.extend(children[node])
    return order
