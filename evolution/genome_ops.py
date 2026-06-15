"""
evolution/genome_ops.py

Mutation and crossover operators for Genome evolution.

Each operator:
  1. Works on a deep copy of the genome (never mutates in-place).
  2. Calls validate_genome() and retries / returns None if the result is invalid.
  3. Is safe to call multiple times without side effects.
"""

from __future__ import annotations

import copy
import math
import random
import string
from typing import Optional

from creature.morphology import (
    Bone,
    Genome,
    Joint,
    build_bone_tree,
    get_leaf_bones,
    get_root_bone,
    validate_genome,
)

MAX_RETRIES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_id(prefix: str = "bone", length: int = 4) -> str:
    chars = string.ascii_lowercase + string.digits
    suffix = "".join(random.choices(chars, k=length))
    return f"{prefix}_{suffix}"


def _copy(genome: Genome) -> Genome:
    return genome.copy()


def _try_validate(genome: Genome) -> Optional[Genome]:
    try:
        validate_genome(genome)
        return genome
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------


def mutate_bone_size(
    genome: Genome,
    scale_range: tuple = (0.8, 1.2),
) -> Optional[Genome]:
    """Scale the length and/or width of a random bone."""
    g = _copy(genome)
    if not g.bones:
        return None
    bone = random.choice(g.bones)
    scale_l = random.uniform(*scale_range)
    scale_w = random.uniform(*scale_range)
    bone.length = max(0.05, bone.length * scale_l)
    bone.width  = max(0.03, bone.width  * scale_w)
    return _try_validate(g)


def mutate_joint_params(genome: Genome) -> Optional[Genome]:
    """Perturb angle limits and/or max_motor_torque of a random joint."""
    g = _copy(genome)
    if not g.joints:
        return None
    jnt = random.choice(g.joints)
    # Perturb angle limits by ±15 degrees
    delta = random.uniform(-15, 15)
    jnt.angle_limit_deg[0] = max(-170, jnt.angle_limit_deg[0] + delta)
    jnt.angle_limit_deg[1] = min(170,  jnt.angle_limit_deg[1] + delta)
    if jnt.angle_limit_deg[0] > jnt.angle_limit_deg[1]:
        jnt.angle_limit_deg[0], jnt.angle_limit_deg[1] = (
            jnt.angle_limit_deg[1], jnt.angle_limit_deg[0]
        )
    # Perturb torque by ±20%
    factor = random.uniform(0.8, 1.2)
    jnt.max_motor_torque = max(100.0, jnt.max_motor_torque * factor)
    return _try_validate(g)


def add_random_bone(genome: Genome) -> Optional[Genome]:
    """Pick a random existing bone and attach a new child bone + joint."""
    for _ in range(MAX_RETRIES):
        g = _copy(genome)
        parent = random.choice(g.bones)

        new_id = _random_id("bone")
        while g.get_bone_by_id(new_id):
            new_id = _random_id("bone")

        # Random child offset
        length = random.uniform(0.2, 0.7)
        width  = random.uniform(0.08, 0.25)
        ap_x   = random.uniform(-parent.width / 2, parent.width / 2)
        ap_y   = random.uniform(-parent.length / 2, parent.length / 2)

        new_bone = Bone(
            id=new_id,
            parent=parent.id,
            shape="box",
            length=length,
            width=width,
            density=random.uniform(0.8, 1.3),
            color=[
                random.randint(60, 220),
                random.randint(60, 220),
                random.randint(60, 220),
            ],
            attach_point=[round(ap_x, 3), round(ap_y, 3)],
        )
        g.bones.append(new_bone)

        joint_id = _random_id("j")
        while g.get_joint_by_id(joint_id):
            joint_id = _random_id("j")

        new_joint = Joint(
            id=joint_id,
            bone_a=parent.id,
            bone_b=new_id,
            anchor_a=[round(ap_x, 3), round(ap_y, 3)],
            anchor_b=[0.0, round(length / 2, 3)],
            angle_limit_deg=[
                random.uniform(-80, -20),
                random.uniform(20, 80),
            ],
            max_motor_torque=random.uniform(300, 900),
            is_motorized=True,
        )
        g.joints.append(new_joint)

        result = _try_validate(g)
        if result:
            return result
    return None


def remove_random_leaf_bone(genome: Genome) -> Optional[Genome]:
    """Remove a random leaf bone (no children) and its connecting joint.
    Never removes the root/torso.
    """
    for _ in range(MAX_RETRIES):
        g = _copy(genome)
        leaves = get_leaf_bones(g)
        root = get_root_bone(g)
        # Exclude root from removal
        candidates = [lid for lid in leaves if lid != root.id]
        if not candidates:
            return None

        target_id = random.choice(candidates)
        # Remove bone
        g.bones = [b for b in g.bones if b.id != target_id]
        # Remove connecting joints (any joint referencing this bone)
        g.joints = [j for j in g.joints if j.bone_a != target_id and j.bone_b != target_id]

        result = _try_validate(g)
        if result:
            return result
    return None


def toggle_motor(genome: Genome) -> Optional[Genome]:
    """Flip is_motorized on a random joint."""
    g = _copy(genome)
    if not g.joints:
        return None
    jnt = random.choice(g.joints)
    jnt.is_motorized = not jnt.is_motorized
    return _try_validate(g)


def mutate_bone_color(genome: Genome) -> Optional[Genome]:
    """Randomize the color of a random bone (cosmetic only)."""
    g = _copy(genome)
    if not g.bones:
        return None
    bone = random.choice(g.bones)
    bone.color = [random.randint(60, 220) for _ in range(3)]
    return _try_validate(g)


def mutate_density(genome: Genome) -> Optional[Genome]:
    """Perturb density of a random bone."""
    g = _copy(genome)
    if not g.bones:
        return None
    bone = random.choice(g.bones)
    bone.density = max(0.3, bone.density * random.uniform(0.8, 1.2))
    return _try_validate(g)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(genome_a: Genome, genome_b: Genome) -> Optional[Genome]:
    """Swap a random limb subtree from genome_b into genome_a.

    Strategy:
      1. Pick a non-root bone from genome_b as the "donor root".
      2. Collect the donor subtree (donor root + all descendants).
      3. Build a fresh genome_a copy.
      4. Pick a random non-root attachment point in genome_a.
      5. Re-root the donor subtree under that attachment bone,
         renaming all IDs to avoid conflicts.
      6. Validate; retry if invalid.
    """
    for _ in range(MAX_RETRIES):
        g_a = _copy(genome_a)
        g_b = _copy(genome_b)

        root_b = get_root_bone(g_b)
        non_root_b = [b for b in g_b.bones if b.id != root_b.id]
        if not non_root_b:
            return None

        donor_root = random.choice(non_root_b)
        donor_subtree_ids = _collect_subtree(g_b, donor_root.id)
        donor_bones  = [b for b in g_b.bones  if b.id in donor_subtree_ids]
        donor_joints = [j for j in g_b.joints if j.bone_a in donor_subtree_ids
                        or j.bone_b in donor_subtree_ids]

        # Also grab the joint connecting donor_root to its parent
        parent_joint = next(
            (j for j in g_b.joints if j.bone_b == donor_root.id), None
        )

        if parent_joint is None:
            continue

        # Target attachment in genome_a
        non_root_a = [b for b in g_a.bones if b.parent is not None]
        if not non_root_a:
            attach_bone = get_root_bone(g_a)
        else:
            attach_bone = random.choice(non_root_a + [get_root_bone(g_a)])

        # Rename all donor IDs to avoid conflicts with g_a
        existing_ids = {b.id for b in g_a.bones} | {j.id for j in g_a.joints}
        id_map: dict = {}
        for bid in donor_subtree_ids:
            new_id = bid
            while new_id in existing_ids:
                new_id = _random_id(bid.split("_")[0])
            id_map[bid] = new_id
            existing_ids.add(new_id)
        pj_new_id = parent_joint.id
        while pj_new_id in existing_ids:
            pj_new_id = _random_id("j")
        id_map[parent_joint.id] = pj_new_id

        # Remap bone IDs
        new_bones = []
        for b in donor_bones:
            nb = copy.deepcopy(b)
            nb.id     = id_map[b.id]
            nb.parent = attach_bone.id if b.id == donor_root.id else id_map.get(b.parent, b.parent)
            new_bones.append(nb)

        # Remap joint IDs
        def _remap_jid(jid):
            return id_map.get(jid, jid)

        new_joints = []
        for j in donor_joints:
            nj = copy.deepcopy(j)
            nj.id     = id_map.get(j.id, j.id)
            nj.bone_a = _remap_jid(j.bone_a)
            nj.bone_b = _remap_jid(j.bone_b)
            new_joints.append(nj)

        # Connecting joint
        conn_joint = copy.deepcopy(parent_joint)
        conn_joint.id     = pj_new_id
        conn_joint.bone_a = attach_bone.id
        conn_joint.bone_b = id_map[donor_root.id]

        g_a.bones  += new_bones
        g_a.joints += new_joints + [conn_joint]
        g_a.metadata["lineage"] = [genome_a.name, genome_b.name]

        result = _try_validate(g_a)
        if result:
            return result

    return None


def _collect_subtree(genome: Genome, root_id: str) -> set:
    """Return the set of bone IDs in the subtree rooted at root_id."""
    children = build_bone_tree(genome)
    result = set()
    stack = [root_id]
    while stack:
        node = stack.pop()
        result.add(node)
        stack.extend(children.get(node, []))
    return result


# ---------------------------------------------------------------------------
# Mirror tool
# ---------------------------------------------------------------------------


def mirror_limb(genome: Genome, bone_id: str) -> Optional[Genome]:
    """Mirror a limb subtree across the torso's vertical (Y) axis.

    Creates a mirrored copy of the subtree rooted at bone_id,
    appending '_r' to all IDs (or '_l' if id ends with '_r').
    """
    g = _copy(genome)
    subtree_ids = _collect_subtree(g, bone_id)
    existing_ids = {b.id for b in g.bones} | {j.id for j in g.joints}

    def _mirror_id(bid: str) -> str:
        if bid.endswith("_l"):
            candidate = bid[:-2] + "_r"
        elif bid.endswith("_r"):
            candidate = bid[:-2] + "_l"
        else:
            candidate = bid + "_m"
        while candidate in existing_ids:
            candidate += "x"
        existing_ids.add(candidate)
        return candidate

    id_map = {bid: _mirror_id(bid) for bid in subtree_ids}

    subtree_bones  = [b for b in g.bones  if b.id in subtree_ids]
    subtree_joints = [j for j in g.joints if j.bone_a in subtree_ids
                      and j.bone_b in subtree_ids]
    connect_joint  = next(
        (j for j in g.joints if j.bone_b == bone_id), None
    )

    new_bones = []
    for b in subtree_bones:
        nb = copy.deepcopy(b)
        nb.id = id_map[b.id]
        nb.parent = id_map.get(b.parent, b.parent) if b.parent in subtree_ids else b.parent
        # Mirror x attach point
        nb.attach_point = [-nb.attach_point[0], nb.attach_point[1]]
        new_bones.append(nb)

    new_joints = []
    for j in subtree_joints:
        nj = copy.deepcopy(j)
        nj.id     = _mirror_id(j.id)
        nj.bone_a = id_map.get(j.bone_a, j.bone_a)
        nj.bone_b = id_map.get(j.bone_b, j.bone_b)
        nj.anchor_a = [-nj.anchor_a[0], nj.anchor_a[1]]
        nj.anchor_b = [-nj.anchor_b[0], nj.anchor_b[1]]
        new_joints.append(nj)

    if connect_joint:
        ncj = copy.deepcopy(connect_joint)
        ncj.id     = _mirror_id(connect_joint.id)
        ncj.bone_b = id_map[bone_id]
        ncj.anchor_a = [-ncj.anchor_a[0], ncj.anchor_a[1]]
        ncj.anchor_b = [-ncj.anchor_b[0], ncj.anchor_b[1]]
        new_joints.append(ncj)

    g.bones  += new_bones
    g.joints += new_joints
    return _try_validate(g)
