"""
editor/creature_editor.py

Interactive Pygame GUI editor for creature genomes.

Features:
  - Canvas showing current creature with grid background
  - Click to select a bone or joint
  - Side panel with editable fields (length, width, density, color, angle limits,
    max_motor_torque, is_motorized)
  - Add Bone: click parent → drag to define direction/length
  - Delete Bone: remove selected bone subtree (with in-canvas confirmation)
  - Test Drive: toggle physics on/off to preview creature behaviour
  - Save / Load: text-input filename dialog with path traversal protection
  - Mirror: mirror selected limb across torso Y axis

Security:
  - All user-provided filenames are sanitized with os.path.basename()
    and joined against a hardcoded SAFE_SAVE_DIR.
  - Paths are resolved with os.path.realpath() and validated to be
    inside SAFE_SAVE_DIR before any file I/O.
  - No eval(), no pickle, no shell commands.

Run:
    python -m editor.creature_editor [--genome creature/presets/biped.json]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pygame
import pymunk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creature.builder import Creature, build_creature, PIXELS_PER_METER
from creature.morphology import Bone, Genome, Joint, build_bone_tree, get_root_bone, validate_genome
from evolution.genome_ops import mirror_limb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_W, WINDOW_H = 1280, 720
PANEL_W = 300
CANVAS_W = WINDOW_W - PANEL_W

# Hardcoded safe directory for genome saves (traversal protection)
SAFE_SAVE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "creature", "saved")
)

# Colours
BG_CANVAS     = (14, 16, 26)
BG_PANEL      = (22, 24, 38)
GRID_CLR      = (28, 30, 46)
GROUND_CLR    = (40, 45, 65)
GROUND_LINE   = (70, 80, 110)
BONE_OUTLINE  = (0, 0, 0)
SEL_OUTLINE   = (255, 220, 0)
JOINT_CLR     = (255, 190, 40)
JOINT_MOTOR   = (100, 220, 100)
JOINT_PASSIVE = (150, 150, 200)
TEXT_CLR      = (210, 215, 240)
TEXT_DIM      = (120, 125, 160)
ACCENT        = (80, 160, 255)
DANGER        = (220, 60, 60)
SUCCESS       = (60, 200, 100)
INPUT_BG      = (30, 32, 50)
INPUT_ACTIVE  = (40, 44, 70)
SLIDER_BG     = (35, 38, 58)
SLIDER_FG     = (80, 140, 240)
BTN_NORMAL    = (38, 42, 68)
BTN_HOVER     = (55, 60, 95)
BTN_ACTIVE    = (80, 140, 240)
BTN_DANGER    = (120, 30, 30)

GROUND_Y       = 100   # canvas-space (before camera transform)
CAMERA_START_X = -200.0


# ---------------------------------------------------------------------------
# Minimal custom UI widgets
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect: pygame.Rect, label: str, color=BTN_NORMAL, danger=False):
        self.rect  = rect
        self.label = label
        self.color = DANGER if danger else color
        self.hover_color = (min(self.color[0]+20, 255), min(self.color[1]+20, 255), min(self.color[2]+20, 255))
        self._hovered = False
        self._pressed = False

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        col = self.hover_color if self._hovered else self.color
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, (60, 65, 100), self.rect, 1, border_radius=6)
        lbl = font.render(self.label, True, TEXT_CLR)
        surf.blit(lbl, (self.rect.centerx - lbl.get_width() // 2,
                        self.rect.centery - lbl.get_height() // 2))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True on click."""
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class TextInput:
    def __init__(self, rect: pygame.Rect, value: str = "", max_len: int = 40):
        self.rect    = rect
        self.value   = value
        self.max_len = max_len
        self.active  = False
        self._cursor_timer = 0

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        bg = INPUT_ACTIVE if self.active else INPUT_BG
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, ACCENT if self.active else (50, 55, 80), self.rect, 1, border_radius=4)
        text_surf = font.render(self.value, True, TEXT_CLR)
        surf.blit(text_surf, (self.rect.x + 5, self.rect.centery - text_surf.get_height() // 2))
        if self.active:
            self._cursor_timer += 1
            if (self._cursor_timer // 30) % 2 == 0:
                cx = self.rect.x + 5 + text_surf.get_width() + 1
                pygame.draw.line(surf, TEXT_CLR, (cx, self.rect.y + 4), (cx, self.rect.bottom - 4))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
                return True
            elif event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.active = False
            elif len(self.value) < self.max_len and event.unicode.isprintable():
                self.value += event.unicode
                return True
        return False


class Slider:
    def __init__(self, rect: pygame.Rect, min_val: float, max_val: float, value: float):
        self.rect    = rect
        self.min_val = min_val
        self.max_val = max_val
        self.value   = value
        self._dragging = False

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        track = pygame.Rect(self.rect.x, self.rect.centery - 3, self.rect.width, 6)
        pygame.draw.rect(surf, SLIDER_BG, track, border_radius=3)
        t = (self.value - self.min_val) / max(self.max_val - self.min_val, 1e-6)
        fill_w = int(t * self.rect.width)
        if fill_w > 0:
            fill = pygame.Rect(self.rect.x, self.rect.centery - 3, fill_w, 6)
            pygame.draw.rect(surf, SLIDER_FG, fill, border_radius=3)
        knob_x = self.rect.x + fill_w
        pygame.draw.circle(surf, ACCENT, (knob_x, self.rect.centery), 8)
        lbl = font.render(f"{self.value:.2f}", True, TEXT_DIM)
        surf.blit(lbl, (self.rect.right - lbl.get_width(), self.rect.y - 18))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._dragging = True
                t = (event.pos[0] - self.rect.x) / max(self.rect.width, 1)
                t = max(0.0, min(1.0, t))
                self.value = self.min_val + t * (self.max_val - self.min_val)
                return True
        if event.type == pygame.MOUSEBUTTONUP:
            self._dragging = False
        if event.type == pygame.MOUSEMOTION and self._dragging:
            t = (event.pos[0] - self.rect.x) / max(self.rect.width, 1)
            t = max(0.0, min(1.0, t))
            self.value = self.min_val + t * (self.max_val - self.min_val)
            return True
        return False


class Checkbox:
    def __init__(self, rect: pygame.Rect, checked: bool = False):
        self.rect    = rect
        self.checked = checked

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, label: str) -> None:
        col = ACCENT if self.checked else SLIDER_BG
        pygame.draw.rect(surf, col, self.rect, border_radius=3)
        pygame.draw.rect(surf, (70, 80, 120), self.rect, 1, border_radius=3)
        if self.checked:
            pygame.draw.line(surf, TEXT_CLR,
                             (self.rect.x + 3, self.rect.centery),
                             (self.rect.centerx, self.rect.bottom - 3), 2)
            pygame.draw.line(surf, TEXT_CLR,
                             (self.rect.centerx, self.rect.bottom - 3),
                             (self.rect.right - 2, self.rect.y + 3), 2)
        lbl = font.render(label, True, TEXT_CLR)
        surf.blit(lbl, (self.rect.right + 8, self.rect.centery - lbl.get_height() // 2))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                return True
        return False


# ---------------------------------------------------------------------------
# Editor application
# ---------------------------------------------------------------------------

class CreatureEditor:
    """Full creature editor application."""

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self.space: Optional[pymunk.Space] = None
        self.creature: Optional[Creature] = None

        self.selected_bone_id: Optional[str] = None
        self.selected_joint_id: Optional[str] = None
        self._dragging_joint_id: Optional[str] = None

        # Physics test drive
        self.test_drive = False
        self._test_drive_physics_steps = 0

        # Camera
        self.camera_x = CAMERA_START_X

        # Add-bone drag state
        self._adding_bone = False
        self._add_parent_id: Optional[str] = None
        self._add_drag_start: Optional[Tuple[int, int]] = None
        self._add_drag_end: Optional[Tuple[int, int]] = None

        # Confirmation dialog
        self._confirm_dialog = False
        self._confirm_action: Optional[str] = None

        # File dialog
        self._file_dialog: Optional[str] = None  # "save" | "load"
        self._file_input_value: str = ""

        # Message banner
        self._message: str = ""
        self._message_timer: int = 0

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(f"Creature Editor — {genome.name}")
        self.clock = pygame.time.Clock()
        self.font_h  = pygame.font.SysFont("monospace", 15, bold=True)
        self.font    = pygame.font.SysFont("monospace", 13)
        self.font_sm = pygame.font.SysFont("monospace", 11)

        # Panel widgets (rebuilt when selection changes)
        self._panel_widgets: Dict[str, Any] = {}
        self._toolbar_buttons: List[Button] = []
        self._build_toolbar()

        # Rebuild creature physics in static (paused) mode
        self._rebuild_creature()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            self.clock.tick(60)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                self._handle_event(event)

            if self.test_drive and self.space:
                for _ in range(4):
                    self.space.step(1.0 / 60.0 / 4.0)
                self._test_drive_physics_steps += 1

            self._draw()
            pygame.display.flip()

    # ------------------------------------------------------------------
    # Build toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        btn_y  = 8
        btn_h  = 34
        btn_w  = 110
        margin = 8
        x = PANEL_W + margin
        labels = [
            ("Add Bone",    "add_bone",   False),
            ("Delete Bone", "delete_bone",True),
            ("Test Drive",  "test_drive", False),
            ("Mirror",      "mirror",     False),
            ("Save",        "save",       False),
            ("Load",        "load",       False),
        ]
        self._toolbar_buttons = []
        for label, tag, danger in labels:
            btn = Button(pygame.Rect(x, btn_y, btn_w, btn_h), label, danger=danger)
            btn._tag = tag
            self._toolbar_buttons.append(btn)
            x += btn_w + margin

    # ------------------------------------------------------------------
    # Creature rebuild
    # ------------------------------------------------------------------

    def _rebuild_creature(self) -> None:
        """Rebuild the static Pymunk space from current genome."""
        self.space = pymunk.Space()
        self.space.gravity = (0, -900)
        self.space.damping = 0.9

        ground = pymunk.Segment(self.space.static_body, (-5000, GROUND_Y), (50000, GROUND_Y), 5)
        ground.friction = 1.0
        self.space.add(ground)

        self.creature = build_creature(
            self.space, self.genome, position=(300, GROUND_Y + 300)
        )
        self.creature.space = self.space
        self._test_drive_physics_steps = 0

        # Rebuild panel widgets for current selection
        self._rebuild_panel_widgets()

    # ------------------------------------------------------------------
    # Panel widgets
    # ------------------------------------------------------------------

    def _rebuild_panel_widgets(self) -> None:
        """Build side-panel widgets based on current selection."""
        w: Dict[str, Any] = {}
        px, py = 10, 60
        fw = PANEL_W - 20

        if self.selected_bone_id:
            bone = self.genome.get_bone_by_id(self.selected_bone_id)
            if bone:
                w["length_lbl"]  = ("Length", py)
                w["length"]      = Slider(pygame.Rect(px, py+18, fw, 20), 0.1, 3.0, bone.length); py += 55
                w["width_lbl"]   = ("Width", py)
                w["width"]       = Slider(pygame.Rect(px, py+18, fw, 20), 0.05, 1.5, bone.width); py += 55
                w["density_lbl"] = ("Density", py)
                w["density"]     = Slider(pygame.Rect(px, py+18, fw, 20), 0.3, 3.0, bone.density); py += 55

                w["color_r_lbl"] = ("R", py)
                w["color_r"]     = Slider(pygame.Rect(px, py+18, fw, 20), 0, 255, bone.color[0]); py += 55
                w["color_g_lbl"] = ("G", py)
                w["color_g"]     = Slider(pygame.Rect(px, py+18, fw, 20), 0, 255, bone.color[1]); py += 55
                w["color_b_lbl"] = ("B", py)
                w["color_b"]     = Slider(pygame.Rect(px, py+18, fw, 20), 0, 255, bone.color[2]); py += 60

                # Find the joint connecting this bone
                joint = next((j for j in self.genome.joints if j.bone_b == self.selected_bone_id), None)
                if joint:
                    w["_active_joint"] = joint
                    w["ang_lo_lbl"] = ("Angle Min (deg)", py)
                    w["ang_lo"]     = Slider(pygame.Rect(px, py+18, fw, 20), -179, 0, joint.angle_limit_deg[0]); py += 55
                    w["ang_hi_lbl"] = ("Angle Max (deg)", py)
                    w["ang_hi"]     = Slider(pygame.Rect(px, py+18, fw, 20), 0, 179, joint.angle_limit_deg[1]); py += 55
                    w["torque_lbl"] = ("Max Torque", py)
                    w["torque"]     = Slider(pygame.Rect(px, py+18, fw, 20), 50, 2000, joint.max_motor_torque); py += 55
                    w["motorized"]  = Checkbox(pygame.Rect(px, py, 20, 20), joint.is_motorized); py += 40

        elif self.selected_joint_id:
            joint = self.genome.get_joint_by_id(self.selected_joint_id)
            if joint:
                w["_active_joint"] = joint
                w["ang_lo_lbl"] = ("Angle Min (deg)", py)
                w["ang_lo"]     = Slider(pygame.Rect(px, py+18, fw, 20), -179, 0, joint.angle_limit_deg[0]); py += 55
                w["ang_hi_lbl"] = ("Angle Max (deg)", py)
                w["ang_hi"]     = Slider(pygame.Rect(px, py+18, fw, 20), 0, 179, joint.angle_limit_deg[1]); py += 55
                w["torque_lbl"] = ("Max Torque", py)
                w["torque"]     = Slider(pygame.Rect(px, py+18, fw, 20), 50, 2000, joint.max_motor_torque); py += 55
                w["motorized"]  = Checkbox(pygame.Rect(px, py, 20, 20), joint.is_motorized); py += 40

        self._panel_widgets = w

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_event(self, event: pygame.event.Event) -> None:
        # File dialog takes priority
        if self._file_dialog:
            self._handle_file_dialog_event(event)
            return

        # Confirmation dialog
        if self._confirm_dialog:
            self._handle_confirm_event(event)
            return

        # Toolbar buttons
        for btn in self._toolbar_buttons:
            if btn.handle_event(event):
                self._handle_toolbar(btn._tag)
                return

        # Panel sliders / checkboxes
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                           pygame.MOUSEMOTION, pygame.KEYDOWN):
            changed = self._handle_panel_event(event)
            if changed:
                self._apply_panel_to_genome()
                self._rebuild_creature()
                return

        # Canvas mouse events
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            canvas_pos = event.pos
            if canvas_pos[0] > PANEL_W:
                if self._adding_bone:
                    self._start_add_bone_drag(canvas_pos)
                else:
                    self._try_select(canvas_pos)
                    if self.selected_joint_id:
                        self._dragging_joint_id = self.selected_joint_id

        if event.type == pygame.MOUSEMOTION:
            if self._adding_bone and self._add_drag_start:
                self._add_drag_end = event.pos
            elif self._dragging_joint_id:
                world_pos = self._screen_to_world(event.pos)
                joint = self.genome.get_joint_by_id(self._dragging_joint_id)
                if joint and self.creature and joint.bone_a in self.creature.bodies:
                    body_a = self.creature.bodies[joint.bone_a]
                    local_pt = body_a.world_to_local(world_pos)
                    ap_x = round(local_pt.x / PIXELS_PER_METER, 3)
                    ap_y = round(local_pt.y / PIXELS_PER_METER, 3)
                    joint.anchor_a = [ap_x, ap_y]
                    child_bone = self.genome.get_bone_by_id(joint.bone_b)
                    if child_bone:
                        child_bone.attach_point = [ap_x, ap_y]
                    self._rebuild_creature()

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._adding_bone and self._add_drag_start and self._add_drag_end:
                self._finish_add_bone_drag()
            if self._dragging_joint_id:
                self._dragging_joint_id = None

        # Camera scroll
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.camera_x -= 50
            if event.key == pygame.K_RIGHT:
                self.camera_x += 50

    def _handle_toolbar(self, tag: str) -> None:
        if tag == "add_bone":
            self._adding_bone = not self._adding_bone
            self._add_drag_start = None
            self._add_drag_end   = None
            self._set_message("Click a parent bone, then drag to set length/direction" if self._adding_bone else "Add Bone cancelled")
        elif tag == "delete_bone":
            if self.selected_bone_id:
                root = get_root_bone(self.genome)
                if self.selected_bone_id == root.id:
                    self._set_message("Cannot delete root bone!")
                else:
                    self._confirm_dialog = True
                    self._confirm_action = "delete_bone"
            else:
                self._set_message("Select a bone first")
        elif tag == "test_drive":
            self.test_drive = not self.test_drive
            if self.test_drive:
                self._set_message("Test Drive ON — physics running (motors disabled)")
            else:
                self._rebuild_creature()
                self._set_message("Test Drive OFF — creature reset")
        elif tag == "mirror":
            if self.selected_bone_id:
                root = get_root_bone(self.genome)
                if self.selected_bone_id == root.id:
                    self._set_message("Cannot mirror root bone")
                else:
                    result = mirror_limb(self.genome, self.selected_bone_id)
                    if result:
                        self.genome = result
                        self._rebuild_creature()
                        self._set_message("Limb mirrored!")
                    else:
                        self._set_message("Mirror failed (invalid result)")
            else:
                self._set_message("Select a bone to mirror")
        elif tag == "save":
            self._file_dialog = "save"
            self._file_input_value = self.genome.name + ".json"
        elif tag == "load":
            self._file_dialog = "load"
            self._file_input_value = ""

    def _handle_panel_event(self, event: pygame.event.Event) -> bool:
        changed = False
        w = self._panel_widgets
        for key, widget in w.items():
            if isinstance(widget, (Slider, Checkbox, TextInput)):
                if widget.handle_event(event):
                    changed = True
        return changed

    def _apply_panel_to_genome(self) -> None:
        """Write slider/checkbox values back into the genome."""
        w = self._panel_widgets
        if self.selected_bone_id:
            bone = self.genome.get_bone_by_id(self.selected_bone_id)
            if bone:
                if "length" in w: bone.length  = w["length"].value
                if "width"  in w: bone.width   = w["width"].value
                if "density" in w: bone.density = w["density"].value
                if "color_r" in w:
                    bone.color = [
                        int(w["color_r"].value),
                        int(w["color_g"].value),
                        int(w["color_b"].value),
                    ]
                joint = w.get("_active_joint")
                if joint:
                    if "ang_lo"    in w: joint.angle_limit_deg[0] = w["ang_lo"].value
                    if "ang_hi"    in w: joint.angle_limit_deg[1] = w["ang_hi"].value
                    if "torque"    in w: joint.max_motor_torque   = w["torque"].value
                    if "motorized" in w: joint.is_motorized       = w["motorized"].checked

        elif self.selected_joint_id:
            joint = w.get("_active_joint")
            if joint:
                if "ang_lo"    in w: joint.angle_limit_deg[0] = w["ang_lo"].value
                if "ang_hi"    in w: joint.angle_limit_deg[1] = w["ang_hi"].value
                if "torque"    in w: joint.max_motor_torque   = w["torque"].value
                if "motorized" in w: joint.is_motorized       = w["motorized"].checked

    def _handle_confirm_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_y:
                self._execute_confirm()
            elif event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                self._confirm_dialog = False
                self._confirm_action = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            yes_rect = pygame.Rect(PANEL_W + CANVAS_W // 2 - 90, WINDOW_H // 2 + 30, 80, 36)
            no_rect  = pygame.Rect(PANEL_W + CANVAS_W // 2 + 10, WINDOW_H // 2 + 30, 80, 36)
            if yes_rect.collidepoint(event.pos):
                self._execute_confirm()
            elif no_rect.collidepoint(event.pos):
                self._confirm_dialog = False
                self._confirm_action = None

    def _execute_confirm(self) -> None:
        if self._confirm_action == "delete_bone" and self.selected_bone_id:
            self._delete_bone(self.selected_bone_id)
        self._confirm_dialog = False
        self._confirm_action = None

    def _handle_file_dialog_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self._file_input_value = self._file_input_value[:-1]
            elif event.key == pygame.K_RETURN:
                self._execute_file_dialog()
            elif event.key == pygame.K_ESCAPE:
                self._file_dialog = None
            elif event.unicode.isprintable() and len(self._file_input_value) < 60:
                self._file_input_value += event.unicode
        if event.type == pygame.MOUSEBUTTONDOWN:
            ok_rect     = pygame.Rect(PANEL_W + CANVAS_W // 2 - 90, WINDOW_H // 2 + 50, 80, 36)
            cancel_rect = pygame.Rect(PANEL_W + CANVAS_W // 2 + 10, WINDOW_H // 2 + 50, 80, 36)
            if ok_rect.collidepoint(event.pos):
                self._execute_file_dialog()
            elif cancel_rect.collidepoint(event.pos):
                self._file_dialog = None

    def _execute_file_dialog(self) -> None:
        """Perform save or load with path traversal protection.

        Security:
          - Filename sanitized with os.path.basename() to strip traversal.
          - Resolved path validated to be inside SAFE_SAVE_DIR.
        """
        raw_filename = self._file_input_value.strip()
        # Sanitize: strip any directory traversal
        safe_filename = os.path.basename(raw_filename)
        if not safe_filename.endswith(".json"):
            safe_filename += ".json"

        os.makedirs(SAFE_SAVE_DIR, exist_ok=True)
        full_path = os.path.realpath(os.path.join(SAFE_SAVE_DIR, safe_filename))

        # Validate path stays inside safe directory
        if not full_path.startswith(SAFE_SAVE_DIR + os.sep):
            self._set_message("Invalid filename (path traversal detected)")
            self._file_dialog = None
            return

        if self._file_dialog == "save":
            self.genome.save(full_path)
            self._set_message(f"Saved → {safe_filename}")
        elif self._file_dialog == "load":
            if os.path.isfile(full_path):
                try:
                    new_genome = Genome.load(full_path)
                    self.genome = new_genome
                    self.selected_bone_id  = None
                    self.selected_joint_id = None
                    self._rebuild_creature()
                    self._set_message(f"Loaded '{new_genome.name}'")
                except Exception as e:
                    self._set_message(f"Load failed: {e}")
            else:
                self._set_message(f"File not found: {safe_filename}")

        self._file_dialog = None

    # ------------------------------------------------------------------
    # Bone selection
    # ------------------------------------------------------------------

    def _try_select(self, screen_pos: Tuple[int, int]) -> None:
        if self.creature is None:
            return
        world_pos = self._screen_to_world(screen_pos)

        # Check bones
        best_bone = None
        best_dist = float("inf")
        for bid, body in self.creature.bodies.items():
            dx = body.position.x - world_pos[0]
            dy = body.position.y - world_pos[1]
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_bone = bid

        # Check joints (click on joint circle)
        best_joint = None
        best_jdist = float("inf")
        for jnt in self.genome.joints:
            if jnt.bone_a not in self.creature.bodies:
                continue
            body_a = self.creature.bodies[jnt.bone_a]
            ax, ay = jnt.anchor_a[0] * PIXELS_PER_METER, jnt.anchor_a[1] * PIXELS_PER_METER
            wp = body_a.local_to_world((ax, ay))
            dx = wp.x - world_pos[0]
            dy = wp.y - world_pos[1]
            dist = math.hypot(dx, dy)
            if dist < best_jdist:
                best_jdist = dist
                best_joint = jnt.id

        if best_jdist < 15 and best_jdist < best_dist:
            self.selected_joint_id = best_joint
            self.selected_bone_id  = None
        elif best_dist < 80:
            self.selected_bone_id  = best_bone
            self.selected_joint_id = None
        else:
            self.selected_bone_id  = None
            self.selected_joint_id = None

        self._rebuild_panel_widgets()

    # ------------------------------------------------------------------
    # Add bone drag
    # ------------------------------------------------------------------

    def _start_add_bone_drag(self, screen_pos: Tuple[int, int]) -> None:
        if self.creature is None:
            return
        world_pos = self._screen_to_world(screen_pos)
        # Find closest bone as parent
        best_bone = None
        best_dist = float("inf")
        for bid, body in self.creature.bodies.items():
            dist = math.hypot(body.position.x - world_pos[0], body.position.y - world_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_bone = bid
        if best_bone and best_dist < 150:
            self._add_parent_id  = best_bone
            self._add_drag_start = screen_pos
            self._add_drag_end   = screen_pos
            self._set_message(f"Drag to set length (parent: {best_bone})")

    def _finish_add_bone_drag(self) -> None:
        if self._add_parent_id is None or self._add_drag_start is None:
            return
        sx, sy = self._add_drag_start
        ex, ey = self._add_drag_end
        dx = ex - sx
        dy = ey - sy
        length_px = math.hypot(dx, dy)
        if length_px < 10:
            self._set_message("Drag was too short; bone not added")
            self._adding_bone    = False
            self._add_drag_start = None
            self._add_drag_end   = None
            return

        length_m = length_px / PIXELS_PER_METER
        length_m = max(0.1, min(3.0, length_m))
        angle    = math.atan2(-dy, dx)  # note: screen y is flipped

        parent_bone = self.genome.get_bone_by_id(self._add_parent_id)
        ap_x = 0.0
        ap_y = -(parent_bone.length / 2) if parent_bone else -0.3

        # Generate unique ID
        used_ids = {b.id for b in self.genome.bones}
        new_id = f"bone_{len(self.genome.bones)}"
        counter = 0
        while new_id in used_ids:
            counter += 1
            new_id = f"bone_{len(self.genome.bones)+counter}"

        new_bone = Bone(
            id=new_id,
            parent=self._add_parent_id,
            shape="box",
            length=round(length_m, 3),
            width=0.12,
            density=1.0,
            color=[100, 180, 240],
            attach_point=[round(ap_x, 3), round(ap_y, 3)],
        )

        jid = f"j_{new_id}"
        while jid in {j.id for j in self.genome.joints}:
            jid += "_x"

        new_joint = Joint(
            id=jid,
            bone_a=self._add_parent_id,
            bone_b=new_id,
            anchor_a=[ap_x, ap_y],
            anchor_b=[0.0, round(length_m / 2, 3)],
            angle_limit_deg=[-60.0, 60.0],
            max_motor_torque=500.0,
            is_motorized=True,
        )

        self.genome.bones.append(new_bone)
        self.genome.joints.append(new_joint)

        try:
            validate_genome(self.genome)
            self.selected_bone_id = new_id
            self._rebuild_creature()
            self._set_message(f"Added bone '{new_id}' (length={length_m:.2f}m)")
        except ValueError as e:
            self.genome.bones.pop()
            self.genome.joints.pop()
            self._set_message(f"Invalid bone: {e}")

        self._adding_bone    = False
        self._add_parent_id  = None
        self._add_drag_start = None
        self._add_drag_end   = None

    # ------------------------------------------------------------------
    # Delete bone
    # ------------------------------------------------------------------

    def _delete_bone(self, bone_id: str) -> None:
        """Remove a bone and its entire subtree + associated joints."""
        from creature.morphology import build_bone_tree
        children = build_bone_tree(self.genome)
        to_remove = set()
        stack = [bone_id]
        while stack:
            node = stack.pop()
            to_remove.add(node)
            stack.extend(children.get(node, []))

        self.genome.bones  = [b for b in self.genome.bones  if b.id not in to_remove]
        self.genome.joints = [j for j in self.genome.joints
                              if j.bone_a not in to_remove and j.bone_b not in to_remove]
        self.selected_bone_id = None
        self._rebuild_creature()
        self._set_message(f"Deleted '{bone_id}' and its subtree")

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = int(wx - self.camera_x) + PANEL_W
        sy = int(WINDOW_H - wy)
        return (sx, sy)

    def _screen_to_world(self, screen_pos: Tuple[int, int]) -> Tuple[float, float]:
        sx, sy = screen_pos
        wx = (sx - PANEL_W) + self.camera_x
        wy = WINDOW_H - sy
        return (wx, wy)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(BG_CANVAS)
        # Draw canvas area
        canvas_surf = self.screen.subsurface(pygame.Rect(PANEL_W, 0, CANVAS_W, WINDOW_H))
        self._draw_canvas(canvas_surf)
        # Draw panel
        panel_surf = self.screen.subsurface(pygame.Rect(0, 0, PANEL_W, WINDOW_H))
        self._draw_panel(panel_surf)
        # Draw toolbar overlay (on main screen, above canvas)
        self._draw_toolbar()
        # Dialogs
        if self._confirm_dialog:
            self._draw_confirm_dialog()
        if self._file_dialog:
            self._draw_file_dialog()
        # Message banner
        if self._message_timer > 0:
            self._message_timer -= 1
            msg = self.font.render(self._message, True, TEXT_CLR)
            mx = PANEL_W + (CANVAS_W - msg.get_width()) // 2
            self.screen.blit(msg, (mx, WINDOW_H - 40))

    def _draw_canvas(self, surf: pygame.Surface) -> None:
        surf.fill(BG_CANVAS)
        # Grid
        grid_sp = 100
        start_gx = int(self.camera_x // grid_sp) * grid_sp
        for gx in range(start_gx, start_gx + CANVAS_W + grid_sp * 2, grid_sp):
            sx = int(gx - self.camera_x)
            pygame.draw.line(surf, GRID_CLR, (sx, 0), (sx, WINDOW_H))
        for gy in range(0, WINDOW_H, 100):
            pygame.draw.line(surf, GRID_CLR, (0, gy), (CANVAS_W, gy))

        # Ground
        gs_y = WINDOW_H - GROUND_Y
        pygame.draw.rect(surf, GROUND_CLR, (0, gs_y, CANVAS_W, WINDOW_H - gs_y))
        pygame.draw.line(surf, GROUND_LINE, (0, gs_y), (CANVAS_W, gs_y), 2)

        if self.creature:
            self._draw_creature_on(surf)
            self._draw_joints_on(surf)

        # Add-bone drag preview
        if self._adding_bone and self._add_drag_start and self._add_drag_end:
            s = (self._add_drag_start[0] - PANEL_W, self._add_drag_start[1])
            e = (self._add_drag_end[0]   - PANEL_W, self._add_drag_end[1])
            pygame.draw.line(surf, ACCENT, s, e, 3)
            pygame.draw.circle(surf, ACCENT, e, 6)

    def _draw_creature_on(self, surf: pygame.Surface) -> None:
        for bid, body in self.creature.bodies.items():
            shape = self.creature.shapes[bid]
            bone  = self.genome.get_bone_by_id(bid)
            color = tuple(bone.color) if bone else (180, 180, 180)
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            screen_verts = [(int(v.x - self.camera_x), int(WINDOW_H - v.y)) for v in verts]
            pygame.draw.polygon(surf, color, screen_verts)
            outline = SEL_OUTLINE if bid == self.selected_bone_id else BONE_OUTLINE
            width   = 3 if bid == self.selected_bone_id else 1
            pygame.draw.polygon(surf, outline, screen_verts, width)
            com = (int(body.position.x - self.camera_x), int(WINDOW_H - body.position.y))
            lbl = self.font_sm.render(bid, True, (255, 255, 255))
            surf.blit(lbl, (com[0] - lbl.get_width() // 2, com[1] - lbl.get_height() // 2))

    def _draw_joints_on(self, surf: pygame.Surface) -> None:
        for jnt in self.genome.joints:
            if jnt.bone_a not in self.creature.bodies:
                continue
            ba = self.creature.bodies[jnt.bone_a]
            ax, ay = jnt.anchor_a[0] * PIXELS_PER_METER, jnt.anchor_a[1] * PIXELS_PER_METER
            wp = ba.local_to_world((ax, ay))
            sx = int(wp.x - self.camera_x)
            sy = int(WINDOW_H - wp.y)
            color = JOINT_MOTOR if jnt.is_motorized else JOINT_PASSIVE
            sel   = jnt.id == self.selected_joint_id
            pygame.draw.circle(surf, color, (sx, sy), 8 if sel else 5)
            if sel:
                pygame.draw.circle(surf, SEL_OUTLINE, (sx, sy), 8, 2)

    def _draw_toolbar(self) -> None:
        """Draw toolbar buttons (on main screen above canvas)."""
        for btn in self._toolbar_buttons:
            if btn._tag == "add_bone" and self._adding_bone:
                btn.color = BTN_ACTIVE
            elif btn._tag == "test_drive" and self.test_drive:
                btn.color = SUCCESS
            else:
                btn.color = DANGER if btn._tag == "delete_bone" else BTN_NORMAL
            btn.draw(self.screen, self.font)

    def _draw_panel(self, surf: pygame.Surface) -> None:
        surf.fill(BG_PANEL)
        pygame.draw.line(surf, (50, 55, 85), (PANEL_W - 1, 0), (PANEL_W - 1, WINDOW_H))

        # Header
        title = self.font_h.render("CREATURE EDITOR", True, ACCENT)
        surf.blit(title, (10, 10))
        name_lbl = self.font_sm.render(f"Genome: {self.genome.name}", True, TEXT_DIM)
        surf.blit(name_lbl, (10, 30))
        bones_lbl = self.font_sm.render(
            f"Bones: {len(self.genome.bones)}  |  Joints: {len(self.genome.joints)}", True, TEXT_DIM
        )
        surf.blit(bones_lbl, (10, 44))

        # Selection info
        if self.selected_bone_id:
            sel_lbl = self.font.render(f"▶ {self.selected_bone_id}", True, SEL_OUTLINE)
            surf.blit(sel_lbl, (10, 58))
        elif self.selected_joint_id:
            sel_lbl = self.font.render(f"▶ {self.selected_joint_id}", True, JOINT_CLR)
            surf.blit(sel_lbl, (10, 58))

        # Draw widgets
        for key, widget in self._panel_widgets.items():
            if isinstance(widget, Slider):
                # Draw label
                lbl_key = key + "_lbl"
                if lbl_key in self._panel_widgets:
                    lbl_y = self._panel_widgets[lbl_key][1] if isinstance(self._panel_widgets[lbl_key], tuple) else 0
                    lbl_text = self._panel_widgets[lbl_key][0] if isinstance(self._panel_widgets[lbl_key], tuple) else ""
                    lbl = self.font_sm.render(lbl_text, True, TEXT_DIM)
                    surf.blit(lbl, (10, lbl_y))
                widget.draw(surf, self.font_sm)
            elif isinstance(widget, Checkbox):
                widget.draw(surf, self.font, "Motorized")
            elif isinstance(key, str) and not key.startswith("_") and not key.endswith("_lbl"):
                pass  # handled above

        # Test drive physics info
        if self.test_drive:
            info = self.font_sm.render(f"Physics: {self._test_drive_physics_steps} steps", True, SUCCESS)
            surf.blit(info, (10, WINDOW_H - 30))

    def _draw_confirm_dialog(self) -> None:
        overlay = pygame.Surface((CANVAS_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (PANEL_W, 0))
        cx = PANEL_W + CANVAS_W // 2
        cy = WINDOW_H // 2
        box = pygame.Rect(cx - 200, cy - 60, 400, 140)
        pygame.draw.rect(self.screen, (28, 30, 50), box, border_radius=10)
        pygame.draw.rect(self.screen, (70, 75, 110), box, 2, border_radius=10)
        msg = self.font_h.render(f"Delete '{self.selected_bone_id}' and subtree?", True, DANGER)
        self.screen.blit(msg, (cx - msg.get_width() // 2, cy - 40))
        hint = self.font_sm.render("Press Y / N  or click below", True, TEXT_DIM)
        self.screen.blit(hint, (cx - hint.get_width() // 2, cy - 15))
        yes_btn = Button(pygame.Rect(cx - 90, cy + 30, 80, 36), "YES", danger=True)
        no_btn  = Button(pygame.Rect(cx + 10, cy + 30, 80, 36), "NO")
        yes_btn.draw(self.screen, self.font)
        no_btn.draw(self.screen, self.font)

    def _draw_file_dialog(self) -> None:
        overlay = pygame.Surface((CANVAS_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (PANEL_W, 0))
        cx = PANEL_W + CANVAS_W // 2
        cy = WINDOW_H // 2
        box = pygame.Rect(cx - 220, cy - 80, 440, 180)
        pygame.draw.rect(self.screen, (28, 30, 50), box, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT, box, 2, border_radius=10)
        mode = "SAVE GENOME" if self._file_dialog == "save" else "LOAD GENOME"
        lbl = self.font_h.render(mode, True, ACCENT)
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - 65))
        dir_note = self.font_sm.render(f"Dir: creature/saved/", True, TEXT_DIM)
        self.screen.blit(dir_note, (cx - dir_note.get_width() // 2, cy - 40))
        # Input box
        input_rect = pygame.Rect(cx - 180, cy - 15, 360, 30)
        pygame.draw.rect(self.screen, INPUT_ACTIVE, input_rect, border_radius=4)
        pygame.draw.rect(self.screen, ACCENT, input_rect, 1, border_radius=4)
        val_surf = self.font.render(self._file_input_value + "|", True, TEXT_CLR)
        self.screen.blit(val_surf, (input_rect.x + 5, input_rect.centery - val_surf.get_height() // 2))
        ok_btn  = Button(pygame.Rect(cx - 90, cy + 50, 80, 36), "OK")
        can_btn = Button(pygame.Rect(cx + 10, cy + 50, 80, 36), "Cancel")
        ok_btn.draw(self.screen, self.font)
        can_btn.draw(self.screen, self.font)

    # ------------------------------------------------------------------
    # Message banner
    # ------------------------------------------------------------------

    def _set_message(self, msg: str, duration: int = 180) -> None:
        self._message       = msg
        self._message_timer = duration


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Creature Genome Editor")
    parser.add_argument("--genome", default="creature/presets/biped.json",
                        help="Path to genome JSON to open")
    args = parser.parse_args()

    genome_path = args.genome
    if not os.path.isfile(genome_path):
        print(f"[error] Genome not found: {genome_path}")
        sys.exit(1)

    genome = Genome.load(genome_path)
    editor = CreatureEditor(genome)
    editor.run()


if __name__ == "__main__":
    main()
