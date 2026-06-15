"""
environment/renderer.py

Pygame renderer for Creature Evolution RL.
Draws bones (filled polygons), joints (circles), ground, HUD overlay,
and follows the creature's torso with camera tracking.

Usage:
    renderer = Renderer(width=1000, height=600)
    renderer.init()
    renderer.render(space, creature, info={...})
    renderer.tick(fps=60)
    renderer.close()
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import pygame
import pymunk
import pymunk.pygame_util

from creature.builder import PIXELS_PER_METER, Creature


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_COLOR        = (18, 18, 28)
GROUND_COLOR    = (55, 60, 80)
GROUND_LINE_CLR = (80, 90, 120)
JOINT_COLOR     = (255, 200, 60)
HUD_BG          = (10, 10, 20, 180)
HUD_TEXT        = (220, 220, 255)
GRID_COLOR      = (28, 28, 42)
SELECT_COLOR    = (255, 240, 80)


class Renderer:
    """Standalone Pygame renderer for a creature simulation.

    Args:
        width:  Window width in pixels.
        height: Window height in pixels.
        title:  Window caption.
    """

    def __init__(self, width: int = 1000, height: int = 600, title: str = "Creature Evolution RL") -> None:
        self.width = width
        self.height = height
        self.title = title
        self.screen: Optional[pygame.Surface] = None
        self.font_large: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
        self.clock: Optional[pygame.time.Clock] = None
        self._initialized = False
        self.camera_x: float = 0.0   # current camera offset (pixels)
        self.ground_y: float = 50.0  # ground y in physics space (pymunk)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialize Pygame window. Call before first render()."""
        if self._initialized:
            return
        pygame.init()
        pygame.display.set_caption(self.title)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 13)
        self.clock = pygame.time.Clock()
        self._initialized = True

    def close(self) -> None:
        """Quit Pygame cleanly."""
        if self._initialized:
            pygame.quit()
            self._initialized = False

    def tick(self, fps: int = 60) -> None:
        """Advance display clock and pump events. Returns True if quit requested."""
        self.clock.tick(fps)
        pygame.display.flip()

    def poll_quit(self) -> bool:
        """Return True if the user closed the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    # ------------------------------------------------------------------
    # Core render
    # ------------------------------------------------------------------

    def render(
        self,
        creature: Creature,
        info: Optional[Dict[str, Any]] = None,
        camera_follow: bool = True,
        selected_bone_id: Optional[str] = None,
    ) -> None:
        """Draw one frame.

        Args:
            creature:         The Creature to draw.
            info:             Dict of strings to show in HUD.
            camera_follow:    If True, camera tracks torso x-position.
            selected_bone_id: If set, outline that bone with a highlight.
        """
        if not self._initialized:
            self.init()

        # Camera tracking
        if camera_follow:
            torso_x, torso_y = creature.get_torso_position()
            target_cam = torso_x - self.width * 0.35
            self.camera_x += (target_cam - self.camera_x) * 0.08

        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_ground()
        self._draw_creature(creature, selected_bone_id)
        self._draw_joints(creature)
        if info:
            self._draw_hud(info)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert physics world coordinates to screen pixels.

        Pymunk has Y-up, Pygame has Y-down → flip Y.
        """
        sx = int(wx - self.camera_x)
        sy = int(self.height - wy)
        return (sx, sy)

    def _draw_grid(self) -> None:
        """Draw a subtle vertical/horizontal grid."""
        grid_spacing = 100
        # Vertical lines
        start_x = int(self.camera_x // grid_spacing) * grid_spacing
        for gx in range(start_x, start_x + self.width + grid_spacing, grid_spacing):
            sx = int(gx - self.camera_x)
            pygame.draw.line(self.screen, GRID_COLOR, (sx, 0), (sx, self.height))
        # Horizontal lines
        for gy_pct in range(0, self.height + 100, 100):
            pygame.draw.line(self.screen, GRID_COLOR, (0, gy_pct), (self.width, gy_pct))

    def _draw_ground(self) -> None:
        """Draw the ground as a filled rect below the ground line."""
        ground_screen_y = self.height - int(self.ground_y)
        # Ground fill
        pygame.draw.rect(
            self.screen, GROUND_COLOR,
            (0, ground_screen_y, self.width, self.height - ground_screen_y)
        )
        # Ground line
        pygame.draw.line(
            self.screen, GROUND_LINE_CLR,
            (0, ground_screen_y), (self.width, ground_screen_y), 3
        )
        # Distance markers every 5 metres
        for m in range(-5, int(self.camera_x / PIXELS_PER_METER) + 30, 5):
            wx = m * PIXELS_PER_METER
            sx = int(wx - self.camera_x)
            if -20 < sx < self.width + 20:
                pygame.draw.line(
                    self.screen, GROUND_LINE_CLR,
                    (sx, ground_screen_y - 6), (sx, ground_screen_y + 4), 1
                )
                label = self.font_small.render(f"{m}m", True, (80, 90, 130))
                self.screen.blit(label, (sx - 12, ground_screen_y + 6))

    def _draw_creature(self, creature: Creature, selected_bone_id: Optional[str]) -> None:
        """Draw each bone as a filled, rotated polygon."""
        for bid, body in creature.bodies.items():
            shape = creature.shapes[bid]
            bone = creature.genome.get_bone_by_id(bid)
            color = tuple(bone.color) if bone else (180, 180, 180)

            # Get world-space vertices
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            screen_verts = [self._world_to_screen(v.x, v.y) for v in verts]

            pygame.draw.polygon(self.screen, color, screen_verts)

            # Outline
            outline_color = SELECT_COLOR if bid == selected_bone_id else (0, 0, 0)
            outline_width = 3 if bid == selected_bone_id else 1
            pygame.draw.polygon(self.screen, outline_color, screen_verts, outline_width)

            # Bone ID label (small, at COM)
            com_screen = self._world_to_screen(body.position.x, body.position.y)
            label = self.font_small.render(bid, True, (255, 255, 255))
            self.screen.blit(label, (com_screen[0] - label.get_width() // 2,
                                     com_screen[1] - label.get_height() // 2))

    def _draw_joints(self, creature: Creature) -> None:
        """Draw joint positions as small circles."""
        for jnt in creature.genome.joints:
            if jnt.bone_a not in creature.bodies:
                continue
            body_a = creature.bodies[jnt.bone_a]
            # Convert local anchor_a to world position
            ax = jnt.anchor_a[0] * PIXELS_PER_METER
            ay = jnt.anchor_a[1] * PIXELS_PER_METER
            world_pt = body_a.local_to_world((ax, ay))
            sx, sy = self._world_to_screen(world_pt.x, world_pt.y)
            radius = 6 if jnt.is_motorized else 4
            pygame.draw.circle(self.screen, JOINT_COLOR, (sx, sy), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), radius, 1)

    def _draw_hud(self, info: Dict[str, Any]) -> None:
        """Draw a semi-transparent HUD in the top-left corner."""
        lines = [f"{k}: {v}" for k, v in info.items()]
        padding = 10
        line_h = 20
        panel_w = 280
        panel_h = padding * 2 + len(lines) * line_h

        hud_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        hud_surf.fill((10, 10, 20, 180))
        self.screen.blit(hud_surf, (10, 10))

        for i, line in enumerate(lines):
            text = self.font_small.render(line, True, HUD_TEXT)
            self.screen.blit(text, (10 + padding, 10 + padding + i * line_h))

    # ------------------------------------------------------------------
    # Static / paused snapshot
    # ------------------------------------------------------------------

    def render_static(
        self,
        creature: Creature,
        camera_x: float = 0.0,
        selected_bone_id: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Render a single static frame (no camera animation, no flip)."""
        if not self._initialized:
            self.init()
        self.camera_x = camera_x
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_ground()
        self._draw_creature(creature, selected_bone_id)
        self._draw_joints(creature)
        if info:
            self._draw_hud(info)
