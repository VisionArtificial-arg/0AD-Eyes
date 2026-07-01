"""Minimap model value objects (REQUIREMENTS.md §4.4).

The minimap is an independent, low-resolution position source. It is fused with
main-view perception (EPIC G) but modelled separately so a projection error in one
does not silently corrupt the other (risk R2).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .confidence import Confidence
from .geometry import WorldPoint
from .taxonomy import Ownership


class FogState(StrEnum):
    """Per-cell visibility (REQUIREMENTS.md §4.4 / §4.5)."""

    UNEXPLORED = "unexplored"  # never seen (black)
    EXPLORED = "explored"  # seen before, not currently visible (shroud)
    VISIBLE = "visible"


class Blip(BaseModel):
    """A coloured dot on the minimap, resolved to world space."""

    model_config = ConfigDict(frozen=True)

    world_pos: WorldPoint
    ownership: Ownership
    confidence: Confidence


class ViewportRect(BaseModel):
    """The camera's footprint projected onto the minimap, in world space."""

    model_config = ConfigDict(frozen=True)

    top_left: WorldPoint
    bottom_right: WorldPoint


class MinimapModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    blips: tuple[Blip, ...] = ()
    viewport: ViewportRect | None = None
    confidence: Confidence = Confidence.unknown()
