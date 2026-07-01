"""Minimap model value objects (REQUIREMENTS.md §4.4).

The minimap is an independent, low-resolution position source. It is fused with
main-view perception (EPIC G) but modelled separately so a projection error in one
does not silently corrupt the other (risk R2).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

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


class FogGrid(BaseModel):
    """Coarse fog-of-war raster read off the minimap (v0.2, CV-30).

    ``cells`` is row-major (``cells[row][col]``); ``rows``/``cols`` state the
    intended shape so an empty grid still declares its resolution. Kept coarse on
    purpose — the decision layer wants "is this region scouted", not a per-pixel mask.
    """

    model_config = ConfigDict(frozen=True)

    rows: int = Field(ge=0)
    cols: int = Field(ge=0)
    cells: tuple[tuple[FogState, ...], ...] = ()


class TerritoryRegion(BaseModel):
    """A contiguous area of map influence attributed to one owner (v0.2)."""

    model_config = ConfigDict(frozen=True)

    ownership: Ownership
    centroid: WorldPoint
    coverage: float = Field(ge=0.0, le=1.0)  # fraction of the map area


class TerritoryMap(BaseModel):
    """Territory ownership decomposition of the minimap (v0.2)."""

    model_config = ConfigDict(frozen=True)

    regions: tuple[TerritoryRegion, ...] = ()


class MinimapModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    blips: tuple[Blip, ...] = ()
    viewport: ViewportRect | None = None
    fog: FogGrid | None = None
    territory: TerritoryMap | None = None
    confidence: Confidence = Confidence.unknown()
