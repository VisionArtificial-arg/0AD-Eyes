"""Tracked world entity value object (REQUIREMENTS.md §4.3).

An ``Entity`` is the temporally-stable, fused view of a thing in the world: it
carries a persistent id (assigned by tracking, EPIC G), an optional world position,
health, ownership, and a *staleness* counter so the decision layer knows how long
ago the fact was actually observed (memory, §4.1 / G3).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .confidence import Confidence
from .geometry import ScreenBBox, WorldPoint
from .taxonomy import EntityKind, Ownership


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: int = Field(ge=0)
    kind: EntityKind
    ownership: Ownership = Ownership.UNKNOWN
    entity_type: str | None = None
    world_pos: WorldPoint | None = None
    screen_bbox: ScreenBBox | None = None
    health: float | None = Field(default=None, ge=0.0, le=1.0)
    selected: bool = False
    staleness: int = Field(default=0, ge=0)  # frames since last observed
    confidence: Confidence = Confidence.unknown()
