"""Gameplay events derived from temporal state changes (REQUIREMENTS.md EPIC G, CV-34).

Added in schema v0.2: some facts are *transitions*, not states — a unit that
appeared, a building that finished, a resource that ran out. A single frame's
state objects cannot express them; they only exist relative to prior frames.
The perception layer emits them as discrete, self-describing events so the
decision layer does not have to diff world models itself.

v0.3 makes events self-describing: every event names the surface it was read from
(``origin``), carries a typed ``details`` payload (amounts, before/after values) and,
where it has one, a screen and/or world position. Policy: **events over silence** —
any observable transition becomes an event with a confidence; the consumer filters.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .confidence import Confidence
from .geometry import ScreenPoint, WorldPoint

#: A ``details`` value: a scalar only, so every event serialises flat and typed.
DetailValue = str | int | float | bool | None


class EventKind(StrEnum):
    UNIT_APPEARED = "unit_appeared"
    UNIT_DISAPPEARED = "unit_disappeared"
    COMBAT = "combat"
    RESOURCE_DEPLETED = "resource_depleted"
    BUILDING_COMPLETED = "building_completed"
    STATE_CHANGED = "state_changed"


class EventOrigin(StrEnum):
    """The surface an event was read from (v0.3)."""

    HUD = "hud"
    MINIMAP = "minimap"
    ENTITY = "entity"  # main-viewport entities (tracking)
    SYSTEM = "system"  # the perception layer itself: calibration, OCR, capture, model
    UNKNOWN = "unknown"  # v0.2 producers did not record an origin


class Event(BaseModel):
    """A discrete transition observed between frames."""

    model_config = ConfigDict(frozen=True)

    kind: EventKind
    frame_id: int = Field(ge=0)
    entity_id: int | None = None
    confidence: Confidence = Confidence.unknown()
    # v0.3 payload; all optional, so v0.2 events still validate.
    origin: EventOrigin = EventOrigin.UNKNOWN
    details: dict[str, DetailValue] = Field(default_factory=dict)
    screen_pos: ScreenPoint | None = None  # full-capture pixels
    world_pos: WorldPoint | None = None
