"""Gameplay events derived from temporal state changes (REQUIREMENTS.md EPIC G, CV-34).

Added in schema v0.2: some facts are *transitions*, not states — a unit that
appeared, a building that finished, a resource that ran out. A single frame's
state objects cannot express them; they only exist relative to prior frames.
The perception layer emits them as discrete, self-describing events so the
decision layer does not have to diff world models itself.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .confidence import Confidence


class EventKind(StrEnum):
    UNIT_APPEARED = "unit_appeared"
    UNIT_DISAPPEARED = "unit_disappeared"
    COMBAT = "combat"
    RESOURCE_DEPLETED = "resource_depleted"
    BUILDING_COMPLETED = "building_completed"
    STATE_CHANGED = "state_changed"


class Event(BaseModel):
    """A discrete transition observed between frames."""

    model_config = ConfigDict(frozen=True)

    kind: EventKind
    frame_id: int = Field(ge=0)
    entity_id: int | None = None
    confidence: Confidence = Confidence.unknown()
