"""The world model — this layer's deliverable (REQUIREMENTS.md §4).

``WorldModel`` is what the perception layer hands to the decision layer through the
contract (EPIC H). Everything else in the codebase exists to populate it. It is a
pure value object: serialisable, versioned, and free of pixels or frameworks.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .entities import Entity
from .hud import HudState
from .minimap import MinimapModel

SCHEMA_VERSION = "0.1.0"


class FrameMeta(BaseModel):
    """Provenance of the frame a world model was derived from (§4.1)."""

    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp: float  # seconds; monotonic capture time
    source: str  # "live" | recording id
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class WorldModel(BaseModel):
    """Structured, machine-readable interpretation of one game frame."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    meta: FrameMeta
    hud: HudState | None = None
    minimap: MinimapModel | None = None
    entities: tuple[Entity, ...] = ()
