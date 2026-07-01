"""The ``PerceptionModel`` port output value object (MP1/MP2).

This is the *only* type the rest of the pipeline sees coming out of the model
seam. The stub adapter and the real (learned) adapter both produce exactly this,
so nothing downstream can tell them apart (REQUIREMENTS.md §5.10).

Raw pixel masks stay in ``infrastructure``; the domain contract carries geometry
as a bbox plus an optional polygon contour, keeping the core framework-free.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .confidence import Confidence
from .geometry import ScreenBBox, ScreenPoint
from .taxonomy import EntityKind


class Detection(BaseModel):
    """A single thing the perception model claims to see in one frame."""

    model_config = ConfigDict(frozen=True)

    kind: EntityKind
    bbox: ScreenBBox
    confidence: Confidence
    entity_type: str | None = None  # fine type when legible (§4.3)
    contour: tuple[ScreenPoint, ...] = ()  # optional instance outline (CV-31)


class Detections(BaseModel):
    """All detections for a single frame, tagged with that frame's id."""

    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    items: tuple[Detection, ...] = ()

    def __len__(self) -> int:
        return len(self.items)
