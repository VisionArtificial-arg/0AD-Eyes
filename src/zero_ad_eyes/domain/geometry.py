"""Coordinate value objects: screen space and world space.

The layer reasons in two frames of reference (REQUIREMENTS.md §4.6):
- *screen* pixels, origin top-left, produced by detection;
- *world* coordinates, produced by projecting screen detections through the
  recovered camera geometry (EPIC F) or the minimap calibration (D6).

Keeping them as distinct types prevents accidentally mixing the two.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScreenPoint(BaseModel):
    """A pixel location in the captured frame."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class ScreenBBox(BaseModel):
    """An axis-aligned bounding box in screen pixels."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)

    @property
    def center(self) -> ScreenPoint:
        return ScreenPoint(x=self.x + self.width / 2.0, y=self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


class WorldPoint(BaseModel):
    """A location in game/navigation space (units are engine-defined)."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
