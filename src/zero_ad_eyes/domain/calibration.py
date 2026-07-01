"""Session calibration value object (REQUIREMENTS.md EPIC B).

HUD layout is discovered per session by anchor detection, never hard-coded (A4).
Downstream readers (HUD, minimap) compile against this type; the calibration
*implementation* is built in parallel behind the ``Calibrator`` port.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .geometry import ScreenBBox


class Calibration(BaseModel):
    """Where the HUD regions live in the current frame."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    ui_scale: float = Field(default=1.0, gt=0.0)
    top_bar: ScreenBBox | None = None
    minimap: ScreenBBox | None = None
    selection_panel: ScreenBBox | None = None
