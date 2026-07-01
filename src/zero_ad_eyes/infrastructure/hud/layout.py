"""Relative HUD sub-region layout (EPIC C — C1..C5).

Calibration (EPIC B) locates the *whole* ``top_bar`` and ``selection_panel``
boxes. This module subdivides those boxes into the individual counters/swatches
the reader OCRs, expressed as fractions of the parent box so a single layout
works across resolutions and UI scales (A4). All fractions are configuration
(NF7): the defaults below encode the standard 0 A.D. top-bar ordering
(food, wood, stone, metal, population, then phase on the right) and are the
project's *assumption*, not a measured fact — override per theme if needed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.geometry import ScreenBBox


class FractionalRegion(BaseModel):
    """A box expressed as fractions [0, 1] of a parent :class:`ScreenBBox`."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)

    def project(self, parent: ScreenBBox) -> ScreenBBox:
        """Resolve this fractional region against a concrete parent box."""

        return ScreenBBox(
            x=parent.x + self.x * parent.width,
            y=parent.y + self.y * parent.height,
            width=self.width * parent.width,
            height=self.height * parent.height,
        )


# Full-height text slots; the number sits centred vertically in the bar.
def _slot(x: float, width: float) -> FractionalRegion:
    return FractionalRegion(x=x, y=0.0, width=width, height=1.0)


class TopBarLayout(BaseModel):
    """Where each top-bar element sits, as fractions of the ``top_bar`` box."""

    model_config = ConfigDict(frozen=True)

    food: FractionalRegion = _slot(0.03, 0.10)
    wood: FractionalRegion = _slot(0.15, 0.10)
    stone: FractionalRegion = _slot(0.27, 0.10)
    metal: FractionalRegion = _slot(0.39, 0.10)
    population: FractionalRegion = _slot(0.51, 0.12)
    phase: FractionalRegion = _slot(0.80, 0.18)
    # Own-player colour swatch: the small coloured emblem, upper-left of the bar.
    swatch: FractionalRegion = FractionalRegion(x=0.0, y=0.2, width=0.025, height=0.6)
    # Civ emblem / name region (best-effort OCR), right of the swatch.
    civ: FractionalRegion = _slot(0.66, 0.13)


class SelectionPanelLayout(BaseModel):
    """Sub-regions of the bottom-centre selection panel (C5, best-effort)."""

    model_config = ConfigDict(frozen=True)

    entity_type: FractionalRegion = FractionalRegion(x=0.05, y=0.05, width=0.9, height=0.25)
    health: FractionalRegion = FractionalRegion(x=0.05, y=0.35, width=0.5, height=0.25)
    queue: FractionalRegion = FractionalRegion(x=0.05, y=0.7, width=0.9, height=0.28)
