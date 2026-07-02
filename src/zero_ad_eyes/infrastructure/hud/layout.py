"""Relative HUD sub-region layout (EPIC C — C1..C5).

Calibration (EPIC B) locates the *whole* ``top_bar`` and ``selection_panel``
boxes. This module subdivides those boxes into the individual counters/swatches
the reader OCRs, expressed as fractions of the parent box so a single layout
works across resolutions and UI scales (A4). All fractions are configuration
(NF7): they are supplied at the composition root from the ``hud`` config section
(the generated default encodes the standard 0 A.D. top-bar ordering — food, wood,
stone, metal, population, then phase on the right), never baked in here.
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


class TopBarLayout(BaseModel):
    """Where each top-bar element sits, as fractions of the ``top_bar`` box."""

    model_config = ConfigDict(frozen=True)

    food: FractionalRegion
    wood: FractionalRegion
    stone: FractionalRegion
    metal: FractionalRegion
    population: FractionalRegion
    phase: FractionalRegion
    # Own-player colour swatch: the small coloured emblem, upper-left of the bar.
    swatch: FractionalRegion
    # Civ emblem / name region (best-effort OCR), right of the swatch.
    civ: FractionalRegion


class SelectionPanelLayout(BaseModel):
    """Sub-regions of the bottom-centre selection panel (C5, best-effort)."""

    model_config = ConfigDict(frozen=True)

    entity_type: FractionalRegion
    health: FractionalRegion
    queue: FractionalRegion
