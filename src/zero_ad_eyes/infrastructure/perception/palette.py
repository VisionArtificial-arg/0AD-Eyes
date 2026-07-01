"""E3 — Configurable player-colour palette (support for ownership segmentation).

0 A.D. paints every entity with its owner's player colour. Ownership (self /
ally / enemy / gaia) is therefore a *relative* interpretation of an absolute
player colour. This module models that mapping as data: a palette of
``PlayerColor`` entries, each an ``Ownership`` plus one or more HSV bands.

Working in HSV (not BGR) is what buys robustness to lighting, terrain and
shadow (E3's hard requirement): hue stays stable while brightness swings, so we
threshold tightly on hue/saturation but leave value generous. Reds straddle the
hue wrap-around (0/180), hence a colour may carry several bands.

The palette is pure configuration — swap ``DEFAULT_PALETTE`` for a colour-blind
palette without touching the segmentation code.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.taxonomy import Ownership


class HsvBand(BaseModel):
    """An inclusive HSV threshold window (OpenCV ranges: H 0-179, S/V 0-255)."""

    model_config = ConfigDict(frozen=True)

    h_lo: int = Field(ge=0, le=179)
    h_hi: int = Field(ge=0, le=179)
    s_lo: int = Field(default=70, ge=0, le=255)
    s_hi: int = Field(default=255, ge=0, le=255)
    v_lo: int = Field(default=50, ge=0, le=255)
    v_hi: int = Field(default=255, ge=0, le=255)

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Binary (0/255) mask of pixels falling inside this band."""

        lo = np.array([self.h_lo, self.s_lo, self.v_lo], dtype=np.uint8)
        hi = np.array([self.h_hi, self.s_hi, self.v_hi], dtype=np.uint8)
        return cv2.inRange(hsv, lo, hi)


class PlayerColor(BaseModel):
    """A named player colour mapped to an ownership relation and its HSV bands."""

    model_config = ConfigDict(frozen=True)

    name: str
    ownership: Ownership
    bands: tuple[HsvBand, ...]

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Union of this colour's bands over an HSV image."""

        combined: np.ndarray = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for band in self.bands:
            combined = cv2.bitwise_or(combined, band.mask(hsv))
        return combined


class PlayerPalette(BaseModel):
    """The set of player colours in play, ordered by matching priority."""

    model_config = ConfigDict(frozen=True)

    colors: tuple[PlayerColor, ...]


# Default relative palette. SELF=blue, ALLY=green, ENEMY=red (hue-wrapped),
# GAIA=yellow. Deliberately conservative saturation/value floors so weakly
# tinted terrain does not register as a player colour.
DEFAULT_PALETTE = PlayerPalette(
    colors=(
        PlayerColor(
            name="blue",
            ownership=Ownership.SELF,
            bands=(HsvBand(h_lo=100, h_hi=130),),
        ),
        PlayerColor(
            name="green",
            ownership=Ownership.ALLY,
            bands=(HsvBand(h_lo=45, h_hi=85),),
        ),
        PlayerColor(
            name="red",
            ownership=Ownership.ENEMY,
            bands=(HsvBand(h_lo=0, h_hi=10), HsvBand(h_lo=170, h_hi=179)),
        ),
        PlayerColor(
            name="yellow",
            ownership=Ownership.GAIA,
            bands=(HsvBand(h_lo=22, h_hi=34),),
        ),
    ),
)
