"""Canonical HUD layout ratios and region math (EPIC B — B2).

``HudLayoutRatios`` encodes the *only* layout knowledge this adapter holds: the
0 A.D. HUD arrangement (A1) expressed as fractions of frame width/height — top bar
spanning the top edge, minimap in the bottom-left, selection panel along the
bottom-centre. Because every value is a fraction, the same ratios yield a correct
layout at any resolution (A4); no absolute per-resolution pixel constant appears.

Each region method optionally accepts an *anchor* fraction (a band thickness the
calibrator recovered from pixels). When present the anchor refines the region;
when ``None`` the ratio is the fallback. This keeps B2 a pure, deterministic
function of width/height/ui_scale (+optional anchor), independent of any detector.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.geometry import ScreenBBox


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class HudLayoutRatios(BaseModel):
    """Canonical, resolution-relative fractions of the 0 A.D. HUD (A1)."""

    model_config = ConfigDict(frozen=True)

    # Top resource/phase bar: full-width band hugging the top edge.
    top_bar_height: float = Field(default=0.035, gt=0.0, le=1.0)
    # Bottom-left minimap: a square whose side is a fraction of frame *height*.
    minimap_side: float = Field(default=0.20, gt=0.0, le=1.0)
    # Bottom-centre selection panel: fraction of width x fraction of height.
    selection_width: float = Field(default=0.34, gt=0.0, le=1.0)
    selection_height: float = Field(default=0.16, gt=0.0, le=1.0)

    def top_bar(self, width: int, height: int, ui_scale: float, anchor: float | None) -> ScreenBBox:
        frac = anchor if anchor is not None else self.top_bar_height * ui_scale
        h = clamp(frac * height, 1.0, float(height))
        return ScreenBBox(x=0.0, y=0.0, width=float(width), height=h)

    def minimap(self, width: int, height: int, ui_scale: float, band: float | None) -> ScreenBBox:
        side_frac = band if band is not None else self.minimap_side * ui_scale
        side = clamp(side_frac * height, 1.0, float(min(width, height)))
        return ScreenBBox(x=0.0, y=float(height) - side, width=side, height=side)

    def selection_panel(
        self, width: int, height: int, ui_scale: float, band: float | None
    ) -> ScreenBBox:
        h_frac = band if band is not None else self.selection_height * ui_scale
        h = clamp(h_frac * height, 1.0, float(height))
        w = clamp(self.selection_width * ui_scale * width, 1.0, float(width))
        x = (float(width) - w) / 2.0
        return ScreenBBox(x=x, y=float(height) - h, width=w, height=h)
