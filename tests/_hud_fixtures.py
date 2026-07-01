"""Synthetic HUD frame builder shared by the calibration tests (EPIC B).

A black "scene" with optional opaque gray bands painted along the top/bottom edges,
mimicking the 0 A.D. resource bar and bottom control strip. No display, recording,
or model is required. Band thicknesses are given as *fractions* of frame height so
the same helper works at any resolution.
"""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.world_model import FrameMeta


def make_hud_frame(
    *,
    width: int = 640,
    height: int = 480,
    top_frac: float = 0.0,
    bottom_frac: float = 0.0,
    frame_id: int = 0,
) -> Frame:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if top_frac > 0.0:
        image[0 : int(round(top_frac * height)), :, :] = 180
    if bottom_frac > 0.0:
        image[height - int(round(bottom_frac * height)) : height, :, :] = 200
    return Frame(
        image=image,
        meta=FrameMeta(
            frame_id=frame_id,
            timestamp=float(frame_id),
            source="test",
            width=width,
            height=height,
        ),
    )


def within_frame(box: ScreenBBox, width: int, height: int) -> bool:
    return (
        box.x >= 0.0
        and box.y >= 0.0
        and box.x + box.width <= width + 1e-6
        and box.y + box.height <= height + 1e-6
    )
