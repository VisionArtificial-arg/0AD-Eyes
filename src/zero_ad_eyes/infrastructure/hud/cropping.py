"""Crop and colour-sampling helpers for HUD regions (EPIC C — C1/C4).

Pure numpy utilities that turn a :class:`ScreenBBox` into a pixel sub-array and
sample a representative colour. Kept separate from the reader so they can be unit
tested against tiny synthetic images without OCR or a display.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zero_ad_eyes.domain.geometry import ScreenBBox


def crop(image: Any, bbox: ScreenBBox) -> Any:
    """Return the sub-image inside ``bbox``, clamped to the image bounds.

    Coordinates are rounded to ints and clamped to ``[0, dim]`` so an out-of-range
    or partially-off-screen box yields a (possibly empty) array instead of raising
    (NF4). The result is a *view* into ``image`` when possible.
    """

    height, width = image.shape[0], image.shape[1]
    x0 = int(round(bbox.x))
    y0 = int(round(bbox.y))
    x1 = int(round(bbox.x + bbox.width))
    y1 = int(round(bbox.y + bbox.height))

    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))

    if x1 <= x0 or y1 <= y0:
        return image[0:0, 0:0]
    return image[y0:y1, x0:x1]


def sample_color_rgb(image: Any, bbox: ScreenBBox) -> tuple[int, int, int] | None:
    """Sample a representative RGB colour from ``bbox`` (self-swatch, C4).

    The input frame is BGR (OpenCV); the returned tuple is RGB to match
    :attr:`~zero_ad_eyes.domain.hud.HudState.self_player_color`. The median across
    the crop is used so a few anti-aliased edge pixels do not skew the reading.
    Returns ``None`` when the crop is empty.
    """

    region = crop(image, bbox)
    if region.size == 0:
        return None
    flat = region.reshape(-1, region.shape[-1]) if region.ndim == 3 else region.reshape(-1, 1)
    median = np.median(flat[:, :3], axis=0)
    b, g, r = (int(round(float(c))) for c in median[:3])
    return (r, g, b)
