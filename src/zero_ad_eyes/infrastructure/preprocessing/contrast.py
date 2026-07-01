"""Contrast enhancement via CLAHE (P5, CV-06).

Small, low-contrast objects — a lone unit against terrain, a faint resource icon —
get lost under global contrast operators. CLAHE (Contrast-Limited Adaptive
Histogram Equalization) equalizes locally within tiles and clips the histogram to
avoid amplifying noise, surfacing those objects without blowing out bright regions.

For a colour frame, equalizing each BGR channel would shift hue; instead we
equalize only the *lightness* channel in Lab space and convert back, preserving
colour while boosting local contrast.
"""

from __future__ import annotations

import cv2

from .base import Image, ImageStep


class ClaheContrast(ImageStep):
    """Apply CLAHE to the luminance of a frame (P5).

    Accepts either a single-channel (grayscale) image — equalized directly — or a
    3-channel BGR image, in which case only the Lab *L* channel is equalized so
    colour is preserved. ``clip_limit`` caps local contrast amplification;
    ``tile_grid_size`` sets the adaptive tile granularity.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        if clip_limit <= 0.0:
            raise ValueError("clip_limit must be positive")
        rows, cols = tile_grid_size
        if rows < 1 or cols < 1:
            raise ValueError("tile_grid_size dimensions must be positive")
        # createCLAHE holds no per-frame state; one instance is reused across calls.
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def transform(self, image: Image) -> Image:
        if image.ndim == 2:
            return self._clahe.apply(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        equalized = self._clahe.apply(lightness)
        merged = cv2.merge((equalized, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
