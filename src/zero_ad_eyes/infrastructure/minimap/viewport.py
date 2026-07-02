"""D5 — Camera-viewport rectangle extraction (REQUIREMENTS.md §4.4, CV-31).

0 A.D. draws the camera's footprint on the minimap as a bright, hollow quadrilateral
outline. This stage isolates that outline by its brightness, takes the largest
contour (the viewport dwarfs point-like gaia blips that share the white colour), and
projects its bounding box corners into world space (D6) as a :class:`ViewportRect`.

The white threshold and minimum size are configurable (NF7); ``None`` is returned
when no viewport-like shape is present, so the reader emits a model without a
viewport rather than a fabricated one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from zero_ad_eyes.domain.minimap import ViewportRect

from .projector import MinimapProjector
from .segmentation import Segmentation


@dataclass(frozen=True)
class ViewportDetector:
    """Extracts the camera footprint rectangle from the minimap (D5)."""

    white_min: int
    min_area: int
    min_side: int

    def detect(
        self, segmentation: Segmentation, projector: MinimapProjector
    ) -> ViewportRect | None:
        outline = self._white_mask(segmentation)
        contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = self._largest_rect(contours)
        if best is None:
            return None
        x, y, w, h = best
        return ViewportRect(
            top_left=projector.to_world(float(x), float(y)),
            bottom_right=projector.to_world(float(x + w), float(y + h)),
        )

    def _white_mask(self, segmentation: Segmentation) -> np.ndarray:
        region = segmentation.region
        bright = np.all(region >= self.white_min, axis=2)
        active = segmentation.mask > 0
        return (bright & active).astype(np.uint8)

    def _largest_rect(self, contours: Any) -> tuple[int, int, int, int] | None:
        best: tuple[int, int, int, int] | None = None
        best_area = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if w < self.min_side or h < self.min_side or area < self.min_area:
                continue
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(w), int(h))
        return best
