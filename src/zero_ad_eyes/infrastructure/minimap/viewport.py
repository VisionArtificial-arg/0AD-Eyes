"""D5 — Camera-viewport quad extraction (REQUIREMENTS.md §4.4, CV-31).

0 A.D. draws the camera's footprint on the minimap as a bright, hollow quadrilateral
outline. This stage isolates that outline by its brightness, takes the largest
contour (the viewport dwarfs point-like gaia blips that share the white colour),
recovers its four corners (a general quad, so a tilted camera's foreshortened
footprint is preserved — not collapsed to an axis-aligned box), and projects them
into world space (D6) as a :class:`ViewportQuad` in canonical TL, TR, BR, BL order.

The white threshold, minimum size, and polygon-approximation tolerance are
configurable (NF7); ``None`` is returned when no viewport-like shape is present, so
the reader emits a model without a viewport rather than a fabricated one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from zero_ad_eyes.domain.minimap import ViewportQuad

from .projector import MinimapProjector
from .segmentation import Segmentation

_Corner = tuple[float, float]


@dataclass(frozen=True)
class ViewportDetector:
    """Extracts the camera footprint quad from the minimap (D5)."""

    white_min: int
    min_area: int
    min_side: int
    approx_epsilon_fraction: float

    def detect(
        self, segmentation: Segmentation, projector: MinimapProjector
    ) -> ViewportQuad | None:
        outline = self._white_mask(segmentation)
        contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = self._largest_contour(contours)
        if contour is None:
            return None
        corners = self._quad_corners(contour)
        if corners is None:
            return None
        tl, tr, br, bl = corners
        return ViewportQuad(
            top_left=projector.to_world(*tl),
            top_right=projector.to_world(*tr),
            bottom_right=projector.to_world(*br),
            bottom_left=projector.to_world(*bl),
        )

    def _white_mask(self, segmentation: Segmentation) -> np.ndarray:
        region = segmentation.region
        bright = np.all(region >= self.white_min, axis=2)
        active = segmentation.mask > 0
        return (bright & active).astype(np.uint8)

    def _largest_contour(self, contours: Any) -> Any | None:
        """The largest contour whose bounding box clears the size filters."""

        best: Any | None = None
        best_area = 0
        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            area = w * h
            if w < self.min_side or h < self.min_side or area < self.min_area:
                continue
            if area > best_area:
                best_area = area
                best = contour
        return best

    def _quad_corners(self, contour: Any) -> tuple[_Corner, _Corner, _Corner, _Corner] | None:
        """Four corners of ``contour`` in canonical TL, TR, BR, BL order.

        ``approxPolyDP`` recovers a general quad (keeping perspective foreshortening);
        if it does not resolve to exactly four vertices, the minimum-area rotated
        rectangle is used as a fallback (a rectangle, so perspective is lost, but a
        usable footprint is still produced).
        """

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.approx_epsilon_fraction * perimeter, True)
        points = approx.reshape(-1, 2).astype(np.float64)
        if points.shape[0] != 4:
            points = np.asarray(cv2.boxPoints(cv2.minAreaRect(contour)), dtype=np.float64)
        return _order_corners(points)


def _order_corners(points: np.ndarray) -> tuple[_Corner, _Corner, _Corner, _Corner]:
    """Order four image-space points as top-left, top-right, bottom-right, bottom-left.

    Uses the coordinate sum (TL smallest, BR largest) and the ``y - x`` difference
    (TR smallest, BL largest) — the standard order under an image y-down axis.
    """

    coordinate_sum = points.sum(axis=1)
    diagonal_diff = points[:, 1] - points[:, 0]
    top_left = points[int(np.argmin(coordinate_sum))]
    bottom_right = points[int(np.argmax(coordinate_sum))]
    top_right = points[int(np.argmin(diagonal_diff))]
    bottom_left = points[int(np.argmax(diagonal_diff))]
    return (
        (float(top_left[0]), float(top_left[1])),
        (float(top_right[0]), float(top_right[1])),
        (float(bottom_right[0]), float(bottom_right[1])),
        (float(bottom_left[0]), float(bottom_left[1])),
    )
