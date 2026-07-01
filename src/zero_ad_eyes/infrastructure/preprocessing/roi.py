"""Region-of-Interest gating (P7, CV-28).

Most of a frame is irrelevant to a given consumer: a HUD reader cares about the top
bar, a minimap reader about one corner. Restricting work to a
:class:`~zero_ad_eyes.domain.geometry.ScreenBBox` cuts latency and suppresses
distractors before heavier stages run.

Two modes make an explicit trade-off:
- ``MASK`` (default): zero everything outside the ROI but keep the frame's shape.
  ``frame.meta`` stays fully consistent with ``image`` — the safe, composable
  default when downstream code assumes capture-resolution coordinates.
- ``CROP``: return only the ROI sub-image (fewer pixels, real latency win). The
  returned image is smaller than ``meta.width``/``meta.height``; ``meta`` is
  preserved as capture *provenance*, so downstream must map ROI-local coordinates
  back through the box origin. Choose this only when the consumer is ROI-aware.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox

from .base import Image, ImageStep


class GateMode(Enum):
    """How the ROI restricts the frame."""

    MASK = "mask"
    CROP = "crop"


def _pixel_bounds(bbox: ScreenBBox, height: int, width: int) -> tuple[int, int, int, int]:
    """Clamp a float bbox to integer ``(y0, y1, x0, x1)`` inside the image."""

    x0 = max(0, min(int(round(bbox.x)), width))
    y0 = max(0, min(int(round(bbox.y)), height))
    x1 = max(0, min(int(round(bbox.x + bbox.width)), width))
    y1 = max(0, min(int(round(bbox.y + bbox.height)), height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI is empty after clamping to the frame bounds")
    return y0, y1, x0, x1


class RoiGate(ImageStep):
    """Restrict a frame to a screen bounding box (P7)."""

    def __init__(self, bbox: ScreenBBox, mode: GateMode = GateMode.MASK) -> None:
        self._bbox = bbox
        self._mode = mode

    @property
    def bbox(self) -> ScreenBBox:
        return self._bbox

    @property
    def mode(self) -> GateMode:
        return self._mode

    def transform(self, image: Image) -> Image:
        height, width = image.shape[:2]
        y0, y1, x0, x1 = _pixel_bounds(self._bbox, height, width)
        if self._mode == GateMode.CROP:
            return image[y0:y1, x0:x1].copy()
        gated = np.zeros_like(image)
        gated[y0:y1, x0:x1] = image[y0:y1, x0:x1]
        return gated

    def __call__(self, frame: Frame) -> Frame:
        """Apply the gate, recording the crop origin on the frame (v0.2).

        CROP shifts the coordinate origin to the ROI's top-left, so the frame's
        ``crop_offset`` accumulates ``(x0, y0)`` — the remaining record of where the
        sub-image sits in the original capture, letting downstream map ROI-local
        detections back. MASK keeps capture-resolution coordinates, so the offset is
        untouched.
        """

        new_image = self.transform(frame.image)
        if self._mode != GateMode.CROP:
            return replace(frame, image=new_image)
        height, width = frame.image.shape[:2]
        y0, _, x0, _ = _pixel_bounds(self._bbox, height, width)
        offset_x, offset_y = frame.crop_offset
        return replace(frame, image=new_image, crop_offset=(offset_x + x0, offset_y + y0))
