"""D1 — Segment the minimap disc/region from calibration (REQUIREMENTS.md EPIC D).

The ``Calibration.minimap`` bounding box (EPIC B) tells us *where* the minimap sits
in the frame; this stage crops that region and produces a mask of the **active play
area** inside it, so later stages never mistake the surrounding HUD chrome (frame,
buttons) for map content.

0 A.D.'s minimap is square, so :attr:`MinimapShape.SQUARE` (the whole crop) is the
usual choice; :attr:`MinimapShape.DISC` inscribes a circle for skins/mods that render
a round minimap. The shape is supplied per session from the ``minimap`` config
(``disc_shape``), never assumed in code (NF7).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from zero_ad_eyes.domain.geometry import ScreenBBox


class MinimapShape(Enum):
    """Shape of the active minimap area within its bounding box."""

    SQUARE = "square"
    DISC = "disc"


@dataclass(frozen=True)
class Segmentation:
    """The cropped minimap region plus a mask of its active play area."""

    region: np.ndarray  # HxWx3 BGR crop
    mask: np.ndarray  # HxW uint8, 255 = active play area
    origin_x: int  # crop offset in the source frame
    origin_y: int

    @property
    def height(self) -> int:
        return int(self.region.shape[0])

    @property
    def width(self) -> int:
        return int(self.region.shape[1])


class MinimapSegmenter:
    """Crops the calibrated minimap region and masks its active area (D1)."""

    def __init__(self, shape: MinimapShape) -> None:
        self._shape = shape

    def segment(self, image: np.ndarray, bbox: ScreenBBox) -> Segmentation | None:
        """Crop ``bbox`` from ``image``; return ``None`` if it lands off-frame."""

        frame_h, frame_w = image.shape[:2]
        x0 = max(0, int(round(bbox.x)))
        y0 = max(0, int(round(bbox.y)))
        x1 = min(frame_w, int(round(bbox.x + bbox.width)))
        y1 = min(frame_h, int(round(bbox.y + bbox.height)))
        if x1 <= x0 or y1 <= y0:
            return None

        region = np.ascontiguousarray(image[y0:y1, x0:x1])
        mask = self._build_mask(region.shape[0], region.shape[1])
        return Segmentation(region=region, mask=mask, origin_x=x0, origin_y=y0)

    def _build_mask(self, height: int, width: int) -> np.ndarray:
        if self._shape is MinimapShape.DISC:
            mask = np.zeros((height, width), dtype=np.uint8)
            center = (width // 2, height // 2)
            radius = min(width, height) // 2
            cv2.circle(mask, center, radius, 255, thickness=-1)
            return mask
        return np.full((height, width), 255, dtype=np.uint8)
