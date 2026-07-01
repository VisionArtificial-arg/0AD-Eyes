"""Optical flow & motion estimation (REQUIREMENTS.md G6, CV-17/CV-18).

Two complementary routes to per-entity direction/speed, both expressed as the
``Motion`` value object (screen-pixel space, y growing downward):

- *trajectory motion* (:func:`motion_from_trajectory`): the displacement of a
  track's centroid between its last two observations. Deterministic, pixel-free,
  and driven purely by the tracker's own history (CV-18).
- *dense optical flow* (:class:`FarnebackMotionEstimator`): cv2 Farneback dense
  flow between two frames, averaged over a region of interest (CV-17). Also usable
  to drive ROI gating. This is the only cv2-dependent piece here.

Both are infrastructure helpers; neither leaks into the domain. ``Motion`` stays a
local value object because the emitted ``Entity`` contract carries position, not
velocity — motion is a tracking-internal derivation exposed alongside entities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from zero_ad_eyes.domain.geometry import ScreenBBox, ScreenPoint


class Motion(BaseModel):
    """A per-frame velocity in screen pixels, with derived speed and heading."""

    model_config = ConfigDict(frozen=True)

    dx: float
    dy: float

    @property
    def speed(self) -> float:
        """Pixels moved per frame."""

        return math.hypot(self.dx, self.dy)

    @property
    def direction_deg(self) -> float:
        """Heading in degrees, ``atan2(dy, dx)`` (0 = +x/right, 90 = +y/down)."""

        return math.degrees(math.atan2(self.dy, self.dx))

    @property
    def is_moving(self) -> bool:
        return self.speed > 1e-9

    @classmethod
    def still(cls) -> Motion:
        return cls(dx=0.0, dy=0.0)


def motion_from_trajectory(trajectory: Sequence[ScreenPoint]) -> Motion:
    """Velocity from the last two centroids of a track (CV-18). Still if too short."""

    if len(trajectory) < 2:
        return Motion.still()
    prev, last = trajectory[-2], trajectory[-1]
    return Motion(dx=last.x - prev.x, dy=last.y - prev.y)


class FarnebackMotionEstimator:
    """Dense optical flow via ``cv2.calcOpticalFlowFarneback`` (CV-17)."""

    def __init__(
        self,
        *,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        self._pyr_scale = pyr_scale
        self._levels = levels
        self._winsize = winsize
        self._iterations = iterations
        self._poly_n = poly_n
        self._poly_sigma = poly_sigma

    def flow(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Dense HxWx2 flow field mapping ``previous`` pixels to ``current``."""

        prev_gray = self._to_gray(previous)
        curr_gray = self._to_gray(current)
        # cv2 accepts ``None`` for the optional out-flow buffer at runtime, but the
        # bundled type stub only lists an ndarray overload for it.
        result: np.ndarray = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
            prev_gray,
            curr_gray,
            None,
            self._pyr_scale,
            self._levels,
            self._winsize,
            self._iterations,
            self._poly_n,
            self._poly_sigma,
            0,
        )
        return result

    def estimate(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        roi: ScreenBBox | None = None,
    ) -> Motion:
        """Mean flow over ``roi`` (whole frame if ``None``) as a :class:`Motion`."""

        field = self.flow(previous, current)
        if roi is not None:
            x0 = max(0, int(math.floor(roi.x)))
            y0 = max(0, int(math.floor(roi.y)))
            x1 = min(field.shape[1], int(math.ceil(roi.x + roi.width)))
            y1 = min(field.shape[0], int(math.ceil(roi.y + roi.height)))
            field = field[y0:y1, x0:x1]
        if field.size == 0:
            return Motion.still()
        mean = field.reshape(-1, 2).mean(axis=0)
        return Motion(dx=float(mean[0]), dy=float(mean[1]))

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
