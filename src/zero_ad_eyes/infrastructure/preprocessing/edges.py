"""Edge detection (P6, CV-07).

Edge maps feed the classical detection path — contours, template matching, HUD box
geometry. Two operators are offered:
- Canny: thin, connected binary edges (the usual contour input).
- Sobel: gradient-magnitude edges (softer, useful as a feature channel).

Both reduce colour to a single-channel edge map. Because the rest of EPIC P speaks
BGR frames, :class:`EdgeDetect` re-expands the map to 3 channels by default so it
stays composable in a pipeline; set ``as_bgr=False`` to keep it single-channel.
The bare array transforms are also exposed as :func:`canny_edges` /
:func:`sobel_edges` for callers that want the map without a ``Frame`` wrapper.
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from .base import Image, ImageStep


class EdgeOperator(Enum):
    """Supported edge operators."""

    CANNY = "canny"
    SOBEL = "sobel"


def _as_gray(image: Image) -> Image:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny_edges(image: Image, low: float = 100.0, high: float = 200.0) -> Image:
    """Return a single-channel Canny edge map (0/255)."""

    return cv2.Canny(_as_gray(image), low, high)


def sobel_edges(image: Image, ksize: int = 3) -> Image:
    """Return a single-channel Sobel gradient-magnitude edge map (uint8)."""

    gray = _as_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(gx, gy)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


class EdgeDetect(ImageStep):
    """Produce an edge map from a frame (P6).

    ``as_bgr`` (default) re-expands the single-channel map to 3 channels so the
    result remains a valid BGR ``Frame`` and composes with other steps.
    """

    def __init__(
        self,
        operator: EdgeOperator = EdgeOperator.CANNY,
        *,
        low: float = 100.0,
        high: float = 200.0,
        ksize: int = 3,
        as_bgr: bool = True,
    ) -> None:
        if operator == EdgeOperator.CANNY and low >= high:
            raise ValueError("Canny low threshold must be below high threshold")
        if operator == EdgeOperator.SOBEL and (ksize < 1 or ksize % 2 == 0):
            raise ValueError("Sobel ksize must be a positive odd integer")
        self._operator = operator
        self._low = low
        self._high = high
        self._ksize = ksize
        self._as_bgr = as_bgr

    def transform(self, image: Image) -> Image:
        if self._operator == EdgeOperator.CANNY:
            edge_map = canny_edges(image, self._low, self._high)
        else:
            edge_map = sobel_edges(image, self._ksize)
        if self._as_bgr:
            return cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        return edge_map
