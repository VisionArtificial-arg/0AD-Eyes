"""Color-space transforms (P2, CV-03).

0 A.D. encodes meaning in colour: player colours, minimap factions, resource-node
tints, HUD chrome. Separating those cues is far easier outside BGR — HSV isolates
hue from lighting, Lab isolates perceptual lightness from chroma. This step is a
reusable, named conversion between the spaces the rest of EPIC P cares about.

The step is stateless apart from its chosen conversion, so one instance is safe to
share across frames and pipelines.
"""

from __future__ import annotations

from enum import Enum

import cv2

from .base import Image, ImageStep


class ColorSpace(Enum):
    """The colour spaces this layer converts between."""

    BGR = "BGR"
    RGB = "RGB"
    HSV = "HSV"
    LAB = "LAB"


# OpenCV conversion codes, keyed by (source, target). Only the pairs the
# perception stages actually use are listed; unknown pairs fail loudly.
_CONVERSIONS: dict[tuple[ColorSpace, ColorSpace], int] = {
    (ColorSpace.BGR, ColorSpace.RGB): cv2.COLOR_BGR2RGB,
    (ColorSpace.BGR, ColorSpace.HSV): cv2.COLOR_BGR2HSV,
    (ColorSpace.BGR, ColorSpace.LAB): cv2.COLOR_BGR2LAB,
    (ColorSpace.RGB, ColorSpace.BGR): cv2.COLOR_RGB2BGR,
    (ColorSpace.RGB, ColorSpace.HSV): cv2.COLOR_RGB2HSV,
    (ColorSpace.RGB, ColorSpace.LAB): cv2.COLOR_RGB2LAB,
    (ColorSpace.HSV, ColorSpace.BGR): cv2.COLOR_HSV2BGR,
    (ColorSpace.HSV, ColorSpace.RGB): cv2.COLOR_HSV2RGB,
    (ColorSpace.LAB, ColorSpace.BGR): cv2.COLOR_LAB2BGR,
    (ColorSpace.LAB, ColorSpace.RGB): cv2.COLOR_LAB2RGB,
}


class ColorSpaceConvert(ImageStep):
    """Convert an image between two colour spaces (P2).

    ``source`` defaults to BGR — the capture/``Frame`` convention. Converting to a
    non-BGR space and back is lossy only through rounding, so round-trips are close
    but not bit-identical; callers that need to re-enter BGR should add an explicit
    back-conversion step.
    """

    def __init__(self, target: ColorSpace, source: ColorSpace = ColorSpace.BGR) -> None:
        if source == target:
            raise ValueError("source and target colour spaces are identical")
        key = (source, target)
        if key not in _CONVERSIONS:
            raise ValueError(f"unsupported conversion {source.value} -> {target.value}")
        self._source = source
        self._target = target
        self._code = _CONVERSIONS[key]

    @property
    def source(self) -> ColorSpace:
        return self._source

    @property
    def target(self) -> ColorSpace:
        return self._target

    def transform(self, image: Image) -> Image:
        return cv2.cvtColor(image, self._code)
