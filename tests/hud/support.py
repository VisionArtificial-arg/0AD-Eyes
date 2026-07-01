"""Test doubles for HUD OCR — no tesseract binary required.

``MarkerOcr`` is an injectable :class:`OcrEngine` that returns canned text keyed
by a marker value painted into a region's blue channel. This lets a single
synthetic frame drive many distinct OCR readings deterministically, regardless of
the order in which the reader crops regions, without any real OCR.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.hud.layout import FractionalRegion


class MarkerOcr:
    """Return canned text based on a marker painted into channel 0 of the crop."""

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)
        self.calls: list[int] = []

    def read_text(self, image: Any) -> str:
        if getattr(image, "size", 0) == 0:
            return ""
        marker = int(image[0, 0, 0])
        self.calls.append(marker)
        return self._texts.get(marker, "")


def paint(image: Any, bbox: ScreenBBox, marker: int) -> None:
    """Fill ``bbox`` in channel 0 with ``marker`` (1..255)."""

    x0, y0 = int(round(bbox.x)), int(round(bbox.y))
    x1, y1 = int(round(bbox.x + bbox.width)), int(round(bbox.y + bbox.height))
    image[y0:y1, x0:x1, 0] = marker


def build_hud(
    parent: ScreenBBox,
    regions: list[tuple[FractionalRegion, str]],
    *,
    width: int = 240,
    height: int = 24,
) -> tuple[Frame, MarkerOcr]:
    """Paint markers for each (region, text) pair; return a frame + matching OCR.

    Markers are assigned 1, 2, 3, … so the black background (0) maps to ``""``.
    """

    image = np.zeros((height, width, 3), dtype=np.uint8)
    texts: dict[int, str] = {}
    for index, (region, text) in enumerate(regions, start=1):
        paint(image, region.project(parent), index)
        texts[index] = text
    frame = Frame(
        image=image,
        meta=FrameMeta(frame_id=0, timestamp=0.0, source="test", width=width, height=height),
    )
    return frame, MarkerOcr(texts)
