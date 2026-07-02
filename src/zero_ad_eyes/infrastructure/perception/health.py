"""E4 — Health-bar reading → health fraction.

An entity's health bar is a thin horizontal strip above it: a bright coloured
*filled* portion (green→red as health drops) over a dark background, filling
left-to-right. Reading it is two steps:

1. ``locate_health_bar`` — scan the band just above the entity for the row that
   looks most like a bar (a wide run of bright, saturated pixels).
2. ``measure_fill`` — over that bar's width, the fraction of columns that carry
   a bright pixel is the health fraction.

This is deliberately art-agnostic (no exact sprite templates), trading precision
for robustness — a classical, ``Provenance.CLASSICAL`` estimate. Its known blind
spot: a (near-)empty bar has almost no bright pixels, so it may read as "no bar
found" (``None``) rather than 0.0; callers treat ``None`` as "unknown".
"""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox


def _bright_saturated(hsv: np.ndarray, s_min: int, v_min: int) -> np.ndarray:
    lo = np.array([0, s_min, v_min], dtype=np.uint8)
    hi = np.array([179, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def locate_health_bar(
    frame: Frame,
    entity_bbox: ScreenBBox,
    max_offset: int = 20,
    s_min: int = 60,
    v_min: int = 60,
    min_run: float = 0.15,
) -> ScreenBBox | None:
    """Find the health bar in the band above ``entity_bbox``.

    The scan stays strictly *above* the entity (bars sit over an entity's top
    edge) so the entity's own body cannot masquerade as a bar. Returns a bar
    bbox spanning the entity's width at the detected row, or ``None`` when no
    sufficiently wide bright run is present. ``min_run`` is the minimum fraction
    of the width that must be bright for a row to count as a bar — it also sets
    the lowest health this heuristic can still detect.
    """

    image = frame.image
    h, w = image.shape[:2]
    x0 = max(0, int(entity_bbox.x))
    x1 = min(w, int(entity_bbox.x + entity_bbox.width))
    y_top = max(0, int(entity_bbox.y) - max_offset)
    y_bot = min(h, int(entity_bbox.y))
    if x1 <= x0 or y_bot <= y_top:
        return None

    band = image[y_top:y_bot, x0:x1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = _bright_saturated(hsv, s_min, v_min)
    width = mask.shape[1]
    row_counts = (mask > 0).sum(axis=1)
    threshold = max(1.0, min_run * width)

    bar_rows = np.where(row_counts >= threshold)[0]
    if bar_rows.size == 0:
        return None

    peak = int(bar_rows[np.argmax(row_counts[bar_rows])])
    # Grow contiguously around the peak to capture the bar's height.
    top = peak
    while top - 1 in set(bar_rows.tolist()):
        top -= 1
    bottom = peak
    while bottom + 1 in set(bar_rows.tolist()):
        bottom += 1

    return ScreenBBox(
        x=float(x0),
        y=float(y_top + top),
        width=float(x1 - x0),
        height=float(bottom - top + 1),
    )


def measure_fill(frame: Frame, bar_bbox: ScreenBBox, s_min: int = 60, v_min: int = 60) -> float:
    """Fraction of the bar's columns carrying a bright pixel, clamped to [0, 1]."""

    image = frame.image
    h, w = image.shape[:2]
    x0 = max(0, int(bar_bbox.x))
    y0 = max(0, int(bar_bbox.y))
    x1 = min(w, int(bar_bbox.x + bar_bbox.width))
    y1 = min(h, int(bar_bbox.y + bar_bbox.height))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = _bright_saturated(hsv, s_min, v_min)
    filled_cols = int(np.count_nonzero(mask.max(axis=0)))
    total_cols = mask.shape[1]
    if total_cols == 0:
        return 0.0
    return max(0.0, min(1.0, filled_cols / total_cols))


def read_health(
    frame: Frame,
    entity_bbox: ScreenBBox,
    max_offset: int = 20,
    s_min: int = 60,
    v_min: int = 60,
    min_run: float = 0.15,
) -> float | None:
    """Locate the health bar above ``entity_bbox`` and return its fill fraction.

    ``None`` means no bar was found (health unknown), distinct from a found bar
    that reads 0.0.
    """

    bar = locate_health_bar(
        frame, entity_bbox, max_offset=max_offset, s_min=s_min, v_min=v_min, min_run=min_run
    )
    if bar is None:
        return None
    return measure_fill(frame, bar, s_min=s_min, v_min=v_min)
