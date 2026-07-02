"""E3 — Ownership assignment via player-colour segmentation.

Given a detection's bounding box, decide whose it is by segmenting the crop in
HSV against a configurable ``PlayerPalette`` and taking the dominant player
colour. Returns the ``Ownership`` plus the fraction of the box that colour
covered, which callers fold into a ``Confidence`` (classical provenance).

Robustness to lighting/terrain/shadow comes from the palette living in HSV with
generous value windows (see ``palette``); this module only does the counting.
"""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import Ownership

from .palette import PlayerColor, PlayerPalette


def _crop(image: np.ndarray, bbox: ScreenBBox) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, int(bbox.x))
    y0 = max(0, int(bbox.y))
    x1 = min(w, int(bbox.x + bbox.width))
    y1 = min(h, int(bbox.y + bbox.height))
    return image[y0:y1, x0:x1]


def ownership_mask(frame: Frame, color: PlayerColor) -> np.ndarray:
    """Full-frame binary mask of pixels matching one player colour."""

    hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
    return color.mask(hsv)


def assign_ownership(
    frame: Frame,
    bbox: ScreenBBox,
    palette: PlayerPalette,
    min_fraction: float,
) -> tuple[Ownership, float]:
    """Dominant player colour inside ``bbox`` → (ownership, coverage fraction).

    If no colour covers at least ``min_fraction`` of the box the result is
    ``Ownership.UNKNOWN`` with fraction ``0.0`` — an honest "cannot tell", rather
    than a coin-flip.
    """

    crop = _crop(frame.image, bbox)
    if crop.size == 0:
        return Ownership.UNKNOWN, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = float(crop.shape[0] * crop.shape[1])

    best_owner = Ownership.UNKNOWN
    best_fraction = 0.0
    for color in palette.colors:
        matched = int(cv2.countNonZero(color.mask(hsv)))
        fraction = matched / total if total > 0 else 0.0
        if fraction > best_fraction:
            best_fraction = fraction
            best_owner = color.ownership

    if best_fraction < min_fraction:
        return Ownership.UNKNOWN, 0.0
    return best_owner, best_fraction
