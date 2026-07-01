"""E3 ownership segmentation tests — coloured rectangles as player entities."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.perception.ownership import assign_ownership, ownership_mask
from zero_ad_eyes.infrastructure.perception.palette import DEFAULT_PALETTE


def _frame(width: int = 120, height: int = 90) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _wrap(image: np.ndarray) -> Frame:
    h, w = image.shape[:2]
    return Frame(
        image=image,
        meta=FrameMeta(frame_id=1, timestamp=1.0, source="test", width=w, height=h),
    )


# BGR fills that land squarely inside the default HSV bands.
BLUE = (200, 40, 40)
GREEN = (40, 200, 40)
RED = (40, 40, 200)


def test_blue_rectangle_is_self() -> None:
    img = _frame()
    cv2.rectangle(img, (10, 10), (50, 50), BLUE, -1)
    owner, frac = assign_ownership(_wrap(img), ScreenBBox(x=10, y=10, width=40, height=40))
    assert owner is Ownership.SELF
    assert frac > 0.8


def test_red_rectangle_is_enemy_across_hue_wrap() -> None:
    img = _frame()
    cv2.rectangle(img, (10, 10), (50, 50), RED, -1)
    owner, _ = assign_ownership(_wrap(img), ScreenBBox(x=10, y=10, width=40, height=40))
    assert owner is Ownership.ENEMY


def test_green_rectangle_is_ally() -> None:
    img = _frame()
    cv2.rectangle(img, (10, 10), (50, 50), GREEN, -1)
    owner, _ = assign_ownership(_wrap(img), ScreenBBox(x=10, y=10, width=40, height=40))
    assert owner is Ownership.ALLY


def test_black_box_is_unknown() -> None:
    img = _frame()
    owner, frac = assign_ownership(_wrap(img), ScreenBBox(x=10, y=10, width=40, height=40))
    assert owner is Ownership.UNKNOWN
    assert frac == 0.0


def test_robust_to_brightness_variation() -> None:
    # A dim (shadowed) blue still classifies as SELF because hue is preserved.
    img = _frame()
    cv2.rectangle(img, (10, 10), (50, 50), (110, 25, 25), -1)
    owner, _ = assign_ownership(_wrap(img), ScreenBBox(x=10, y=10, width=40, height=40))
    assert owner is Ownership.SELF


def test_ownership_mask_covers_expected_region() -> None:
    img = _frame()
    cv2.rectangle(img, (10, 10), (50, 50), BLUE, -1)
    blue_color = next(c for c in DEFAULT_PALETTE.colors if c.name == "blue")
    mask = ownership_mask(_wrap(img), blue_color)
    assert mask[30, 30] == 255
    assert mask[80, 100] == 0
