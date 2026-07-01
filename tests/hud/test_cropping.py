"""Unit tests for crop + colour sampling (EPIC C — C1/C4)."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.hud.cropping import crop, sample_color_rgb


def test_crop_clamps_out_of_bounds_box_to_empty() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    empty = crop(image, ScreenBBox(x=50, y=50, width=5, height=5))
    assert empty.size == 0


def test_sample_color_returns_rgb_from_bgr_image() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    # Paint a red patch in BGR: blue=10, green=20, red=200.
    image[2:8, 2:8] = (10, 20, 200)
    rgb = sample_color_rgb(image, ScreenBBox(x=2, y=2, width=6, height=6))
    assert rgb == (200, 20, 10)


def test_sample_color_none_on_empty_crop() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert sample_color_rgb(image, ScreenBBox(x=50, y=50, width=1, height=1)) is None
