"""D1 — MinimapSegmenter crop + active-area mask tests."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.minimap import MinimapSegmenter, MinimapShape


def test_crops_the_calibrated_region_with_offset() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[150:190, 10:70] = (255, 0, 0)  # a blue block where the minimap sits
    bbox = ScreenBBox(x=10, y=150, width=60, height=40)

    seg = MinimapSegmenter(MinimapShape.SQUARE).segment(frame, bbox)

    assert seg is not None
    assert (seg.width, seg.height) == (60, 40)
    assert (seg.origin_x, seg.origin_y) == (10, 150)
    assert bool((seg.region == (255, 0, 0)).all())


def test_square_mask_is_fully_active() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    seg = MinimapSegmenter(MinimapShape.SQUARE).segment(
        frame, ScreenBBox(x=0, y=0, width=40, height=40)
    )
    assert seg is not None
    assert int(seg.mask.min()) == 255


def test_disc_mask_excludes_corners_keeps_centre() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    seg = MinimapSegmenter(MinimapShape.DISC).segment(
        frame, ScreenBBox(x=0, y=0, width=40, height=40)
    )
    assert seg is not None
    assert int(seg.mask[20, 20]) == 255  # centre inside the disc
    assert int(seg.mask[0, 0]) == 0  # corner outside the disc


def test_off_frame_bbox_returns_none() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert (
        MinimapSegmenter(MinimapShape.SQUARE).segment(
            frame, ScreenBBox(x=100, y=100, width=10, height=10)
        )
        is None
    )


def test_bbox_is_clamped_to_frame_bounds() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    seg = MinimapSegmenter(MinimapShape.SQUARE).segment(
        frame, ScreenBBox(x=40, y=40, width=40, height=40)
    )
    assert seg is not None
    assert (seg.width, seg.height) == (10, 10)
