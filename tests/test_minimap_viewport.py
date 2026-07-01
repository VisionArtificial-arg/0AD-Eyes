"""D5 — ViewportDetector camera-rectangle extraction tests (synthetic minimap)."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.infrastructure.minimap import (
    MinimapProjector,
    Segmentation,
    ViewportDetector,
    WorldExtent,
)


def _full_segmentation(region: np.ndarray) -> Segmentation:
    h, w = region.shape[:2]
    return Segmentation(
        region=region, mask=np.full((h, w), 255, dtype=np.uint8), origin_x=0, origin_y=0
    )


def _projector(seg: Segmentation) -> MinimapProjector:
    return MinimapProjector(
        region_width=seg.width, region_height=seg.height, extent=WorldExtent.square(100.0)
    )


def test_extracts_the_camera_rectangle_corners() -> None:
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(region, (20, 25), (60, 55), (255, 255, 255), thickness=1)

    seg = _full_segmentation(region)
    proj = _projector(seg)
    rect = ViewportDetector().detect(seg, proj)

    assert rect is not None
    tl = proj.to_world(20.0, 25.0)
    br = proj.to_world(60.0, 55.0)
    assert abs(rect.top_left.x - tl.x) < 2.0
    assert abs(rect.top_left.y - tl.y) < 2.0
    assert abs(rect.bottom_right.x - br.x) < 2.0
    assert abs(rect.bottom_right.y - br.y) < 2.0


def test_no_viewport_returns_none() -> None:
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    seg = _full_segmentation(region)
    assert ViewportDetector().detect(seg, _projector(seg)) is None


def test_small_white_blip_is_not_a_viewport() -> None:
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(region, (40, 40), 2, (255, 255, 255), thickness=-1)
    seg = _full_segmentation(region)
    assert ViewportDetector().detect(seg, _projector(seg)) is None
