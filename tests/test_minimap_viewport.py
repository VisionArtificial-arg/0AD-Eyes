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


def _detector() -> ViewportDetector:
    return ViewportDetector(white_min=200, min_area=64, min_side=8, approx_epsilon_fraction=0.02)


def test_extracts_the_camera_quad_corners() -> None:
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(region, (20, 25), (60, 55), (255, 255, 255), thickness=1)

    seg = _full_segmentation(region)
    proj = _projector(seg)
    quad = _detector().detect(seg, proj)

    assert quad is not None
    tl = proj.to_world(20.0, 25.0)
    tr = proj.to_world(60.0, 25.0)
    br = proj.to_world(60.0, 55.0)
    bl = proj.to_world(20.0, 55.0)
    assert abs(quad.top_left.x - tl.x) < 2.0 and abs(quad.top_left.y - tl.y) < 2.0
    assert abs(quad.top_right.x - tr.x) < 2.0 and abs(quad.top_right.y - tr.y) < 2.0
    assert abs(quad.bottom_right.x - br.x) < 2.0 and abs(quad.bottom_right.y - br.y) < 2.0
    assert abs(quad.bottom_left.x - bl.x) < 2.0 and abs(quad.bottom_left.y - bl.y) < 2.0


def test_extracts_a_perspective_trapezoid() -> None:
    # A tilted-camera footprint: the far (top) edge is narrower than the near edge.
    # A general quad must preserve this; an axis-aligned bbox would not.
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    trapezoid = np.array([[30, 20], [50, 20], [65, 60], [15, 60]], dtype=np.int32)
    cv2.polylines(region, [trapezoid], True, (255, 255, 255), thickness=1)

    seg = _full_segmentation(region)
    proj = _projector(seg)
    quad = _detector().detect(seg, proj)

    assert quad is not None
    top_width = abs(quad.top_right.x - quad.top_left.x)
    bottom_width = abs(quad.bottom_right.x - quad.bottom_left.x)
    assert top_width < bottom_width  # foreshortening preserved


def test_no_viewport_returns_none() -> None:
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    seg = _full_segmentation(region)
    assert _detector().detect(seg, _projector(seg)) is None


def test_small_white_blip_is_not_a_viewport() -> None:
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(region, (40, 40), 2, (255, 255, 255), thickness=-1)
    seg = _full_segmentation(region)
    assert _detector().detect(seg, _projector(seg)) is None
