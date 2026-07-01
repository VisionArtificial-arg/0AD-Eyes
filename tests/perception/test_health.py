"""E4 health-bar reading tests — synthetic bars above entity boxes."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.perception.health import (
    locate_health_bar,
    measure_fill,
    read_health,
)

ENTITY = ScreenBBox(x=20.0, y=30.0, width=40.0, height=40.0)


def _wrap(image: np.ndarray) -> Frame:
    h, w = image.shape[:2]
    return Frame(
        image=image,
        meta=FrameMeta(frame_id=1, timestamp=1.0, source="test", width=w, height=h),
    )


def _scene(fill_fraction: float, bar_y: int = 24) -> Frame:
    img = np.zeros((90, 120, 3), dtype=np.uint8)
    # The entity itself (a blue block).
    cv2.rectangle(img, (20, 30), (60, 70), (200, 40, 40), -1)
    # Bar background (dark), spanning the entity width.
    cv2.rectangle(img, (20, bar_y), (60, bar_y + 3), (25, 25, 25), -1)
    # Filled portion: bright green, left-to-right.
    filled_px = int(round(40 * fill_fraction))
    if filled_px > 0:
        cv2.rectangle(img, (20, bar_y), (20 + filled_px, bar_y + 3), (0, 200, 0), -1)
    return _wrap(img)


def test_reads_full_health() -> None:
    assert read_health(_scene(1.0), ENTITY) == 1.0


def test_reads_partial_health() -> None:
    val = read_health(_scene(0.5), ENTITY)
    assert val is not None
    assert abs(val - 0.5) <= 0.1


def test_reads_low_health() -> None:
    val = read_health(_scene(0.25), ENTITY)
    assert val is not None
    assert abs(val - 0.25) <= 0.1


def test_no_bar_returns_none() -> None:
    img = np.zeros((90, 120, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 30), (60, 70), (200, 40, 40), -1)
    assert read_health(_wrap(img), ENTITY) is None


def test_locate_then_measure_directly() -> None:
    scene = _scene(0.75)
    bar = locate_health_bar(scene, ENTITY)
    assert bar is not None
    assert bar.width == ENTITY.width
    frac = measure_fill(scene, bar)
    assert abs(frac - 0.75) <= 0.1
