"""E5 state-cue tests — synthetic selection rings, scaffolds, garrison badges."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.perception.state import (
    detect_construction,
    detect_garrison,
    detect_selection,
    read_state_cues,
)

BOX = ScreenBBox(x=30.0, y=30.0, width=40.0, height=40.0)


def _wrap(image: np.ndarray) -> Frame:
    h, w = image.shape[:2]
    return Frame(
        image=image,
        meta=FrameMeta(frame_id=1, timestamp=1.0, source="test", width=w, height=h),
    )


def _base() -> np.ndarray:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), (120, 60, 40), -1)  # a muted entity body
    return img


def test_selection_ring_detected() -> None:
    img = _base()
    cv2.rectangle(img, (29, 29), (71, 71), (255, 255, 255), 2)  # bright ring on border
    assert detect_selection(_wrap(img), BOX) is True


def test_no_selection_ring() -> None:
    assert detect_selection(_wrap(_base()), BOX) is False


def test_construction_scaffold_detected() -> None:
    img = _base()
    # A wireframe lattice inside the box → high internal edge density.
    for x in range(32, 70, 4):
        cv2.line(img, (x, 30), (x, 70), (200, 200, 120), 1)
    for y in range(32, 70, 4):
        cv2.line(img, (30, y), (70, y), (200, 200, 120), 1)
    assert detect_construction(_wrap(img), BOX) is True


def test_solid_building_not_under_construction() -> None:
    assert detect_construction(_wrap(_base()), BOX) is False


def test_garrison_badge_detected() -> None:
    img = _base()
    cv2.rectangle(img, (34, 32), (42, 40), (245, 245, 245), -1)  # small white badge
    assert detect_garrison(_wrap(img), BOX) is True


def test_no_garrison_badge() -> None:
    assert detect_garrison(_wrap(_base()), BOX) is False


def test_read_state_cues_bundles_all() -> None:
    cues = read_state_cues(_wrap(_base()), BOX)
    assert cues.selected is False
    assert cues.garrisoned is False
