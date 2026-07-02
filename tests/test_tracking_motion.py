"""G6 — optical flow (Farneback) and per-entity motion estimation (deterministic)."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox, ScreenPoint
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.tracking import (
    FarnebackMotionEstimator,
    IouTracker,
    Motion,
    motion_from_trajectory,
)

from .conftest import make_frame


def _det(x: float, y: float) -> Detection:
    return Detection(
        kind=EntityKind.UNIT,
        bbox=ScreenBBox(x=x, y=y, width=10, height=10),
        confidence=Confidence(value=1.0, provenance=Provenance.CLASSICAL),
    )


def _dets(frame_id: int, *detections: Detection) -> Detections:
    return Detections(frame_id=frame_id, items=detections)


def test_motion_from_trajectory_derives_direction_and_speed() -> None:
    points = [ScreenPoint(x=0, y=0), ScreenPoint(x=3, y=4)]
    motion = motion_from_trajectory(points)
    assert motion.dx == 3
    assert motion.dy == 4
    assert motion.speed == 5.0
    assert round(motion.direction_deg, 3) == round(53.13010235, 3)


def test_motion_still_when_trajectory_too_short() -> None:
    assert motion_from_trajectory([ScreenPoint(x=1, y=1)]) == Motion.still()


def test_tracker_reports_rightward_motion_for_moving_entity() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=15, decay=0.85)
    tracker.update(_dets(0, _det(0, 0)), make_frame(0))
    tracker.update(_dets(1, _det(5, 0)), make_frame(1))  # +5px in x, overlapping

    motions = tracker.motions()
    assert set(motions) == {0}
    assert motions[0].dx == 5.0
    assert motions[0].dy == 0.0
    assert motions[0].is_moving


def _textured_frame(frame_id: int, shift_x: int, size: int = 64) -> Frame:
    rng = np.random.default_rng(1234)
    base = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    canvas = np.zeros((size, size), dtype=np.uint8)
    # place the texture shifted horizontally so flow points in +x
    canvas[:, shift_x : shift_x + size - shift_x] = base[:, : size - shift_x]
    image = np.dstack([canvas, canvas, canvas])
    return Frame(
        image=image,
        meta=FrameMeta(
            frame_id=frame_id, timestamp=float(frame_id), source="test", width=size, height=size
        ),
    )


def test_farneback_recovers_horizontal_flow_sign() -> None:
    estimator = FarnebackMotionEstimator()
    prev = _textured_frame(0, shift_x=0)
    curr = _textured_frame(1, shift_x=4)

    motion = estimator.estimate(prev.image, curr.image)
    assert motion.dx > 0.0  # texture moved right → positive x flow
    assert abs(motion.dy) < abs(motion.dx)  # predominantly horizontal


def test_farneback_static_pair_is_near_still() -> None:
    estimator = FarnebackMotionEstimator()
    frame = _textured_frame(0, shift_x=0)
    motion = estimator.estimate(frame.image, frame.image)
    assert motion.speed < 1e-3
