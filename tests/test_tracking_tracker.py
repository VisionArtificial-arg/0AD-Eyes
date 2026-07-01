"""G1 — multi-object IoU tracker: stable ids + trajectories (deterministic)."""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.tracking import IouTracker
from zero_ad_eyes.infrastructure.tracking.association import greedy_match, iou

from .conftest import make_frame


def _det(x: float, y: float, *, kind: EntityKind = EntityKind.UNIT, w: float = 10.0) -> Detection:
    return Detection(
        kind=kind,
        bbox=ScreenBBox(x=x, y=y, width=w, height=w),
        confidence=Confidence(value=0.9, provenance=Provenance.CLASSICAL),
    )


def _dets(frame_id: int, *detections: Detection) -> Detections:
    return Detections(frame_id=frame_id, items=detections)


def test_iou_and_greedy_match_are_deterministic() -> None:
    a = ScreenBBox(x=0, y=0, width=10, height=10)
    b = ScreenBBox(x=5, y=0, width=10, height=10)
    assert iou(a, a) == 1.0
    assert 0.0 < iou(a, b) < 1.0
    assert iou(a, ScreenBBox(x=100, y=100, width=1, height=1)) == 0.0

    assignment = greedy_match([a], [b, ScreenBBox(x=200, y=200, width=10, height=10)])
    assert assignment.matches == ((0, 0),)
    assert assignment.unmatched_detections == (1,)


def test_single_object_keeps_stable_id_across_frames() -> None:
    tracker = IouTracker()
    e1 = tracker.update(_dets(0, _det(0, 0)), make_frame(0))
    e2 = tracker.update(_dets(1, _det(2, 0)), make_frame(1))  # small move, IoU overlaps
    e3 = tracker.update(_dets(2, _det(4, 0)), make_frame(2))

    assert len(e1) == len(e2) == len(e3) == 1
    assert e1[0].entity_id == e2[0].entity_id == e3[0].entity_id


def test_two_objects_get_distinct_stable_ids() -> None:
    tracker = IouTracker()
    left, right = _det(0, 0), _det(100, 0)
    first = tracker.update(_dets(0, left, right), make_frame(0))
    second = tracker.update(_dets(1, _det(2, 0), _det(102, 0)), make_frame(1))

    ids_first = {e.entity_id for e in first}
    ids_second = {e.entity_id for e in second}
    assert len(ids_first) == 2
    assert ids_first == ids_second


def test_new_detection_births_a_new_id() -> None:
    tracker = IouTracker()
    tracker.update(_dets(0, _det(0, 0)), make_frame(0))
    entities = tracker.update(_dets(1, _det(0, 0), _det(200, 0)), make_frame(1))
    assert len(entities) == 2
    assert {e.entity_id for e in entities} == {0, 1}


def test_motion_is_none_on_first_sighting_then_populated() -> None:
    tracker = IouTracker()
    (first,) = tracker.update(_dets(0, _det(0, 0)), make_frame(0))
    assert first.motion is None  # one observation → velocity unknown, not fabricated

    (second,) = tracker.update(_dets(1, _det(3, 0)), make_frame(1))  # moved +3 in x
    assert second.motion is not None
    assert second.motion.dx == 3.0
    assert second.motion.dy == 0.0
    assert second.motion.confidence.provenance is Provenance.CLASSICAL


def test_confidence_and_type_carry_through() -> None:
    tracker = IouTracker()
    (entity,) = tracker.update(_dets(0, _det(0, 0, kind=EntityKind.BUILDING)), make_frame(0))
    assert entity.kind is EntityKind.BUILDING
    assert entity.confidence.provenance is Provenance.CLASSICAL
    assert entity.confidence.value == 0.9
