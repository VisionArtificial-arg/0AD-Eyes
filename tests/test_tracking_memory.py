"""G3 — memory: recently-lost tracks persist with rising staleness and decay."""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.tracking import IouTracker

from .conftest import make_frame


def _det(x: float = 0.0, y: float = 0.0, value: float = 1.0) -> Detection:
    return Detection(
        kind=EntityKind.UNIT,
        bbox=ScreenBBox(x=x, y=y, width=10, height=10),
        confidence=Confidence(value=value, provenance=Provenance.CLASSICAL),
    )


def _dets(frame_id: int, *detections: Detection) -> Detections:
    return Detections(frame_id=frame_id, items=detections)


def test_staleness_increments_each_missed_frame() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=5, decay=1.0)
    tracker.update(_dets(0, _det()), make_frame(0))

    stalenesses = []
    for f in range(1, 4):
        (entity,) = tracker.update(_dets(f), make_frame(f))
        stalenesses.append(entity.staleness)

    assert stalenesses == [1, 2, 3]


def test_confidence_decays_geometrically_while_lost() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=5, decay=0.5)
    tracker.update(_dets(0, _det(value=1.0)), make_frame(0))

    (miss1,) = tracker.update(_dets(1), make_frame(1))
    (miss2,) = tracker.update(_dets(2), make_frame(2))

    assert miss1.confidence.value == 0.5
    assert miss2.confidence.value == 0.25
    assert miss2.confidence.provenance is Provenance.CLASSICAL  # provenance preserved


def test_track_dies_once_memory_budget_is_exhausted() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=2, decay=1.0)
    tracker.update(_dets(0, _det()), make_frame(0))

    assert len(tracker.update(_dets(1), make_frame(1))) == 1  # staleness 1
    assert len(tracker.update(_dets(2), make_frame(2))) == 1  # staleness 2
    assert tracker.update(_dets(3), make_frame(3)) == ()  # staleness 3 > 2 → dead


def test_reobservation_resets_staleness_and_restores_confidence() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=5, decay=0.5)
    tracker.update(_dets(0, _det(value=1.0)), make_frame(0))
    tracker.update(_dets(1), make_frame(1))  # lost, staleness 1

    (entity,) = tracker.update(_dets(2, _det(value=0.9)), make_frame(2))
    assert entity.staleness == 0
    assert entity.confidence.value == 0.9
