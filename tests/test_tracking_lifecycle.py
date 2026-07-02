"""G2 — track lifecycle: birth (tentative→confirmed) and death on disappearance."""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.tracking import IouTracker, TrackStatus

from .conftest import make_frame


def _det(x: float, y: float) -> Detection:
    return Detection(
        kind=EntityKind.UNIT,
        bbox=ScreenBBox(x=x, y=y, width=10, height=10),
        confidence=Confidence(value=0.8, provenance=Provenance.CLASSICAL),
    )


def _dets(frame_id: int, *detections: Detection) -> Detections:
    return Detections(frame_id=frame_id, items=detections)


def test_birth_is_tentative_then_confirmed_after_min_hits() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=2, max_staleness=15, decay=0.85)
    tracker.update(_dets(0, _det(0, 0)), make_frame(0))
    assert tracker.statuses() == {0: TrackStatus.TENTATIVE}

    tracker.update(_dets(1, _det(1, 0)), make_frame(1))
    assert tracker.statuses() == {0: TrackStatus.CONFIRMED}


def test_disappearance_kills_the_track_with_zero_memory() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=0, decay=0.85)
    born = tracker.update(_dets(0, _det(0, 0)), make_frame(0))
    assert len(born) == 1

    after_gone = tracker.update(_dets(1), make_frame(1))  # nothing detected
    assert after_gone == ()
    assert tracker.statuses() == {}


def test_single_missed_frame_does_not_kill_when_memory_allows() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=2, decay=0.85)
    tracker.update(_dets(0, _det(0, 0)), make_frame(0))
    lost = tracker.update(_dets(1), make_frame(1))

    assert len(lost) == 1
    assert lost[0].entity_id == 0
    assert tracker.statuses() == {0: TrackStatus.LOST}


def test_reacquired_track_returns_to_confirmed_with_same_id() -> None:
    tracker = IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=3, decay=0.85)
    tracker.update(_dets(0, _det(0, 0)), make_frame(0))
    tracker.update(_dets(1), make_frame(1))  # missed → LOST
    reacquired = tracker.update(_dets(2, _det(0, 0)), make_frame(2))

    assert reacquired[0].entity_id == 0
    assert tracker.statuses() == {0: TrackStatus.CONFIRMED}
