"""EVT-6 — system events: calibration, read failures, dropped frames, silent detector."""

from __future__ import annotations

from zero_ad_eyes.application.frames import PerceptionContext
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.events import Event, EventKind, EventOrigin
from zero_ad_eyes.domain.hud import HudState
from zero_ad_eyes.domain.minimap import MinimapModel
from zero_ad_eyes.domain.taxonomy import EntityKind, ResourceType
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel
from zero_ad_eyes.infrastructure.contract.system_events import SystemEventSource

READ = HudState(
    stockpiles={ResourceType.FOOD: 100},
    confidence=Confidence(value=0.9, provenance=Provenance.CLASSICAL),
)
UNIT = Entity(entity_id=1, kind=EntityKind.UNIT)


def _wm(
    frame_id: int, *, hud: HudState | None = READ, minimap: bool = True, units: int = 0
) -> WorldModel:
    meta = FrameMeta(frame_id=frame_id, timestamp=float(frame_id), source="t", width=64, height=48)
    return WorldModel(
        meta=meta,
        hud=hud,
        minimap=MinimapModel() if minimap else None,
        entities=tuple(UNIT.model_copy(update={"entity_id": i}) for i in range(units)),
    )


def _kinds(events: tuple[Event, ...]) -> list[EventKind]:
    return [e.kind for e in events]


def _source(silent_frames: int = 3) -> SystemEventSource:
    return SystemEventSource(low_confidence=0.5, silent_frames=silent_frames)


def test_calibration_drift_and_recalibration_come_from_the_context() -> None:
    events = _source().events(_wm(0), PerceptionContext(recalibrated=True, layout_drift=True))
    assert _kinds(events) == [EventKind.LAYOUT_DRIFT, EventKind.RECALIBRATED]
    assert events[1].details == {"width": 64, "height": 48}
    assert all(e.origin is EventOrigin.SYSTEM for e in events)


def test_hud_failure_is_reported_once_and_its_recovery_too() -> None:
    source = _source()
    frames = [
        source.events(_wm(i, hud=h), PerceptionContext())
        for i, h in enumerate([READ, None, None, READ])
    ]
    assert [_kinds(f) for f in frames] == [
        [],
        [EventKind.HUD_READ_FAILED],
        [],
        [EventKind.HUD_READ_RESTORED],
    ]


def test_an_unreadable_hud_from_the_start_is_reported() -> None:
    assert _kinds(_source().events(_wm(0, hud=HudState()), PerceptionContext())) == [
        EventKind.HUD_READ_FAILED
    ]


def test_low_confidence_hud_reads() -> None:
    shaky = READ.model_copy(
        update={"confidence": Confidence(value=0.2, provenance=Provenance.CLASSICAL)}
    )
    source = _source()
    frames = [
        source.events(_wm(i, hud=h), PerceptionContext())
        for i, h in enumerate([READ, shaky, shaky, READ, shaky])
    ]
    assert [_kinds(f) for f in frames] == [
        [],
        [EventKind.HUD_LOW_CONFIDENCE],
        [],
        [],
        [EventKind.HUD_LOW_CONFIDENCE],
    ]


def test_minimap_failure_and_recovery() -> None:
    source = _source()
    frames = [
        source.events(_wm(i, minimap=m), PerceptionContext())
        for i, m in enumerate([True, False, True])
    ]
    assert [_kinds(f) for f in frames] == [
        [],
        [EventKind.MINIMAP_READ_FAILED],
        [EventKind.MINIMAP_READ_RESTORED],
    ]


def test_dropped_frames_are_jumps_beyond_the_usual_step() -> None:
    source = _source()
    events = [source.events(_wm(i), PerceptionContext()) for i in (0, 10, 20, 50, 60)]
    assert [_kinds(f) for f in events] == [[], [], [], [EventKind.FRAMES_DROPPED], []]
    assert events[3][0].details == {"missed": 2, "step": 30, "usual_step": 10}


def test_detector_silence_after_detections_and_its_return() -> None:
    source = _source(silent_frames=2)
    counts = [0, 0, 1, 0, 0, 0, 2]
    frames = [
        _kinds(source.events(_wm(i, units=n), PerceptionContext())) for i, n in enumerate(counts)
    ]
    assert frames == [
        [],
        [],
        [],
        [],
        [EventKind.DETECTIONS_STOPPED],
        [],
        [EventKind.DETECTIONS_RESUMED],
    ]
