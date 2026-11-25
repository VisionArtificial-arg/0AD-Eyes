"""EVT-2 — one event pipeline: every source sees the frame's world model + context."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.application.frames import Frame, PerceptionContext
from zero_ad_eyes.application.pipeline import PerceptionPipeline
from zero_ad_eyes.application.ports import EventSource
from zero_ad_eyes.domain.calibration import Calibration, CalibrationCheck
from zero_ad_eyes.domain.events import Event, EventKind, EventOrigin
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel
from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource
from zero_ad_eyes.infrastructure.hud.events import HudEventSource
from zero_ad_eyes.infrastructure.model import StubPerceptionModel
from zero_ad_eyes.infrastructure.tracking import ClassicalEventDetector
from zero_ad_eyes.interface.cli import _event_sources
from zero_ad_eyes.interface.default_config import default_config


class _Recorder:
    """An EventSource that emits one event per frame and records what it was given."""

    def __init__(self, origin: EventOrigin) -> None:
        self.origin = origin
        self.seen: list[tuple[WorldModel, PerceptionContext]] = []

    def events(self, world_model: WorldModel, context: PerceptionContext) -> tuple[Event, ...]:
        self.seen.append((world_model, context))
        return (
            Event(
                kind=EventKind.STATE_CHANGED, frame_id=world_model.meta.frame_id, origin=self.origin
            ),
        )


class _Calibrator:
    def calibrate(self, frame: Frame) -> Calibration:
        height, width = frame.image.shape[:2]
        return Calibration(width=width, height=height)


class _Drifting:
    def verify(self, frame: Frame, calibration: Calibration) -> CalibrationCheck:
        return CalibrationCheck(matches=False, confidence=1.0)


def _frames(n: int) -> list[Frame]:
    return [
        Frame(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            meta=FrameMeta(frame_id=i, timestamp=float(i), source="t", width=8, height=8),
        )
        for i in range(n)
    ]


def test_events_from_every_source_are_concatenated_in_source_order() -> None:
    hud, entity = _Recorder(EventOrigin.HUD), _Recorder(EventOrigin.ENTITY)
    pipeline = PerceptionPipeline(
        InMemoryFrameSource(_frames(2)),
        StubPerceptionModel(),
        recalibrate_interval=30,
        event_sources=(hud, entity),
    )
    models = list(pipeline.run())
    assert [[e.origin for e in m.events] for m in models] == [
        [EventOrigin.HUD, EventOrigin.ENTITY],
        [EventOrigin.HUD, EventOrigin.ENTITY],
    ]
    # Sources see the frame's model before events are attached.
    assert all(model.events == () for model, _ in hud.seen)


def test_context_reports_recalibration_and_drift() -> None:
    recorder = _Recorder(EventOrigin.SYSTEM)
    pipeline = PerceptionPipeline(
        InMemoryFrameSource(_frames(3)),
        StubPerceptionModel(),
        recalibrate_interval=1,
        calibrator=_Calibrator(),
        self_check=_Drifting(),
        event_sources=(recorder,),
    )
    list(pipeline.run())
    contexts = [context for _, context in recorder.seen]
    assert contexts[0] == PerceptionContext(recalibrated=True)  # first frame: initial detect
    assert contexts[1] == PerceptionContext(recalibrated=True, layout_drift=True)


def test_no_sources_means_no_events() -> None:
    pipeline = PerceptionPipeline(
        InMemoryFrameSource(_frames(1)), StubPerceptionModel(), recalibrate_interval=30
    )
    assert next(iter(pipeline.run())).events == ()


def test_classical_event_detector_implements_the_port() -> None:
    assert isinstance(ClassicalEventDetector.from_settings(default_config().tracking), EventSource)


def test_entity_source_follows_its_config_switch() -> None:
    cfg = default_config()
    assert [type(s) for s in _event_sources(cfg)] == [HudEventSource, ClassicalEventDetector]
    off = cfg.model_copy(
        update={"events": cfg.events.model_copy(update={"entity": False, "hud": False})}
    )
    assert _event_sources(off) == ()
