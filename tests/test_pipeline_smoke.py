"""End-to-end smoke test: source -> pipeline -> world model, on the stub model.

Proves the seam runs with zero model dependency (DoD-A). Also exercises the stub's
fixture mode and the overlay skeleton.
"""

from __future__ import annotations

from zero_ad_eyes.application.pipeline import PerceptionPipeline
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.domain.world_model import WorldModel
from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource
from zero_ad_eyes.infrastructure.model.stub_adapter import StubPerceptionModel
from zero_ad_eyes.interface.overlay import render

from .conftest import make_frame


def test_pipeline_produces_one_world_model_per_frame() -> None:
    frames = [make_frame(i) for i in range(3)]
    pipeline = PerceptionPipeline(
        InMemoryFrameSource(frames), StubPerceptionModel(), recalibrate_interval=30
    )

    results = list(pipeline.run())

    assert len(results) == 3
    assert all(isinstance(wm, WorldModel) for wm in results)
    assert [wm.meta.frame_id for wm in results] == [0, 1, 2]
    assert results[0].entities == ()  # empty stub → no entities


def test_stub_fixture_mode_replays_detections() -> None:
    det = Detection(
        kind=EntityKind.BUILDING,
        bbox=ScreenBBox(x=1, y=1, width=2, height=2),
        confidence=Confidence(value=0.9, provenance=Provenance.ENGINE_GT),
    )
    fixtures = {0: Detections(frame_id=0, items=(det,))}
    model = StubPerceptionModel(fixtures=fixtures)

    assert len(model.infer(make_frame(0))) == 1
    assert len(model.infer(make_frame(1))) == 0  # no fixture → empty


def test_overlay_renders_without_a_display() -> None:
    frame = make_frame()
    wm = WorldModel(
        meta=frame.meta,
        entities=(
            Entity(
                entity_id=7,
                kind=EntityKind.UNIT,
                screen_bbox=ScreenBBox(x=2, y=2, width=5, height=5),
            ),
        ),
    )
    canvas = render(frame, wm)
    assert canvas.shape == frame.image.shape
