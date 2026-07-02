"""ML8 glue — score predicted world models against an engine export end-to-end.

Exercises the offline accuracy loop with hand-authored inputs (no game, no model):
a couple of predicted world models + an engine ground-truth export → NF3 scorecard.
"""

from __future__ import annotations

import math

from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.hud import HudState
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership, ResourceType
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel
from zero_ad_eyes.infrastructure.data import (
    AlignBy,
    MetricStatus,
    evaluate_against_engine,
)
from zero_ad_eyes.infrastructure.data.ground_truth import (
    EngineEntityState,
    EngineFrameState,
    EngineStateExport,
)

BOX = ScreenBBox(x=10, y=10, width=20, height=20)


def _meta(frame_id: int, timestamp: float) -> FrameMeta:
    return FrameMeta(frame_id=frame_id, timestamp=timestamp, source="rec", width=64, height=48)


def _predicted(frame_id: int, timestamp: float, *, track_id: int) -> WorldModel:
    """A predicted frame: tracker-assigned id (deliberately NOT the engine id)."""

    return WorldModel(
        meta=_meta(frame_id, timestamp),
        hud=HudState(stockpiles={ResourceType.FOOD: 100}),
        entities=(
            Entity(
                entity_id=track_id,
                kind=EntityKind.UNIT,
                ownership=Ownership.SELF,
                screen_bbox=BOX,
            ),
        ),
    )


def _export(*, frame_id: int = 0, timestamp: float = 0.0) -> EngineStateExport:
    frame = EngineFrameState(
        frame_id=frame_id,
        timestamp=timestamp,
        resources={ResourceType.FOOD: 100},
        entities=(
            EngineEntityState(
                entity_id=42,  # engine id — differs from any tracker id
                kind=EntityKind.UNIT,
                owner=1,
                bbox=BOX,
            ),
        ),
    )
    return EngineStateExport(match_id="demo", self_player=1, frames=(frame,))


def test_evaluate_against_engine_scores_classical_metrics() -> None:
    predicted = [_predicted(0, 0.0, track_id=7)]

    report = evaluate_against_engine(predicted, _export())

    # HUD read is exact; ownership + tracking are perfect on this clean pair.
    assert report.hud_read_error.status is MetricStatus.COMPUTED
    assert math.isclose(report.hud_read_error.value, 0.0)  # type: ignore[arg-type]
    assert report.ownership_accuracy.value == 1.0  # matched by IoU despite id mismatch
    assert report.tracking_mota.value == 1.0
    # Detection mAP stays pending until the learned adapter (MP4) lands.
    assert report.detection_map.status is MetricStatus.PENDING_MODEL
    assert report.passed is None  # a pending metric ⇒ no overall verdict


def test_evaluate_against_engine_by_timestamp() -> None:
    # Predicted frame captured at t=0.05; engine frame at t=0.0 within tolerance.
    predicted = [_predicted(5, 0.05, track_id=7)]

    report = evaluate_against_engine(
        predicted,
        _export(frame_id=0, timestamp=0.0),
        align_by=AlignBy.TIMESTAMP,
        time_tolerance=0.1,
    )

    assert report.ownership_accuracy.value == 1.0
    assert report.tracking_mota.value == 1.0


def test_evaluate_against_engine_no_alignment_leaves_metrics_unmeasured() -> None:
    # No engine frame matches frame_id 99 → empty ground truth → nothing to score.
    predicted = [_predicted(99, 9.0, track_id=7)]

    report = evaluate_against_engine(predicted, _export(frame_id=0))

    # evaluate() defaults unmeasured classical metrics rather than crashing.
    assert report.ownership_accuracy.status is MetricStatus.COMPUTED
    assert report.detection_map.status is MetricStatus.PENDING_MODEL
