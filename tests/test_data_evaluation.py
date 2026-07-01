"""ML8 tests — NF3 metric values on known cases + pending-model for the 🔌 metric."""

from __future__ import annotations

import math

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.hud import HudState
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership, ResourceType
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel
from zero_ad_eyes.infrastructure.data.evaluation import (
    EvaluationReport,
    MetricStatus,
    evaluate,
    hud_read_error,
    iou,
    mean_average_precision,
    ownership_accuracy,
    tracking_mota,
)

BOX_A = ScreenBBox(x=0, y=0, width=10, height=10)
BOX_B = ScreenBBox(x=20, y=20, width=10, height=10)


def _meta(frame_id: int) -> FrameMeta:
    return FrameMeta(
        frame_id=frame_id, timestamp=float(frame_id), source="rec", width=64, height=48
    )


def _entity(entity_id: int, box: ScreenBBox, ownership: Ownership = Ownership.UNKNOWN) -> Entity:
    return Entity(entity_id=entity_id, kind=EntityKind.UNIT, ownership=ownership, screen_bbox=box)


def _gt_detection(box: ScreenBBox) -> Detection:
    return Detection(
        kind=EntityKind.UNIT, bbox=box, confidence=Confidence.certain(Provenance.ENGINE_GT)
    )


def _pred_detection(box: ScreenBBox, score: float) -> Detection:
    return Detection(
        kind=EntityKind.UNIT,
        bbox=box,
        confidence=Confidence(value=score, provenance=Provenance.LEARNED),
    )


# --- geometry -------------------------------------------------------------- #


def test_iou_values() -> None:
    assert iou(BOX_A, BOX_A) == 1.0
    assert iou(BOX_A, BOX_B) == 0.0
    half = ScreenBBox(x=5, y=0, width=10, height=10)
    assert math.isclose(iou(BOX_A, half), 1.0 / 3.0)


# --- HUD read error -------------------------------------------------------- #


def test_hud_read_error_relative() -> None:
    truth = HudState(stockpiles={ResourceType.FOOD: 100, ResourceType.WOOD: 100})
    predicted = HudState(stockpiles={ResourceType.FOOD: 110, ResourceType.WOOD: 90})
    error = hud_read_error(predicted, truth)
    assert error is not None
    assert math.isclose(error, 0.1)


def test_hud_read_error_missing_prediction_is_maximal() -> None:
    truth = HudState(stockpiles={ResourceType.FOOD: 100})
    assert hud_read_error(None, truth) == 1.0


# --- ownership accuracy ---------------------------------------------------- #


def test_ownership_accuracy_two_of_three() -> None:
    gt = WorldModel(
        meta=_meta(0),
        entities=(
            _entity(1, BOX_A, Ownership.SELF),
            _entity(2, BOX_A, Ownership.ENEMY),
            _entity(3, BOX_A, Ownership.ALLY),
        ),
    )
    pred = WorldModel(
        meta=_meta(0),
        entities=(
            _entity(1, BOX_A, Ownership.SELF),
            _entity(2, BOX_A, Ownership.ENEMY),
            _entity(3, BOX_A, Ownership.GAIA),  # wrong
        ),
    )
    accuracy = ownership_accuracy([pred], [gt])
    assert accuracy is not None
    assert math.isclose(accuracy, 2.0 / 3.0)


# --- tracking MOTA --------------------------------------------------------- #


def test_tracking_mota_with_one_false_negative() -> None:
    gt = [
        WorldModel(meta=_meta(0), entities=(_entity(1, BOX_A), _entity(2, BOX_B))),
        WorldModel(meta=_meta(1), entities=(_entity(1, BOX_A), _entity(2, BOX_B))),
    ]
    pred = [
        WorldModel(meta=_meta(0), entities=(_entity(10, BOX_A), _entity(20, BOX_B))),
        WorldModel(meta=_meta(1), entities=(_entity(10, BOX_A),)),  # misses gt id 2
    ]
    # total_gt=4, FN=1, FP=0, IDSW=0 -> 1 - 1/4
    mota = tracking_mota(pred, gt, iou_threshold=0.5)
    assert mota is not None
    assert math.isclose(mota, 0.75)


def test_tracking_mota_counts_id_switch() -> None:
    gt = [
        WorldModel(meta=_meta(0), entities=(_entity(1, BOX_A),)),
        WorldModel(meta=_meta(1), entities=(_entity(1, BOX_A),)),
    ]
    pred = [
        WorldModel(meta=_meta(0), entities=(_entity(10, BOX_A),)),
        WorldModel(meta=_meta(1), entities=(_entity(11, BOX_A),)),  # id changed -> IDSW
    ]
    # total_gt=2, FN=0, FP=0, IDSW=1 -> 1 - 1/2
    mota = tracking_mota(pred, gt, iou_threshold=0.5)
    assert mota is not None
    assert math.isclose(mota, 0.5)


# --- detection mAP (pure computation) -------------------------------------- #


def test_map_perfect_predictions() -> None:
    truth = [Detections(frame_id=0, items=(_gt_detection(BOX_A),))]
    predicted = [Detections(frame_id=0, items=(_pred_detection(BOX_A, 0.9),))]
    assert math.isclose(mean_average_precision(predicted, truth), 1.0)


def test_map_no_predictions_is_zero() -> None:
    truth = [Detections(frame_id=0, items=(_gt_detection(BOX_A),))]
    predicted = [Detections(frame_id=0, items=())]
    assert mean_average_precision(predicted, truth) == 0.0


# --- harness: 🔌 detection mAP reports pending-model ----------------------- #


def _paired_worldmodels() -> tuple[list[WorldModel], list[WorldModel]]:
    gt = [
        WorldModel(
            meta=_meta(0),
            hud=HudState(stockpiles={ResourceType.FOOD: 100}),
            entities=(_entity(1, BOX_A, Ownership.SELF),),
        )
    ]
    pred = [
        WorldModel(
            meta=_meta(0),
            hud=HudState(stockpiles={ResourceType.FOOD: 100}),
            entities=(_entity(1, BOX_A, Ownership.SELF),),
        )
    ]
    return pred, gt


def test_evaluate_reports_detection_map_pending_model_by_default() -> None:
    pred, gt = _paired_worldmodels()

    report = evaluate(pred, gt)

    assert isinstance(report, EvaluationReport)
    assert report.detection_map.status is MetricStatus.PENDING_MODEL
    assert report.detection_map.value is None
    assert report.detection_map.passed is None
    assert report.has_pending is True
    assert report.passed is None  # any pending metric ⇒ no overall verdict

    # The non-model metrics ARE computed even without the model.
    assert report.hud_read_error.status is MetricStatus.COMPUTED
    assert report.ownership_accuracy.status is MetricStatus.COMPUTED
    assert report.ownership_accuracy.value == 1.0
    assert report.tracking_mota.status is MetricStatus.COMPUTED


def test_evaluate_computes_map_only_when_model_available() -> None:
    pred, gt = _paired_worldmodels()
    truth_det = [Detections(frame_id=0, items=(_gt_detection(BOX_A),))]
    pred_det = [Detections(frame_id=0, items=(_pred_detection(BOX_A, 0.9),))]

    report = evaluate(
        pred,
        gt,
        predicted_detections=pred_det,
        truth_detections=truth_det,
        model_available=True,
    )

    assert report.detection_map.status is MetricStatus.COMPUTED
    assert report.detection_map.value is not None
    assert math.isclose(report.detection_map.value, 1.0)
    assert report.has_pending is False
    assert report.passed is True  # all four NF3 metrics pass on this clean case
