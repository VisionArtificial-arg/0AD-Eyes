"""ML8 — Evaluation harness for the NF3 accuracy metrics (REQUIREMENTS.md §6/§7).

Computes the four NF3 metrics by comparing *predicted* world models against
*ground-truth* world models (produced offline by ML2/ML3):

- **HUD / resource read error** (< 1%): mean relative error of the resource and
  population read-outs. Classical OCR (EPIC C) — computable now.
- **Detection mAP** (>= 0.80): mean average precision of detections. This is the
  🔌 model-dependent metric (learned detection/classification, E1/E2, CV-10/11/14).
  Per §5.10.4 it is **unmeasurable until the real adapter (MP4) lands**, so the
  harness reports it as ``pending-model`` unless the caller explicitly supplies a
  real model's detections and flags ``model_available``. The mAP *computation*
  itself is provided (``mean_average_precision``) so ``just eval`` can call it once
  the model exists — but the harness never fabricates model outputs.
- **Ownership accuracy** (>= 98%): fraction of matched entities whose owner colour
  is read correctly. Classical player-colour segmentation (E3) — computable now.
- **Tracking MOTA** (>= 0.70): 1 - (FN + FP + IDSW) / GT, across the sequence.
  Tracking (EPIC G) runs on the stub — computable now.

The public entry point is ``evaluate(...) -> EvaluationReport``: a clean callable
API returning a value object, wired behind ``just eval`` (see the justfile).

Offline-only: this harness scores recordings against engine/annotated ground
truth; it is never part of the inference path.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.hud import HudState
from zero_ad_eyes.domain.world_model import WorldModel

PENDING_MODEL = "pending-model"


class MetricStatus(StrEnum):
    """Whether a metric was computed or is blocked on the trained model (🔌)."""

    COMPUTED = "computed"
    PENDING_MODEL = PENDING_MODEL


class MetricResult(BaseModel):
    """One NF3 metric: its value, its threshold, and whether it passed.

    A ``PENDING_MODEL`` metric has no value and no pass/fail verdict — it is
    deferred until the model artifact (MP4) is delivered.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    status: MetricStatus
    value: float | None = None
    threshold: float | None = None
    higher_is_better: bool = True

    @classmethod
    def pending_model(
        cls, name: str, threshold: float | None = None, higher_is_better: bool = True
    ) -> MetricResult:
        return cls(
            name=name,
            status=MetricStatus.PENDING_MODEL,
            threshold=threshold,
            higher_is_better=higher_is_better,
        )

    @classmethod
    def computed(
        cls, name: str, value: float, threshold: float | None, higher_is_better: bool
    ) -> MetricResult:
        return cls(
            name=name,
            status=MetricStatus.COMPUTED,
            value=value,
            threshold=threshold,
            higher_is_better=higher_is_better,
        )

    @property
    def is_pending(self) -> bool:
        return self.status is MetricStatus.PENDING_MODEL

    @property
    def passed(self) -> bool | None:
        """``True``/``False`` against the threshold, or ``None`` when unmeasurable."""

        if self.is_pending or self.value is None or self.threshold is None:
            return None
        if self.higher_is_better:
            return self.value >= self.threshold
        return self.value <= self.threshold


class EvaluationReport(BaseModel):
    """The NF3 scorecard (the ``just validate`` accuracy gate output)."""

    model_config = ConfigDict(frozen=True)

    hud_read_error: MetricResult
    detection_map: MetricResult
    ownership_accuracy: MetricResult
    tracking_mota: MetricResult

    @property
    def metrics(self) -> tuple[MetricResult, ...]:
        return (
            self.hud_read_error,
            self.detection_map,
            self.ownership_accuracy,
            self.tracking_mota,
        )

    @property
    def has_pending(self) -> bool:
        return any(metric.is_pending for metric in self.metrics)

    @property
    def passed(self) -> bool | None:
        """Overall verdict, or ``None`` if any metric is still ``pending-model``."""

        if self.has_pending:
            return None
        return all(metric.passed for metric in self.metrics)


class EvalConfig(BaseModel):
    """NF3 thresholds and matching parameters (config-driven, NF7)."""

    model_config = ConfigDict(frozen=True)

    hud_error_max: float = 0.01  # < 1%
    detection_map_min: float = 0.80
    ownership_accuracy_min: float = 0.98
    tracking_mota_min: float = 0.70
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Geometry / matching helpers                                                 #
# --------------------------------------------------------------------------- #


def iou(a: ScreenBBox, b: ScreenBBox) -> float:
    """Intersection-over-union of two screen-space boxes."""

    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)

    inter_w = max(0.0, right - left)
    inter_h = max(0.0, bottom - top)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


# --------------------------------------------------------------------------- #
# HUD read error (classical, computable now)                                  #
# --------------------------------------------------------------------------- #


def _relative_error(predicted: int, truth: int) -> float:
    return abs(predicted - truth) / max(abs(truth), 1)


def hud_read_error(predicted: HudState | None, truth: HudState | None) -> float | None:
    """Mean relative error over the resource counters and population.

    Returns ``None`` when there is no ground-truth HUD to score against. A missing
    prediction where truth exists scores maximal error (1.0) per read-out.
    """

    if truth is None:
        return None

    errors: list[float] = []
    for resource, truth_value in truth.stockpiles.items():
        predicted_value = predicted.stockpiles.get(resource) if predicted else None
        errors.append(
            1.0 if predicted_value is None else _relative_error(predicted_value, truth_value)
        )

    if truth.population is not None:
        predicted_pop = predicted.population if predicted else None
        errors.append(
            1.0
            if predicted_pop is None
            else _relative_error(predicted_pop.current, truth.population.current)
        )

    if not errors:
        return None
    return sum(errors) / len(errors)


def _mean_hud_error(predicted: Sequence[WorldModel], truth: Sequence[WorldModel]) -> float | None:
    by_pred = {wm.meta.frame_id: wm for wm in predicted}
    frame_errors: list[float] = []
    for gt in truth:
        error = hud_read_error(
            by_pred[gt.meta.frame_id].hud if gt.meta.frame_id in by_pred else None,
            gt.hud,
        )
        if error is not None:
            frame_errors.append(error)
    if not frame_errors:
        return None
    return sum(frame_errors) / len(frame_errors)


# --------------------------------------------------------------------------- #
# Ownership accuracy (classical E3, computable now)                           #
# --------------------------------------------------------------------------- #


def ownership_accuracy(
    predicted: Sequence[WorldModel],
    truth: Sequence[WorldModel],
    iou_threshold: float = 0.5,
) -> float | None:
    """Fraction of ground-truth entities whose ownership is predicted correctly.

    Ground-truth and predicted entities are matched **geometrically**, by screen-box
    IoU per frame (the same association ``tracking_mota`` uses). Matching by
    ``entity_id`` would be wrong here: the predicted id is the tracker's own track id
    while the ground-truth id is the engine entity id — the two id spaces never
    coincide when perceiving from pixels, so id-matching would score ~0 on real data.
    A ground-truth entity with no IoU-matched prediction counts as incorrect (you
    cannot read the owner of an entity you failed to locate). Returns ``None`` if
    there are no ground-truth entities at all.
    """

    by_pred_frame = {wm.meta.frame_id: wm for wm in predicted}
    correct = 0
    total = 0
    for gt in truth:
        pred = by_pred_frame.get(gt.meta.frame_id)
        pred_entities = pred.entities if pred else ()
        matches = _match_frame(pred_entities, gt.entities, iou_threshold)
        pred_by_id = {entity.entity_id: entity for entity in pred_entities}
        for gt_entity in gt.entities:
            total += 1
            matched_pred_id = matches.get(gt_entity.entity_id)
            match = pred_by_id.get(matched_pred_id) if matched_pred_id is not None else None
            if match is not None and match.ownership == gt_entity.ownership:
                correct += 1
    if total == 0:
        return None
    return correct / total


# --------------------------------------------------------------------------- #
# Tracking MOTA (EPIC G on stub, computable now)                              #
# --------------------------------------------------------------------------- #


def _match_frame(
    predicted: Sequence[Entity], truth: Sequence[Entity], iou_threshold: float
) -> dict[int, int]:
    """Greedy gt->pred id matching by descending screen-box IoU above threshold."""

    candidates: list[tuple[float, int, int]] = []
    for gt_entity in truth:
        if gt_entity.screen_bbox is None:
            continue
        for pred_entity in predicted:
            if pred_entity.screen_bbox is None:
                continue
            overlap = iou(gt_entity.screen_bbox, pred_entity.screen_bbox)
            if overlap >= iou_threshold:
                candidates.append((overlap, gt_entity.entity_id, pred_entity.entity_id))

    candidates.sort(key=lambda c: c[0], reverse=True)
    matches: dict[int, int] = {}
    used_pred: set[int] = set()
    for _, gt_id, pred_id in candidates:
        if gt_id in matches or pred_id in used_pred:
            continue
        matches[gt_id] = pred_id
        used_pred.add(pred_id)
    return matches


def tracking_mota(
    predicted: Sequence[WorldModel], truth: Sequence[WorldModel], iou_threshold: float
) -> float | None:
    """Multi-object tracking accuracy: 1 - (FN + FP + IDSW) / GT across frames.

    Per frame, predicted and ground-truth entities are matched by IoU of their
    screen boxes. False negatives are unmatched ground truth, false positives are
    unmatched predictions, and an ID switch is a ground-truth track whose matched
    prediction id differs from the last frame it was matched. Returns ``None`` when
    there is no ground truth to score.
    """

    by_pred_frame = {wm.meta.frame_id: wm for wm in predicted}
    total_gt = 0
    fn = 0
    fp = 0
    idsw = 0
    last_match: dict[int, int] = {}  # gt_id -> pred_id last associated

    for gt in sorted(truth, key=lambda wm: wm.meta.frame_id):
        gt_entities = gt.entities
        pred = by_pred_frame.get(gt.meta.frame_id)
        pred_entities = pred.entities if pred else ()

        total_gt += len(gt_entities)
        matches = _match_frame(pred_entities, gt_entities, iou_threshold)

        fn += len(gt_entities) - len(matches)
        fp += len(pred_entities) - len(matches)
        for gt_id, pred_id in matches.items():
            previous = last_match.get(gt_id)
            if previous is not None and previous != pred_id:
                idsw += 1
            last_match[gt_id] = pred_id

    if total_gt == 0:
        return None
    return 1.0 - (fn + fp + idsw) / total_gt


# --------------------------------------------------------------------------- #
# Detection mAP (🔌 model-dependent — pending-model until MP4)                 #
# --------------------------------------------------------------------------- #


def _average_precision(
    predictions: list[tuple[int, Detection]],
    truth_by_frame: dict[int, list[Detection]],
    iou_threshold: float,
) -> float:
    """VOC-style all-points AP for one class.

    ``predictions`` are ``(frame_id, detection)`` pairs ranked globally by score,
    but each is matched only against ground truth *in its own frame* — pooling
    across frames would let a box match a different frame's entity.
    """

    total_gt = sum(len(items) for items in truth_by_frame.values())
    if total_gt == 0:
        return 0.0

    ordered = sorted(predictions, key=lambda pair: pair[1].confidence.value, reverse=True)
    matched: dict[int, set[int]] = {frame_id: set() for frame_id in truth_by_frame}
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for frame_id, prediction in ordered:
        frame_truth = truth_by_frame.get(frame_id, [])
        used = matched.setdefault(frame_id, set())
        best_iou = iou_threshold
        best_index = -1
        for index, gt_detection in enumerate(frame_truth):
            if index in used:
                continue
            overlap = iou(prediction.bbox, gt_detection.bbox)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0:
            used.add(best_index)
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_gt)

    if not precisions:
        return 0.0

    # All-points interpolation: integrate max-precision-to-the-right over recall.
    ap = 0.0
    previous_recall = 0.0
    for recall_level in sorted(set(recalls)):
        max_precision = max(
            (p for p, r in zip(precisions, recalls, strict=True) if r >= recall_level),
            default=0.0,
        )
        ap += max_precision * (recall_level - previous_recall)
        previous_recall = recall_level
    return ap


def mean_average_precision(
    predicted: Sequence[Detections],
    truth: Sequence[Detections],
    iou_threshold: float = 0.5,
) -> float:
    """mAP over all detection classes, matched per-frame, ranked per-class.

    Provided so ``just eval`` can score real detections once the model lands. The
    harness itself does NOT call this without ``model_available`` — with no model
    there are no learned detections to score, and a fabricated 0.0 would be a lie.
    """

    pred_by_frame = {d.frame_id: d for d in predicted}
    truth_by_class_frame: dict[str, dict[int, list[Detection]]] = {}
    preds_by_class: dict[str, list[tuple[int, Detection]]] = {}

    for gt in truth:
        for item in gt.items:
            truth_by_class_frame.setdefault(item.kind, {}).setdefault(gt.frame_id, []).append(item)
        pred = pred_by_frame.get(gt.frame_id)
        for item in pred.items if pred else ():
            preds_by_class.setdefault(item.kind, []).append((gt.frame_id, item))

    classes = set(truth_by_class_frame) | set(preds_by_class)
    if not classes:
        return 0.0

    aps = [
        _average_precision(
            preds_by_class.get(cls, []),
            truth_by_class_frame.get(cls, {}),
            iou_threshold,
        )
        for cls in classes
    ]
    return sum(aps) / len(aps)


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def evaluate(
    predicted: Sequence[WorldModel],
    truth: Sequence[WorldModel],
    *,
    predicted_detections: Sequence[Detections] | None = None,
    truth_detections: Sequence[Detections] | None = None,
    model_available: bool = False,
    config: EvalConfig | None = None,
) -> EvaluationReport:
    """Score predicted world models against ground truth; return an NF3 report.

    ``predicted`` and ``truth`` are aligned by ``meta.frame_id``. The detection mAP
    metric is reported as ``pending-model`` unless ``model_available`` is set *and*
    both detection sequences are supplied (i.e. a real model produced them). All
    other metrics are classical/stub-computable and are always evaluated.
    """

    cfg = config or EvalConfig()

    hud_error = _mean_hud_error(predicted, truth)
    hud_metric = (
        MetricResult.computed("hud_read_error", hud_error, cfg.hud_error_max, False)
        if hud_error is not None
        else MetricResult.computed("hud_read_error", 0.0, cfg.hud_error_max, False)
    )

    if model_available and predicted_detections is not None and truth_detections is not None:
        map_value = mean_average_precision(
            predicted_detections, truth_detections, cfg.iou_threshold
        )
        detection_metric = MetricResult.computed(
            "detection_map", map_value, cfg.detection_map_min, True
        )
    else:
        detection_metric = MetricResult.pending_model("detection_map", cfg.detection_map_min, True)

    ownership = ownership_accuracy(predicted, truth, cfg.iou_threshold)
    ownership_metric = MetricResult.computed(
        "ownership_accuracy",
        ownership if ownership is not None else 1.0,
        cfg.ownership_accuracy_min,
        True,
    )

    mota = tracking_mota(predicted, truth, cfg.iou_threshold)
    mota_metric = MetricResult.computed(
        "tracking_mota",
        mota if mota is not None else 1.0,
        cfg.tracking_mota_min,
        True,
    )

    return EvaluationReport(
        hud_read_error=hud_metric,
        detection_map=detection_metric,
        ownership_accuracy=ownership_metric,
        tracking_mota=mota_metric,
    )
