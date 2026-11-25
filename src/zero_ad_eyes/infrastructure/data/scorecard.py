"""EVAL-1 — the full scorecard: what the eyes reported vs the engine truth.

``score_run`` scores a pipeline run's world models against an engine export that
carries screen boxes (e.g. ``EngineDatasetRun.to_export()``, boxes from the instance
masks). It scores the eyes' *output*, i.e. what a consumer actually receives:

- **detection:** AP at IoU 0.5 per class plus mAP50 and mAP50-95. The predictions are
  the entities observed this frame (``staleness == 0``) with a screen box, ranked by
  confidence. Classes come at two granularities: ``coarse`` (unit / building /
  resource_node / …) and ``fine`` (the 7 segmentation classes: wood, stone, metal,
  food, fauna, building, unit).
- **HUD:** exact-match rate per field (each resource, population, cap, phase) and the
  mean relative error of the numeric fields.
- **ownership accuracy** and **health MAE** over ground-truth entities matched to a
  prediction by box IoU.
- **tracking MOTA** (the ML8 definition).
- **minimap:** blip precision/recall against the units and buildings the viewing
  player can see, matched by world distance.
- **events:** counts per kind (there is no event truth; this is a sanity profile).

A metric with nothing to score is ``None`` (unmeasured) and never a pass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from zero_ad_eyes.domain.confidence import Confidence
from zero_ad_eyes.domain.detections import Detection
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.hud import HudState
from zero_ad_eyes.domain.taxonomy import EntityKind, Phase, ResourceType
from zero_ad_eyes.domain.world_model import WorldModel
from zero_ad_eyes.infrastructure.model.segmentation_classes import CLASS_NAMES

from .evaluation import _average_precision, _match_frame, tracking_mota
from .ground_truth import EngineEntityState, EngineFrameState, EngineStateExport, GroundTruthAligner
from .yolo_dataset import YOLO_NAMES

MAP_IOUS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
Pair = tuple[WorldModel, WorldModel, EngineFrameState]  # predicted, truth, engine frame
_HUD_FIELDS = (*(r.value for r in ResourceType), "population_current", "population_cap", "phase")


class Rate(BaseModel):
    """A measured fraction (or mean) over ``n`` samples; ``value`` is ``None`` if n == 0."""

    value: float | None = None
    n: int = 0

    @classmethod
    def of(cls, hits: float, n: int) -> Rate:
        return cls(value=hits / n if n else None, n=n)


class DetectionScore(BaseModel):
    ap50: dict[str, float] = Field(default_factory=dict)  # per class with ground truth
    truth_instances: dict[str, int] = Field(default_factory=dict)
    predicted_instances: dict[str, int] = Field(default_factory=dict)
    map50: float | None = None
    map50_95: float | None = None


class Scorecard(BaseModel):
    frames: int
    detection_coarse: DetectionScore
    detection_fine: DetectionScore
    hud_exact: dict[str, Rate]
    hud_mean_relative_error: Rate
    ownership_accuracy: Rate
    health_mae: Rate
    tracking_mota: float | None
    minimap_blip_precision: Rate
    minimap_blip_recall: Rate
    events_per_kind: dict[str, int]


def score_run(
    predicted: Sequence[WorldModel],
    export: EngineStateExport,
    *,
    iou_threshold: float,
    blip_radius: float,
) -> Scorecard:
    """Score ``predicted`` against ``export``, pairing frames by ``frame_id``."""

    aligner = GroundTruthAligner(export)
    pairs: list[Pair] = []
    for world_model in predicted:
        engine_frame = aligner.frame_for_id(world_model.meta.frame_id)
        if engine_frame is not None:
            truth = aligner.labels_for(engine_frame, world_model.meta).world_model
            pairs.append((world_model, truth, engine_frame))

    hud_exact, hud_error = _hud_scores(pairs)
    ownership, health = _entity_attribute_scores(pairs, iou_threshold)
    precision, recall = _minimap_scores(pairs, blip_radius)
    return Scorecard(
        frames=len(pairs),
        detection_coarse=_detection_score(pairs, _coarse_truth, _coarse_pred),
        detection_fine=_detection_score(pairs, _fine_truth, _fine_pred),
        hud_exact=hud_exact,
        hud_mean_relative_error=hud_error,
        ownership_accuracy=ownership,
        health_mae=health,
        tracking_mota=tracking_mota(
            [p for p, _, _ in pairs], [t for _, t, _ in pairs], iou_threshold
        ),
        minimap_blip_precision=precision,
        minimap_blip_recall=recall,
        events_per_kind=dict(
            sorted(Counter(e.kind.value for p, _, _ in pairs for e in p.events).items())
        ),
    )


# --------------------------------------------------------------------------- #
# Detection                                                                   #
# --------------------------------------------------------------------------- #


def _coarse_truth(entity: EngineEntityState) -> str:
    return entity.kind.value


def _fine_truth(entity: EngineEntityState) -> str:
    name = CLASS_NAMES.get(entity.seg_class) if entity.seg_class is not None else None
    return name if name in YOLO_NAMES else entity.kind.value


def _coarse_pred(entity: Entity) -> str:
    return entity.kind.value


def _fine_pred(entity: Entity) -> str:
    return entity.entity_type if entity.entity_type in YOLO_NAMES else entity.kind.value


def _detection_score(
    pairs: Sequence[Pair],
    truth_class: Callable[[EngineEntityState], str],
    pred_class: Callable[[Entity], str],
) -> DetectionScore:
    truth: dict[str, dict[int, list[Detection]]] = {}
    preds: dict[str, list[tuple[int, Detection]]] = {}
    for predicted, _, engine_frame in pairs:
        frame_id = predicted.meta.frame_id
        for gt in engine_frame.entities:
            if gt.bbox is not None:
                truth.setdefault(truth_class(gt), {}).setdefault(frame_id, []).append(
                    Detection(kind=gt.kind, bbox=gt.bbox, confidence=Confidence.certain())
                )
        for entity in predicted.entities:
            if entity.screen_bbox is None or entity.staleness > 0:
                continue
            preds.setdefault(pred_class(entity), []).append(
                (
                    frame_id,
                    Detection(
                        kind=entity.kind, bbox=entity.screen_bbox, confidence=entity.confidence
                    ),
                )
            )
    if not truth:
        return DetectionScore(predicted_instances={c: len(p) for c, p in sorted(preds.items())})

    def ap(cls: str, threshold: float) -> float:
        return _average_precision(preds.get(cls, []), truth[cls], threshold)

    ap50 = {cls: ap(cls, 0.5) for cls in sorted(truth)}
    per_iou = [sum(ap(cls, t) for cls in truth) / len(truth) for t in MAP_IOUS]
    return DetectionScore(
        ap50=ap50,
        truth_instances={c: sum(len(v) for v in f.values()) for c, f in sorted(truth.items())},
        predicted_instances={c: len(p) for c, p in sorted(preds.items())},
        map50=sum(ap50.values()) / len(ap50),
        map50_95=sum(per_iou) / len(per_iou),
    )


# --------------------------------------------------------------------------- #
# HUD                                                                         #
# --------------------------------------------------------------------------- #


def _hud_fields(hud: HudState | None) -> dict[str, int | str]:
    if hud is None:
        return {}
    fields: dict[str, int | str] = {r.value: v for r, v in hud.stockpiles.items()}
    if hud.population is not None:
        fields["population_current"] = hud.population.current
        fields["population_cap"] = hud.population.cap
    if hud.phase is not Phase.UNKNOWN:
        fields["phase"] = hud.phase.value
    return fields


def _hud_scores(pairs: Sequence[Pair]) -> tuple[dict[str, Rate], Rate]:
    hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    errors: list[float] = []
    for predicted, truth, _ in pairs:
        want, got = _hud_fields(truth.hud), _hud_fields(predicted.hud)
        for name, value in want.items():
            totals[name] += 1
            read = got.get(name)
            hits[name] += read == value
            if isinstance(value, int) and name != "phase":
                if isinstance(read, int):
                    errors.append(abs(read - value) / max(abs(value), 1))
                else:
                    errors.append(1.0)  # unread counts as fully wrong
    exact = {name: Rate.of(hits[name], totals[name]) for name in _HUD_FIELDS if totals[name]}
    return exact, Rate.of(sum(errors), len(errors))


# --------------------------------------------------------------------------- #
# Entity attributes                                                            #
# --------------------------------------------------------------------------- #


def _entity_attribute_scores(pairs: Sequence[Pair], iou_threshold: float) -> tuple[Rate, Rate]:
    owned = owned_n = 0
    health_error = 0.0
    health_n = 0
    for predicted, truth, _ in pairs:
        observed = [e for e in predicted.entities if e.staleness == 0]
        matches = _match_frame(observed, truth.entities, iou_threshold)
        by_pred = {e.entity_id: e for e in observed}
        for gt in truth.entities:
            if gt.screen_bbox is None:
                continue
            owned_n += 1
            pred = by_pred.get(matches[gt.entity_id]) if gt.entity_id in matches else None
            if pred is None:
                continue
            owned += pred.ownership == gt.ownership
            if pred.health is not None and gt.health is not None:
                health_error += abs(pred.health - gt.health)
                health_n += 1
    return Rate.of(owned, owned_n), Rate.of(health_error, health_n)


# --------------------------------------------------------------------------- #
# Minimap                                                                     #
# --------------------------------------------------------------------------- #


def _minimap_scores(pairs: Sequence[Pair], radius: float) -> tuple[Rate, Rate]:
    matched = predicted_n = truth_n = 0
    for predicted, _, engine_frame in pairs:
        if predicted.minimap is None:
            continue
        targets = [
            e.world_pos()
            for e in engine_frame.entities
            if e.kind in (EntityKind.UNIT, EntityKind.BUILDING)
            and e.visibility in (None, "visible")
            and e.world_pos() is not None
        ]
        blips = [b.world_pos for b in predicted.minimap.blips]
        predicted_n += len(blips)
        truth_n += len(targets)
        free = list(targets)
        for blip in blips:
            best = min(
                (t for t in free if t is not None),
                key=lambda t: (t.x - blip.x) ** 2 + (t.y - blip.y) ** 2,
                default=None,
            )
            if best is not None and (best.x - blip.x) ** 2 + (best.y - blip.y) ** 2 <= radius**2:
                free.remove(best)
                matched += 1
    return Rate.of(matched, predicted_n), Rate.of(matched, truth_n)
