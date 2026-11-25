"""MOD-4 — YOLO11-seg adapter: instances → Detections, parity with the MP2 contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.model import YoloSegPerceptionModel, assert_satisfies_contract

NAMES = {0: "resource_wood", 5: "building", 6: "unit", 9: "mystery"}


def _frame(frame_id: int = 3) -> Frame:
    return Frame(
        image=np.zeros((90, 160, 3), np.uint8),
        meta=FrameMeta(frame_id=frame_id, timestamp=0.0, source="t", width=160, height=90),
    )


def _result(
    boxes: list[list[float]], conf: list[float], cls: list[int], masks: list[Any] | None
) -> Any:
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array(boxes).reshape(-1, 4), conf=np.array(conf), cls=np.array(cls)
        ),
        masks=None if masks is None else SimpleNamespace(xy=masks),
    )


def _model(result: Any, **kwargs: Any) -> YoloSegPerceptionModel:
    return YoloSegPerceptionModel(lambda image: result, NAMES, **kwargs)


def test_instances_become_learned_detections_with_kind_type_and_outline() -> None:
    square = np.array([[10, 10], [30, 10], [30, 40], [10, 40]], np.float32)
    result = _result([[10, 10, 30, 40], [50, 5, 90, 25]], [0.9, 0.4], [6, 0], [square, square + 40])
    dets = _model(result).infer(_frame())
    assert dets.frame_id == 3 and len(dets) == 2
    unit, tree = dets.items
    assert unit.kind is EntityKind.UNIT and unit.entity_type == "unit"
    assert unit.bbox == ScreenBBox(x=10, y=10, width=20, height=30)
    assert unit.confidence.value == pytest.approx(0.9)
    assert unit.confidence.provenance is Provenance.LEARNED
    assert len(unit.contour) == 4 and unit.contour[0].x == 10
    assert tree.kind is EntityKind.RESOURCE_NODE and tree.entity_type == "resource_wood"


def test_unknown_classes_are_other_and_missing_masks_give_no_outline() -> None:
    dets = _model(_result([[0, 0, 5, 5]], [0.5], [9], None)).infer(_frame())
    assert dets.items[0].kind is EntityKind.OTHER and dets.items[0].entity_type == "mystery"
    assert dets.items[0].contour == ()


def test_long_outlines_are_simplified() -> None:
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.stack([80 + 30 * np.cos(angles), 45 + 30 * np.sin(angles)], axis=1).astype(
        np.float32
    )
    dets = _model(_result([[50, 15, 110, 75]], [0.8], [5], [circle]), max_contour_points=24).infer(
        _frame()
    )
    assert 3 <= len(dets.items[0].contour) <= 24


def test_no_instances_is_an_empty_frame() -> None:
    assert len(_model(_result([], [], [], [])).infer(_frame())) == 0


def test_satisfies_the_model_contract() -> None:
    result = _result(
        [[10, 10, 30, 40]], [0.7], [6], [np.array([[10, 10], [30, 10], [30, 40]], np.float32)]
    )
    assert_satisfies_contract(_model(result), _frame())


def _saved_untrained_model(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    from ultralytics import YOLO

    model = YOLO("yolo11n-seg.yaml")  # architecture only: no download, untrained
    (tmp_path / "weights").mkdir()
    model.save(str(tmp_path / "weights" / "best.pt"))
    names = {str(k): v for k, v in model.names.items()}
    return tmp_path / "sidecar.json", names


def test_from_sidecar_loads_real_weights_and_infers(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    sidecar, names = _saved_untrained_model(tmp_path)
    sidecar.write_text(
        json.dumps(
            {
                "format": "zero-ad-eyes/yolo-seg/1",
                "weights": "weights/best.pt",
                "imgsz": 64,
                "names": names,
            }
        )
    )
    model = YoloSegPerceptionModel.from_sidecar(
        sidecar, conf_threshold=0.25, iou_threshold=0.6, max_detections=10, device="cpu"
    )
    assert_satisfies_contract(model, _frame())


def test_from_sidecar_rejects_mismatched_names_and_foreign_files(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    sidecar, _ = _saved_untrained_model(tmp_path)
    kwargs: dict[str, Any] = {"conf_threshold": 0.25, "iou_threshold": 0.6, "max_detections": 10}
    sidecar.write_text(
        json.dumps(
            {
                "format": "zero-ad-eyes/yolo-seg/1",
                "weights": "weights/best.pt",
                "imgsz": 64,
                "names": {"0": "unit"},
            }
        )
    )
    with pytest.raises(ValueError, match="class names"):
        YoloSegPerceptionModel.from_sidecar(sidecar, **kwargs)
    sidecar.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError, match="sidecar"):
        YoloSegPerceptionModel.from_sidecar(sidecar, **kwargs)
