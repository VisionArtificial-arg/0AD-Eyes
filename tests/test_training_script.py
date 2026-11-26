"""MOD-2 — the standalone YOLO11-seg training script's helpers (no training run here)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from training.train_yolo_seg import (
    SIDECAR_FORMAT,
    metrics_summary,
    parse_args,
    write_sidecar,
)


def test_defaults_train_at_1280_with_early_stopping() -> None:
    args = parse_args(["--data", "d/data.yaml"])
    assert args.imgsz == 1280 and args.patience > 0 and args.model == "yolo11n-seg.pt"


def test_data_yaml_path_is_rewritten_to_where_the_dataset_is(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    from training.train_yolo_seg import class_names, prepare_data_yaml

    src = tmp_path / "upload"
    src.mkdir()
    (src / "data.yaml").write_text(
        "path: /home/someone/else/yolo\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: resource_wood\n  6: unit\n"
    )
    fixed = prepare_data_yaml(src / "data.yaml", tmp_path / "runs" / "v1.data.yaml")
    assert fixed == tmp_path / "runs" / "v1.data.yaml"
    assert str(src.resolve()) in fixed.read_text()
    assert class_names(fixed) == {0: "resource_wood", 6: "unit"}


def test_metrics_summary_reads_box_and_mask_maps() -> None:
    metric = SimpleNamespace(map50=0.8, map=0.5, mp=0.9, mr=0.7, maps=[0.4, 0.6])
    results = SimpleNamespace(names={0: "building", 1: "unit"}, box=metric, seg=metric)
    summary = metrics_summary(results)
    assert summary["box"]["map50"] == 0.8 and summary["seg"]["map50_95"] == 0.5
    assert summary["box"]["per_class_map50_95"] == {"building": 0.4, "unit": 0.6}


def test_sidecar_carries_the_class_map_and_metrics(tmp_path: Path) -> None:
    path = write_sidecar(
        tmp_path,
        model="yolo11n-seg.pt",
        imgsz=1280,
        names={1: "unit", 0: "building"},
        epochs=60,
        test_metrics={"box": {"map50": 0.8}},
        data_report={"duplicates_dropped": 3},
    )
    sidecar = json.loads(path.read_text())
    assert sidecar["format"] == SIDECAR_FORMAT and sidecar["weights"] == "weights/best.pt"
    assert sidecar["names"] == {"0": "building", "1": "unit"} and sidecar["imgsz"] == 1280
    assert sidecar["data_report"]["duplicates_dropped"] == 3
