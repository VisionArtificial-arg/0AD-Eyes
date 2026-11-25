"""ENG-6 — engine-fork runs → YOLO-seg dataset: split by run, dedup, mask polygons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zero_ad_eyes.infrastructure.data.engine_dataset import find_runs
from zero_ad_eyes.infrastructure.data.yolo_dataset import (
    YOLO_INDEX_BY_CLASS_ID,
    YOLO_NAMES,
    Deduplicator,
    build_yolo_dataset,
    instance_polygons,
    split_runs,
    yolo_label_lines,
)
from zero_ad_eyes.interface.cli import main
from zero_ad_eyes.interface.default_config import default_config

from ._engine_fixtures import BUILDING, UNIT, H, W, maps, write_run

SETTINGS = default_config().dataset


def test_yolo_classes_are_the_seven_entity_classes() -> None:
    assert YOLO_NAMES == (
        "resource_wood",
        "resource_stone",
        "resource_metal",
        "resource_food",
        "fauna",
        "building",
        "unit",
    )


def test_split_is_by_run_deterministic_and_covers_every_split() -> None:
    runs = [f"run{i:02d}" for i in range(10)]
    first = split_runs(runs, val_fraction=0.15, test_fraction=0.15, seed=0)
    assert first == split_runs(list(reversed(runs)), val_fraction=0.15, test_fraction=0.15, seed=0)
    assert sorted(first.values()).count("test") == 2 and sorted(first.values()).count("val") == 2
    assert split_runs(runs, val_fraction=0.15, test_fraction=0.15, seed=1) != first


def test_tiny_run_counts_keep_training_data() -> None:
    assert set(split_runs(["a", "b"], val_fraction=0.15, test_fraction=0.15, seed=0).values()) == {
        "train"
    }
    three = split_runs(["a", "b", "c"], val_fraction=0.15, test_fraction=0.15, seed=0)
    assert sorted(three.values()) == ["test", "train", "val"]


def test_deduplicator_drops_near_identical_frames_only() -> None:
    dedup = Deduplicator(threshold=2.0, thumbnail=8)
    dark = np.zeros((20, 20, 3), np.uint8)
    assert not dedup.is_duplicate(dark)
    assert dedup.is_duplicate(dark + 1)  # 1 grey level apart
    assert not dedup.is_duplicate(np.full_like(dark, 120))
    dedup.reset()
    assert not dedup.is_duplicate(dark)


def test_instance_polygons_label_entities_and_skip_hidden_and_small() -> None:
    classes, instances = maps()
    found = instance_polygons(classes, instances, min_pixels=1, epsilon=0.0)
    assert sorted(c for c, _ in found) == sorted(
        [YOLO_INDEX_BY_CLASS_ID[UNIT], YOLO_INDEX_BY_CLASS_ID[BUILDING]]
    )
    unit = next(p for c, p in found if c == YOLO_INDEX_BY_CLASS_ID[UNIT])
    assert (unit[:, 0].min(), unit[:, 0].max(), unit[:, 1].min(), unit[:, 1].max()) == (
        10,
        13,
        5,
        14,
    )
    assert len(instance_polygons(classes, instances, min_pixels=50, epsilon=0.0)) == 1
    hidden = instance_polygons(classes, instances, min_pixels=1, epsilon=0.0, skip=frozenset({300}))
    assert [c for c, _ in hidden] == [YOLO_INDEX_BY_CLASS_ID[UNIT]]


def test_label_lines_are_normalised() -> None:
    square = np.array([[0, 0], [W, 0], [W, H], [0, H]])
    (line,) = yolo_label_lines([(6, square)], W, H)
    assert line == "6 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000 0.000000 1.000000"


def test_build_writes_ultralytics_layout_split_by_run(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    for i in range(4):
        write_run(src, f"run{i}", frames=3, distinct=True)
    report = build_yolo_dataset(find_runs(src), tmp_path / "yolo", SETTINGS)

    by_split = {split: set(part.runs) for split, part in report.splits.items()}
    assert set().union(*by_split.values()) == {f"run{i}" for i in range(4)}
    assert sum(len(runs) for runs in by_split.values()) == 4  # each run in exactly one split
    for split, part in report.splits.items():
        images = sorted((tmp_path / "yolo" / "images" / split).glob("*.png"))
        assert len(images) == part.frames
        assert {p.name.split("_")[0] for p in images} == by_split[split]
        for image in images:
            label = tmp_path / "yolo" / "labels" / split / f"{image.stem}.txt"
            assert len(label.read_text().splitlines()) == 2  # unit + building
    assert report.instances_per_class["unit"] == 12 and report.instances_per_class["building"] == 12
    data_yaml = (tmp_path / "yolo" / "data.yaml").read_text()
    assert "train: images/train" in data_yaml and "  6: unit" in data_yaml
    assert json.loads((tmp_path / "yolo" / "report.json").read_text())["duplicates_dropped"] == 0


def test_build_drops_duplicate_frames(tmp_path: Path) -> None:
    write_run(tmp_path / "raw", "static", frames=3, distinct=False)
    report = build_yolo_dataset(find_runs(tmp_path / "raw"), tmp_path / "yolo", SETTINGS)
    assert report.duplicates_dropped == 2
    assert report.splits["train"].frames == 1


def test_cli_dataset_build(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_run(tmp_path / "raw", "run0", frames=2, distinct=True)
    code = main(
        ["dataset", "build", "--input", str(tmp_path / "raw"), "--output", str(tmp_path / "y")]
    )
    assert code == 0
    assert "train: 1 runs, 2 frames, 4 instances" in capsys.readouterr().out
    assert main(["dataset", "build", "--input", str(tmp_path / "none"), "--output", "x"]) == 1
