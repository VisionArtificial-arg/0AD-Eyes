"""Engine-fork dataset contract: run layout, instance-mask codec, loader (ML2, D6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.data.engine_dataset import (
    EngineDatasetRun,
    decode_mask,
    encode_mask,
    find_runs,
    instance_stats,
    read_mask,
)
from zero_ad_eyes.infrastructure.data.ground_truth import GroundTruthAligner
from zero_ad_eyes.infrastructure.model.segmentation_classes import (
    CLASS_NAMES,
    ENTITY_CLASS_IDS,
    KIND_BY_CLASS_ID,
    NUM_CLASSES,
)

from ._engine_fixtures import BUILDING, GRASS, UNIT, H, W, maps, write_run


def test_class_table_matches_the_engine_enum() -> None:
    assert NUM_CLASSES == 17
    assert set(CLASS_NAMES) == set(KIND_BY_CLASS_ID) == set(range(17))
    assert KIND_BY_CLASS_ID[UNIT] is EntityKind.UNIT
    assert KIND_BY_CLASS_ID[BUILDING] is EntityKind.BUILDING
    assert GRASS not in ENTITY_CLASS_IDS


def test_mask_round_trips_class_and_16_bit_instance() -> None:
    classes, instances = maps()
    decoded_classes, decoded_instances = decode_mask(encode_mask(classes, instances))
    assert np.array_equal(decoded_classes, classes)
    assert np.array_equal(decoded_instances, instances)


def test_mask_round_trips_through_png(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    classes, instances = decode_mask(read_mask(run / "mask" / "000000.png"))
    assert np.array_equal(classes, maps()[0])
    assert np.array_equal(instances, maps()[1])


def test_instance_stats_gives_tight_boxes_and_pixel_counts() -> None:
    stats = instance_stats(maps()[1])
    assert set(stats) == {1, 300}
    assert stats[1].bbox == ScreenBBox(x=10, y=5, width=4, height=10)
    assert stats[1].pixels == 40
    assert stats[300].bbox == ScreenBBox(x=30, y=20, width=20, height=15)


def test_instance_stats_of_an_empty_mask_is_empty() -> None:
    assert instance_stats(np.zeros((H, W), np.uint16)) == {}


def test_labelled_truth_boxes_only_visible_entities_with_pixels(tmp_path: Path) -> None:
    run = EngineDatasetRun.load(write_run(tmp_path))
    truth = run.labelled_truth(run.frames()[0])
    boxes = {e.entity_id: e.bbox for e in truth.entities}
    assert boxes[101] == ScreenBBox(x=10, y=5, width=4, height=10)
    assert boxes[202] == ScreenBBox(x=30, y=20, width=20, height=15)
    assert boxes[303] is None  # no mask pixels → occluded
    assert boxes[404] is None  # hidden
    assert truth.camera is not None and truth.players[0].civ == "athen"


def test_min_pixels_drops_slivers(tmp_path: Path) -> None:
    run = EngineDatasetRun.load(write_run(tmp_path))
    truth = run.labelled_truth(run.frames()[0], min_pixels=50)
    boxes = {e.entity_id: e.bbox for e in truth.entities}
    assert boxes[101] is None  # 40 px < 50
    assert boxes[202] is not None


def test_frames_skips_incomplete_frames(tmp_path: Path) -> None:
    root = write_run(tmp_path, frames=3)
    (root / "truth" / "000001.json").unlink()
    run = EngineDatasetRun.load(root)
    assert [f.stem for f in run.frames()] == ["000000", "000002"]
    assert all(f.rgb_nogui is None for f in run.frames())


def test_export_feeds_the_existing_aligner(tmp_path: Path) -> None:
    run = EngineDatasetRun.load(write_run(tmp_path))
    export = run.to_export()
    assert export.match_id == "a1b2c3d4" and len(export.frames) == 2
    aligner = GroundTruthAligner(export)
    assert len(aligner.frame_for_id(0).entities) == 4  # type: ignore[union-attr]


def test_observer_run_needs_an_explicit_self_player(tmp_path: Path) -> None:
    run = EngineDatasetRun.load(write_run(tmp_path, view_player=-1))
    with pytest.raises(ValueError, match="observer"):
        run.to_export()
    assert run.to_export(self_player=1).self_player == 1


def test_unknown_mask_encoding_is_rejected(tmp_path: Path) -> None:
    root = write_run(tmp_path)
    manifest = json.loads((root / "run.json").read_text())
    (root / "run.json").write_text(json.dumps({**manifest, "mask_encoding": "palette/0"}))
    with pytest.raises(ValueError, match="mask encoding"):
        EngineDatasetRun.load(root)


def test_find_runs_lists_every_run_directory(tmp_path: Path) -> None:
    write_run(tmp_path)
    (tmp_path / "not-a-run").mkdir()
    assert [r.manifest.run_id for r in find_runs(tmp_path)] == ["a1b2c3d4"]
