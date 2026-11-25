"""MOD-1 — segmentation audit: confusion matrix → IoU / mIoU / entity mIoU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.infrastructure.data.engine_dataset import EngineDatasetRun
from zero_ad_eyes.infrastructure.data.segmentation_audit import (
    ConfusionMatrix,
    audit_segmentation,
)

from ._engine_fixtures import BUILDING, GRASS, UNIT, maps, write_run


def test_perfect_prediction_scores_one() -> None:
    truth = maps()[0]
    matrix = ConfusionMatrix()
    matrix.add(truth, truth)
    audit = matrix.result()
    assert audit.miou == 1.0 and audit.miou_entities == 1.0 and audit.pixel_accuracy == 1.0
    assert set(audit.iou) == {"terrain_grass", "building", "unit"}


def test_entity_miou_ignores_the_terrain_that_flatters_plain_miou() -> None:
    truth = maps()[0]
    all_grass = np.full_like(truth, GRASS)
    matrix = ConfusionMatrix()
    matrix.add(all_grass, truth)
    audit = matrix.result()
    assert audit.miou_entities == 0.0  # no unit or building pixel found
    assert audit.iou["terrain_grass"] > 0.8
    assert audit.pixel_accuracy is not None and audit.pixel_accuracy > 0.8


def test_iou_is_per_class_overlap() -> None:
    truth = np.array([[UNIT, UNIT, GRASS, GRASS]], np.uint8)
    pred = np.array([[UNIT, GRASS, GRASS, BUILDING]], np.uint8)
    matrix = ConfusionMatrix()
    matrix.add(pred, truth)
    audit = matrix.result()
    assert audit.iou["unit"] == pytest.approx(0.5)  # 1 hit / (2 truth + 1 pred - 1)
    assert audit.iou["terrain_grass"] == pytest.approx(1 / 3)
    assert audit.iou["building"] == 0.0  # predicted only
    assert audit.miou == pytest.approx((0.5 + 1 / 3) / 2)  # building has no truth pixels


def test_shape_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        ConfusionMatrix().add(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))


def test_empty_audit_is_unmeasured() -> None:
    audit = ConfusionMatrix().result()
    assert audit.frames == 0 and audit.miou is None


def test_audit_runs_a_segmenter_over_every_frame(tmp_path: Path) -> None:
    run = EngineDatasetRun.load(write_run(tmp_path, frames=2))
    seen: list[tuple[int, int]] = []

    def oracle(frame: Frame) -> np.ndarray:
        seen.append(frame.image.shape[:2])
        return maps()[0]

    audit = audit_segmentation(oracle, [run])
    assert audit.frames == 2 and audit.miou == 1.0
    assert seen == [(40, 60), (40, 60)]
