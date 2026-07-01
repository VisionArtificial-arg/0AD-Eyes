"""D4 — FogClassifier cell classification tests (synthetic minimap)."""

from __future__ import annotations

import numpy as np
import pytest

from zero_ad_eyes.domain.minimap import FogState
from zero_ad_eyes.infrastructure.minimap import FogClassifier, Segmentation


def _full_segmentation(region: np.ndarray) -> Segmentation:
    h, w = region.shape[:2]
    return Segmentation(
        region=region, mask=np.full((h, w), 255, dtype=np.uint8), origin_x=0, origin_y=0
    )


def test_classifies_the_three_brightness_tiers() -> None:
    region = np.zeros((30, 30, 3), dtype=np.uint8)  # dark ⇒ unexplored
    region[:, 10:20] = 90  # mid ⇒ explored (shroud)
    region[:, 20:] = 220  # bright ⇒ visible
    seg = _full_segmentation(region)

    grid = FogClassifier(rows=1, cols=3).classify(seg)

    assert grid.rows == 1
    assert grid.cols == 3
    assert grid.at(0, 0) is FogState.UNEXPLORED
    assert grid.at(0, 1) is FogState.EXPLORED
    assert grid.at(0, 2) is FogState.VISIBLE


def test_count_helper_totals_cells() -> None:
    region = np.zeros((16, 16, 3), dtype=np.uint8)  # entirely dark
    grid = FogClassifier(rows=4, cols=4).classify(_full_segmentation(region))
    assert grid.count(FogState.UNEXPLORED) == 16


def test_inactive_cells_are_unexplored() -> None:
    region = np.full((16, 16, 3), 255, dtype=np.uint8)  # all bright
    mask = np.zeros((16, 16), dtype=np.uint8)  # but nothing active
    seg = Segmentation(region=region, mask=mask, origin_x=0, origin_y=0)
    grid = FogClassifier(rows=2, cols=2).classify(seg)
    assert grid.count(FogState.UNEXPLORED) == 4


def test_rejects_non_positive_grid() -> None:
    with pytest.raises(ValueError):
        FogClassifier(rows=0, cols=4)
