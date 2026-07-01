"""E10 mask post-processing tests — synthetic binary masks, no display."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.infrastructure.perception.masks import (
    clean_mask,
    connected_components,
    find_contours,
    morphological_close,
    morphological_open,
    to_binary,
)


def _mask(height: int = 60, width: int = 80) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def test_to_binary_thresholds_grayscale() -> None:
    img = _mask()
    img[:, :40] = 200
    binary = to_binary(img, threshold=127)
    assert set(np.unique(binary)).issubset({0, 255})
    assert binary[0, 0] == 255
    assert binary[0, 60] == 0


def test_open_removes_speckle_noise() -> None:
    mask = _mask()
    cv2.rectangle(mask, (20, 20), (40, 40), 255, -1)
    mask[5, 5] = 255  # lone noise pixel
    opened = morphological_open(mask, ksize=3)
    assert opened[5, 5] == 0
    assert opened[30, 30] == 255


def test_close_fills_small_hole() -> None:
    mask = _mask()
    cv2.rectangle(mask, (20, 20), (45, 45), 255, -1)
    mask[32, 32] = 0  # a hole
    closed = morphological_close(mask, ksize=3)
    assert closed[32, 32] == 255


def test_connected_components_counts_and_bounds_blobs() -> None:
    mask = _mask()
    cv2.rectangle(mask, (5, 5), (15, 15), 255, -1)
    cv2.rectangle(mask, (50, 30), (70, 50), 255, -1)
    comps = connected_components(mask, min_area=5)
    assert len(comps) == 2
    # Sorted by area descending: the larger rectangle first.
    assert comps[0].area > comps[1].area
    biggest = comps[0]
    assert 48 <= biggest.bbox.x <= 52
    assert biggest.bbox.width >= 18


def test_connected_components_min_area_filters() -> None:
    mask = _mask()
    cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)
    mask[50, 50] = 255
    comps = connected_components(mask, min_area=10)
    assert len(comps) == 1


def test_find_contours_returns_polygon() -> None:
    mask = _mask()
    cv2.rectangle(mask, (10, 10), (40, 40), 255, -1)
    polys = find_contours(mask, min_area=50)
    assert len(polys) == 1
    xs = [p.x for p in polys[0]]
    ys = [p.y for p in polys[0]]
    assert min(xs) <= 11 and max(xs) >= 39
    assert min(ys) <= 11 and max(ys) >= 39


def test_clean_mask_denoises_then_consolidates() -> None:
    mask = _mask()
    cv2.rectangle(mask, (20, 20), (50, 50), 255, -1)
    mask[2, 2] = 255
    cleaned = clean_mask(mask, open_ksize=3, close_ksize=5)
    comps = connected_components(cleaned, min_area=5)
    assert len(comps) == 1
