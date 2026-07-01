"""P1 — per-consumer pipeline variant tests."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.preprocessing import hud_pipeline, scene_pipeline

from .preprocessing_support import make_pattern_frame


def test_hud_pipeline_preserves_invariants() -> None:
    frame = make_pattern_frame()
    out = hud_pipeline()(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape
    assert out.image.dtype == np.uint8


def test_hud_pipeline_with_region_preserves_invariants() -> None:
    frame = make_pattern_frame()
    region = ScreenBBox(x=0.0, y=0.0, width=32.0, height=6.0)
    out = hud_pipeline(region)(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape
    assert out.image.dtype == np.uint8


def test_scene_pipeline_preserves_invariants() -> None:
    frame = make_pattern_frame()
    out = scene_pipeline()(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape
    assert out.image.dtype == np.uint8


def test_variants_are_deterministic() -> None:
    frame = make_pattern_frame()
    assert np.array_equal(scene_pipeline()(frame).image, scene_pipeline()(frame).image)


def test_variants_expose_their_steps() -> None:
    assert len(scene_pipeline().steps) == 3
    assert len(hud_pipeline().steps) == 2
