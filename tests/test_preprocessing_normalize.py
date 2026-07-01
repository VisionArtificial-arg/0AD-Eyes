"""P3 — intensity normalization tests."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.infrastructure.preprocessing import BrightnessContrast, MinMaxNormalize

from .preprocessing_support import make_pattern_frame


def test_minmax_stretches_to_full_range() -> None:
    frame = make_pattern_frame()
    out = MinMaxNormalize().transform(frame.image)
    assert out.min() == 0
    assert out.max() == 255
    assert out.dtype == np.uint8


def test_minmax_leaves_flat_image_unchanged() -> None:
    flat = np.full((8, 8, 3), 50, dtype=np.uint8)
    out = MinMaxNormalize().transform(flat)
    assert np.array_equal(out, flat)


def test_minmax_preserves_meta() -> None:
    frame = make_pattern_frame()
    out = MinMaxNormalize()(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape


def test_brightness_contrast_clips_and_is_deterministic() -> None:
    frame = make_pattern_frame()
    step = BrightnessContrast(alpha=2.0, beta=100.0)
    out = step(frame)
    assert out.image.dtype == np.uint8
    assert out.image.max() <= 255
    assert np.array_equal(step(frame).image, out.image)
