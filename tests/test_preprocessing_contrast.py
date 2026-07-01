"""P5 — CLAHE contrast enhancement tests."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.infrastructure.preprocessing import ClaheContrast

from .preprocessing_support import make_pattern_frame


def test_clahe_on_bgr_preserves_shape_and_meta() -> None:
    frame = make_pattern_frame()
    out = ClaheContrast()(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape
    assert out.image.dtype == np.uint8


def test_clahe_on_grayscale() -> None:
    gray = (np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)) % 200
    out = ClaheContrast().transform(gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


def test_clahe_is_deterministic() -> None:
    frame = make_pattern_frame()
    step = ClaheContrast()
    assert np.array_equal(step(frame).image, step(frame).image)
