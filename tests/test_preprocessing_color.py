"""P2 — colour-space transform tests."""

from __future__ import annotations

import numpy as np
import pytest

from zero_ad_eyes.infrastructure.preprocessing import ColorSpace, ColorSpaceConvert

from .preprocessing_support import make_pattern_frame


def test_color_convert_preserves_shape_and_meta() -> None:
    frame = make_pattern_frame()
    out = ColorSpaceConvert(ColorSpace.HSV)(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape
    assert out.image.dtype == np.uint8


def test_color_convert_is_deterministic() -> None:
    frame = make_pattern_frame()
    step = ColorSpaceConvert(ColorSpace.LAB)
    assert np.array_equal(step(frame).image, step(frame).image)


def test_color_convert_roundtrip_is_close() -> None:
    frame = make_pattern_frame()
    to_hsv = ColorSpaceConvert(ColorSpace.HSV)
    back = ColorSpaceConvert(ColorSpace.BGR, source=ColorSpace.HSV)
    restored = back(to_hsv(frame)).image
    assert np.abs(restored.astype(int) - frame.image.astype(int)).max() <= 4


def test_color_convert_rejects_identity() -> None:
    with pytest.raises(ValueError):
        ColorSpaceConvert(ColorSpace.BGR, source=ColorSpace.BGR)


def test_color_convert_rejects_unsupported_pair() -> None:
    with pytest.raises(ValueError):
        ColorSpaceConvert(ColorSpace.HSV, source=ColorSpace.LAB)
