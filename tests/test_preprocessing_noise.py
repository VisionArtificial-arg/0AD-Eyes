"""P4 — noise filtering tests."""

from __future__ import annotations

import numpy as np
import pytest

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.infrastructure.preprocessing import (
    BilateralFilter,
    GaussianBlur,
    MedianBlur,
)
from zero_ad_eyes.infrastructure.preprocessing.base import PreprocessStep

from .preprocessing_support import make_pattern_frame


@pytest.mark.parametrize(
    "step",
    [GaussianBlur(3), MedianBlur(3), BilateralFilter(5, 50.0, 50.0)],
)
def test_noise_filters_preserve_invariants_and_determinism(step: PreprocessStep) -> None:
    frame: Frame = make_pattern_frame()
    out = step(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape
    assert out.image.dtype == np.uint8
    assert np.array_equal(out.image, step(frame).image)


def test_gaussian_rejects_even_kernel() -> None:
    with pytest.raises(ValueError):
        GaussianBlur(2)


def test_median_rejects_even_kernel() -> None:
    with pytest.raises(ValueError):
        MedianBlur(2)
