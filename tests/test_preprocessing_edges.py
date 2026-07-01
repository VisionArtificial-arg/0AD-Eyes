"""P6 — edge detection tests."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.infrastructure.preprocessing import (
    EdgeDetect,
    EdgeOperator,
    canny_edges,
    sobel_edges,
)

from .preprocessing_support import make_pattern_frame


def test_canny_helper_returns_single_channel() -> None:
    frame = make_pattern_frame()
    edges = canny_edges(frame.image)
    assert edges.shape == frame.image.shape[:2]
    assert edges.dtype == np.uint8


def test_sobel_helper_returns_single_channel() -> None:
    frame = make_pattern_frame()
    edges = sobel_edges(frame.image)
    assert edges.shape == frame.image.shape[:2]
    assert edges.dtype == np.uint8


def test_edge_detect_as_bgr_keeps_three_channels() -> None:
    frame = make_pattern_frame()
    out = EdgeDetect(EdgeOperator.CANNY, as_bgr=True)(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape


def test_edge_detect_single_channel_option() -> None:
    frame = make_pattern_frame()
    out = EdgeDetect(EdgeOperator.SOBEL, as_bgr=False)(frame)
    assert out.image.shape == frame.image.shape[:2]


def test_edge_detect_is_deterministic() -> None:
    frame = make_pattern_frame()
    step = EdgeDetect()
    assert np.array_equal(step(frame).image, step(frame).image)
