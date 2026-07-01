"""P7 — region-of-interest gating tests."""

from __future__ import annotations

import numpy as np
import pytest

from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.preprocessing import GateMode, RoiGate

from .preprocessing_support import make_pattern_frame


def test_roi_mask_preserves_shape_and_zeros_outside() -> None:
    frame = make_pattern_frame()
    bbox = ScreenBBox(x=6.0, y=4.0, width=14.0, height=8.0)
    out = RoiGate(bbox, mode=GateMode.MASK)(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape
    assert np.array_equal(out.image[4:12, 6:20], frame.image[4:12, 6:20])
    assert out.image[0, 0].sum() == 0


def test_roi_crop_reduces_shape() -> None:
    frame = make_pattern_frame()
    bbox = ScreenBBox(x=6.0, y=4.0, width=14.0, height=8.0)
    out = RoiGate(bbox, mode=GateMode.CROP)(frame)
    assert out.image.shape == (8, 14, 3)
    assert out.meta == frame.meta  # meta preserved as capture provenance


def test_roi_crop_records_and_accumulates_offset() -> None:
    # v0.2: CROP shifts the origin, so crop_offset tracks where the sub-image sits.
    frame = make_pattern_frame()
    once = RoiGate(ScreenBBox(x=6.0, y=4.0, width=14.0, height=8.0), mode=GateMode.CROP)(frame)
    assert once.crop_offset == (6, 4)

    twice = RoiGate(ScreenBBox(x=2.0, y=1.0, width=5.0, height=4.0), mode=GateMode.CROP)(once)
    assert twice.crop_offset == (8, 5)  # offsets compose through nested crops


def test_roi_mask_leaves_offset_untouched() -> None:
    frame = make_pattern_frame()
    out = RoiGate(ScreenBBox(x=6.0, y=4.0, width=14.0, height=8.0), mode=GateMode.MASK)(frame)
    assert out.crop_offset == (0, 0)


def test_roi_clamps_to_frame_bounds() -> None:
    frame = make_pattern_frame()
    bbox = ScreenBBox(x=0.0, y=0.0, width=1000.0, height=1000.0)
    out = RoiGate(bbox, mode=GateMode.CROP)(frame)
    assert out.image.shape == frame.image.shape


def test_roi_empty_raises() -> None:
    frame = make_pattern_frame()
    off = ScreenBBox(x=1000.0, y=1000.0, width=5.0, height=5.0)
    with pytest.raises(ValueError):
        RoiGate(off)(frame)
