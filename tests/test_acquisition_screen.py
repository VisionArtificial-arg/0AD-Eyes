"""Tests for the live screen-capture source (A1) with a fake grabber."""

from __future__ import annotations

import os

import numpy as np
import pytest

from zero_ad_eyes.infrastructure.acquisition.screen import (
    CaptureRegion,
    ScreenCaptureSource,
    _to_bgr,
)


class FakeGrabber:
    """Returns synthetic BGRA screenshots; records how many grabs happened."""

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self.grabs = 0

    def grab(self) -> np.ndarray:
        self.grabs += 1
        # BGRA, distinct per grab so we can tell frames apart.
        frame = np.full((self._height, self._width, 4), self.grabs, dtype=np.uint8)
        return frame


def test_to_bgr_strips_alpha_channel() -> None:
    bgra = np.zeros((4, 6, 4), dtype=np.uint8)
    bgr = _to_bgr(bgra)
    assert bgr.shape == (4, 6, 3)


def test_to_bgr_passes_through_three_channel() -> None:
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    assert _to_bgr(bgr).shape == (4, 6, 3)


def test_screen_source_emits_bounded_bgr_frames_with_live_meta() -> None:
    grabber = FakeGrabber(width=32, height=24)
    source = ScreenCaptureSource(
        target_fps=1000.0,
        grabber=grabber,
        max_frames=3,
        clock=lambda: 0.0,
        sleep=lambda _: None,
    )

    frames = list(source.frames())

    assert len(frames) == 3
    assert [f.meta.frame_id for f in frames] == [0, 1, 2]
    for frame in frames:
        assert frame.meta.source == "live"
        assert frame.meta.width == 32
        assert frame.meta.height == 24
        assert frame.image.shape == (24, 32, 3)  # alpha stripped
    assert grabber.grabs == 3


def test_capture_region_maps_to_mss_dict() -> None:
    region = CaptureRegion(top=10, left=20, width=100, height=50)
    assert region.as_mss() == {"top": 10, "left": 20, "width": 100, "height": 50}


@pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="no display: skip real mss screen capture",
)
def test_real_mss_capture_smoke() -> None:
    source = ScreenCaptureSource(target_fps=1000.0, max_frames=1)
    try:
        frames = list(source.frames())
    except Exception as exc:  # noqa: BLE001 - headless/virtual X may lack GetImage
        pytest.skip(f"display present but capture unavailable: {exc}")
    assert len(frames) == 1
    assert frames[0].image.ndim == 3
