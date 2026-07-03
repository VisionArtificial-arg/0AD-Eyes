"""Tests for the portal+PipeWire capture backend (A1) — framing, grabber, wiring.

The live portal/GStreamer path is integration-only (needs a compositor + consent);
here we exercise the wire-protocol decode and the grabber's threading/lifecycle with
an injected in-memory stream, so nothing spawns a subprocess or touches a display.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from zero_ad_eyes.infrastructure.acquisition import (
    PortalPipeWireGrabber,
    ScreenCaptureSource,
    read_frame,
)
from zero_ad_eyes.infrastructure.acquisition.screen import CaptureRegion
from zero_ad_eyes.interface.default_config import default_config


def _encode(image: np.ndarray, *, pad: int = 0) -> bytes:
    """Encode a (h,w,3) BGR frame in the helper's wire format, optional row padding."""

    height, width, _ = image.shape
    stride = width * 3 + pad
    rows = bytearray()
    for r in range(height):
        rows += image[r].tobytes()
        rows += b"\x00" * pad
    return struct.pack("<III", width, height, stride) + bytes(rows)


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_read_frame_round_trips_a_packed_frame() -> None:
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    frame = read_frame(io.BytesIO(_encode(image)))
    assert frame is not None
    assert frame.shape == (4, 6, 3)
    assert np.array_equal(frame, image)


def test_read_frame_depads_row_stride() -> None:
    # Rows padded to stride > width*3 (common from GStreamer) must be sliced back.
    image = np.arange(2 * 5 * 3, dtype=np.uint8).reshape(2, 5, 3)
    frame = read_frame(io.BytesIO(_encode(image, pad=7)))
    assert frame is not None
    assert np.array_equal(frame, image)


def test_read_frame_returns_none_on_end_of_stream() -> None:
    assert read_frame(io.BytesIO(b"")) is None
    # truncated payload (header promises more than provided)
    assert read_frame(io.BytesIO(struct.pack("<III", 4, 4, 12) + b"\x00" * 3)) is None


def test_grabber_returns_a_streamed_frame() -> None:
    image = np.full((6, 8, 3), 200, dtype=np.uint8)
    proc = _FakeProc()
    grabber = PortalPipeWireGrabber(
        startup_timeout=5.0, stream_factory=lambda: (proc, io.BytesIO(_encode(image)))
    )

    out = grabber.grab()

    assert out.shape == (6, 8, 3)
    assert np.array_equal(out, image)


def test_grabber_crops_to_region() -> None:
    image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
    region = CaptureRegion(top=2, left=3, width=5, height=4)
    grabber = PortalPipeWireGrabber(
        startup_timeout=5.0,
        region=region,
        stream_factory=lambda: (_FakeProc(), io.BytesIO(_encode(image))),
    )

    out = grabber.grab()

    assert out.shape == (4, 5, 3)
    assert np.array_equal(out, image[2:6, 3:8])


def test_grabber_times_out_when_no_frame_arrives() -> None:
    grabber = PortalPipeWireGrabber(
        startup_timeout=0.3, stream_factory=lambda: (_FakeProc(), io.BytesIO(b""))
    )
    with pytest.raises(OSError, match="no frame"):
        grabber.grab()


def test_grabber_close_terminates_the_helper() -> None:
    proc = _FakeProc()
    grabber = PortalPipeWireGrabber(
        startup_timeout=5.0,
        stream_factory=lambda: (proc, io.BytesIO(_encode(np.zeros((2, 2, 3), np.uint8)))),
    )
    grabber.grab()

    grabber.close()

    assert proc.terminated is True


def test_from_settings_selects_portal_backend() -> None:
    settings = default_config().acquisition.model_copy(update={"capture_backend": "portal"})

    source = ScreenCaptureSource.from_settings(settings)

    assert isinstance(source._grabber, PortalPipeWireGrabber)
