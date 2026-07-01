"""Tests for bounded buffering and the threaded backpressure decorator (A4)."""

from __future__ import annotations

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource
from zero_ad_eyes.infrastructure.acquisition.backpressure import (
    BoundedFrameBuffer,
    ThreadedFrameSource,
)

from .conftest import make_frame


def test_buffer_drops_oldest_when_full() -> None:
    buffer = BoundedFrameBuffer(maxsize=2)
    buffer.put(make_frame(0))
    buffer.put(make_frame(1))
    buffer.put(make_frame(2))  # overflows -> frame 0 dropped

    assert buffer.dropped == 1
    first = buffer.get()
    second = buffer.get()
    assert isinstance(first, Frame) and isinstance(second, Frame)
    assert first.meta.frame_id == 1
    assert second.meta.frame_id == 2


def test_buffer_get_returns_sentinel_when_closed_and_empty() -> None:
    buffer = BoundedFrameBuffer(maxsize=1)
    buffer.close()

    # The sentinel is not a Frame; draining an empty closed buffer never blocks.
    assert not isinstance(buffer.get(), Frame)


def test_threaded_source_forwards_all_frames_when_consumer_keeps_up() -> None:
    frames = [make_frame(i) for i in range(5)]
    threaded = ThreadedFrameSource(InMemoryFrameSource(frames), maxsize=8)

    ids = [frame.meta.frame_id for frame in threaded.frames()]

    assert ids == [0, 1, 2, 3, 4]
    assert threaded.dropped == 0
