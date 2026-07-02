"""Tests for the video-recording passthrough wrapper (A5, ``run --record``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource, VideoFrameRecorder
from zero_ad_eyes.interface.default_config import default_config

from .conftest import make_frame


class FakeSink:
    """Records the ``cv2.VideoWriter`` lifecycle without a real codec."""

    def __init__(self, opened: bool = True) -> None:
        self._opened = opened
        self.written: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2
        return self._opened

    def write(self, image: Any) -> None:
        self.written.append(image)

    def release(self) -> None:
        self.released = True


def _capturing_factory(sink: FakeSink) -> tuple[Any, list[tuple[str, str, float, tuple[int, int]]]]:
    """A factory returning ``sink`` and logging the args it was constructed with."""

    calls: list[tuple[str, str, float, tuple[int, int]]] = []

    def factory(path: str, fourcc: str, fps: float, size: tuple[int, int]) -> FakeSink:
        calls.append((path, fourcc, fps, size))
        return sink

    return factory, calls


def test_recorder_is_passthrough_and_writes_each_frame(tmp_path: Path) -> None:
    sink = FakeSink()
    factory, _calls = _capturing_factory(sink)
    frames_in = [make_frame(i) for i in range(3)]
    recorder = VideoFrameRecorder(
        InMemoryFrameSource(frames_in),
        tmp_path / "out.mkv",
        fourcc="FFV1",
        fps=30.0,
        writer_factory=factory,
    )

    frames_out = list(recorder.frames())

    # Passthrough: identical frames, same order, forwarded untouched.
    assert [f.meta.frame_id for f in frames_out] == [0, 1, 2]
    assert all(a is b for a, b in zip(frames_out, frames_in, strict=True))
    # Every frame's pixels were written to the sink, then it was released.
    assert len(sink.written) == 3
    assert sink.released is True


def test_recorder_opens_sink_lazily_once_with_frame_size(tmp_path: Path) -> None:
    sink = FakeSink()
    factory, calls = _capturing_factory(sink)
    frames_in = [make_frame(i, width=100, height=80) for i in range(4)]
    recorder = VideoFrameRecorder(
        InMemoryFrameSource(frames_in),
        tmp_path / "out.mkv",
        fourcc="FFV1",
        fps=25.0,
        writer_factory=factory,
    )

    list(recorder.frames())

    # Opened exactly once, with (width, height) — cv2's size order — and config codec/fps.
    assert calls == [(str(tmp_path / "out.mkv"), "FFV1", 25.0, (100, 80))]


def test_recorder_never_opens_a_sink_for_an_empty_source(tmp_path: Path) -> None:
    factory, calls = _capturing_factory(FakeSink())
    recorder = VideoFrameRecorder(
        InMemoryFrameSource([]),
        tmp_path / "out.mkv",
        fourcc="FFV1",
        fps=30.0,
        writer_factory=factory,
    )

    assert list(recorder.frames()) == []
    assert calls == []  # no frame => no size => no writer constructed


def test_recorder_raises_when_sink_cannot_open(tmp_path: Path) -> None:
    factory, _calls = _capturing_factory(FakeSink(opened=False))
    recorder = VideoFrameRecorder(
        InMemoryFrameSource([make_frame(0)]),
        tmp_path / "out.mkv",
        fourcc="XVID",
        fps=30.0,
        writer_factory=factory,
    )

    with pytest.raises(OSError, match="cannot open video writer"):
        list(recorder.frames())


def test_from_settings_reads_codec_and_fps_from_config(tmp_path: Path) -> None:
    sink = FakeSink()
    factory, calls = _capturing_factory(sink)
    cfg = default_config()
    recorder = VideoFrameRecorder.from_settings(
        InMemoryFrameSource([make_frame(0, width=8, height=6)]),
        tmp_path / "out.mkv",
        cfg.acquisition,
        writer_factory=factory,
    )

    list(recorder.frames())

    path, fourcc, fps, size = calls[0]
    assert fourcc == cfg.acquisition.record_fourcc
    assert fps == cfg.acquisition.live_fps
    assert size == (8, 6)


def test_rejects_non_positive_fps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        VideoFrameRecorder(InMemoryFrameSource([]), tmp_path / "out.mkv", fourcc="FFV1", fps=0.0)
