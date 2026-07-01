"""Tests for the recording/dump passthrough wrapper (A5)."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource
from zero_ad_eyes.infrastructure.acquisition.recorder import FrameRecorder

from .conftest import make_frame


def test_recorder_is_passthrough_and_writes_each_frame(tmp_path) -> None:
    written: list[str] = []

    def fake_writer(path: str, image: np.ndarray) -> bool:
        written.append(path)
        return True

    frames_in = [make_frame(i) for i in range(3)]
    recorder = FrameRecorder(
        InMemoryFrameSource(frames_in),
        tmp_path,
        writer=fake_writer,
    )

    frames_out = list(recorder.frames())

    # Passthrough: same frames, same order, untouched.
    assert [f.meta.frame_id for f in frames_out] == [0, 1, 2]
    assert all(a is b for a, b in zip(frames_out, frames_in, strict=True))
    assert written == [
        str(tmp_path / "frame_000000.png"),
        str(tmp_path / "frame_000001.png"),
        str(tmp_path / "frame_000002.png"),
    ]


def test_recorder_persists_real_images_that_round_trip(tmp_path) -> None:
    frame = make_frame(7, width=10, height=8)
    recorder = FrameRecorder(InMemoryFrameSource([frame]), tmp_path)

    list(recorder.frames())

    written = tmp_path / "frame_000007.png"
    assert written.exists()
    reloaded = cv2.imread(str(written))
    assert reloaded is not None
    assert reloaded.shape == (8, 10, 3)
