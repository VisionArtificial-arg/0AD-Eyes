"""Tests for the offline video and image-folder sources (A2)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.infrastructure.acquisition.offline import ImageFolderSource, VideoFileSource


class FakeCapture:
    """A minimal ``cv2.VideoCapture`` stand-in yielding scripted frames."""

    def __init__(self, images: list[np.ndarray], fps: float) -> None:
        self._images = images
        self._fps = fps
        self._i = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - matches cv2 API
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0  # POS_MSEC unknown -> source falls back to frame_id / fps

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._i >= len(self._images):
            return False, None
        image = self._images[self._i]
        self._i += 1
        return True, image

    def release(self) -> None:
        self.released = True


def test_video_source_replays_frames_with_indexed_timestamps() -> None:
    images = [np.full((8, 12, 3), i, dtype=np.uint8) for i in range(3)]
    capture = FakeCapture(images, fps=10.0)
    source = VideoFileSource("game.mp4", capture_factory=lambda: capture)

    frames = list(source.frames())

    assert [f.meta.frame_id for f in frames] == [0, 1, 2]
    assert [f.meta.timestamp for f in frames] == pytest.approx([0.0, 0.1, 0.2])
    assert all(f.meta.source == "game" for f in frames)
    assert all(isinstance(f, Frame) and f.meta.width == 12 for f in frames)
    assert capture.released


def test_video_source_raises_when_unopenable() -> None:
    class ClosedCapture:
        def isOpened(self) -> bool:  # noqa: N802
            return False

        def release(self) -> None: ...

    source = VideoFileSource("missing.mp4", capture_factory=ClosedCapture)
    with pytest.raises(OSError):
        list(source.frames())


def test_image_folder_source_reads_sorted_images(tmp_path) -> None:
    for name in ("frame_002.png", "frame_000.png", "frame_001.png"):
        cv2.imwrite(str(tmp_path / name), np.zeros((6, 9, 3), dtype=np.uint8))

    source = ImageFolderSource(tmp_path, fps=5.0, source="rec")
    frames = list(source.frames())

    assert [f.meta.frame_id for f in frames] == [0, 1, 2]
    assert [f.meta.timestamp for f in frames] == pytest.approx([0.0, 0.2, 0.4])
    assert all(f.meta.source == "rec" for f in frames)
    assert all(f.meta.width == 9 and f.meta.height == 6 for f in frames)


def test_image_folder_source_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError):
        ImageFolderSource("somewhere", fps=0.0)
