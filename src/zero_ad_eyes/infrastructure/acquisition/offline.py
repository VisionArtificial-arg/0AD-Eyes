"""Offline frame readers (EPIC A / A2).

``VideoFileSource`` (via ``cv2.VideoCapture``) and ``ImageFolderSource`` (sorted
image files) implement the same ``FrameSource`` port as the live source and emit
identical ``Frame`` objects, so everything downstream is oblivious to whether it is
replaying a recording or watching the game live.

Timestamps are media-relative (video PTS, or ``frame_id / fps``) rather than wall
clock, so replays are reproducible while remaining a monotonically increasing float
— the only property downstream temporal logic relies on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.world_model import FrameMeta

CaptureFactory = Callable[[], Any]
ImageReader = Callable[[str], Any]

_DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class VideoFileSource:
    """Replays a video file frame by frame as a ``FrameSource``."""

    def __init__(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self._path = Path(path)
        self._source = source if source is not None else self._path.stem
        self._open = capture_factory if capture_factory is not None else self._default_factory

    def _default_factory(self) -> Any:
        return cv2.VideoCapture(str(self._path))

    def frames(self) -> Iterator[Frame]:
        capture = self._open()
        if not capture.isOpened():
            raise OSError(f"cannot open video: {self._path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_id = 0
            while True:
                ok, image = capture.read()
                if not ok or image is None:
                    break
                pos_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if pos_ms > 0.0:
                    timestamp = pos_ms / 1000.0
                elif fps > 0.0:
                    timestamp = frame_id / fps
                else:
                    timestamp = float(frame_id)
                height, width = image.shape[:2]
                yield Frame(
                    image=image,
                    meta=FrameMeta(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        source=self._source,
                        width=width,
                        height=height,
                    ),
                )
                frame_id += 1
        finally:
            capture.release()


class ImageFolderSource:
    """Replays a folder of images (lexicographically sorted) as a ``FrameSource``."""

    def __init__(
        self,
        folder: str | Path,
        *,
        extensions: tuple[str, ...] = _DEFAULT_IMAGE_EXTENSIONS,
        fps: float = 30.0,
        source: str | None = None,
        reader: ImageReader = cv2.imread,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self._folder = Path(folder)
        self._extensions = {ext.lower() for ext in extensions}
        self._fps = fps
        self._source = source if source is not None else self._folder.name
        self._reader = reader

    def _files(self) -> list[Path]:
        if not self._folder.is_dir():
            raise OSError(f"not a directory: {self._folder}")
        return sorted(
            path
            for path in self._folder.iterdir()
            if path.is_file() and path.suffix.lower() in self._extensions
        )

    def frames(self) -> Iterator[Frame]:
        for frame_id, path in enumerate(self._files()):
            image = self._reader(str(path))
            if image is None:
                raise OSError(f"cannot read image: {path}")
            height, width = image.shape[:2]
            yield Frame(
                image=image,
                meta=FrameMeta(
                    frame_id=frame_id,
                    timestamp=frame_id / self._fps,
                    source=self._source,
                    width=width,
                    height=height,
                ),
            )
