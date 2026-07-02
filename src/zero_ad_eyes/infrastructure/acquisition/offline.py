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

from .recording import RecordingManifest

CaptureFactory = Callable[[], Any]
ImageReader = Callable[[str], Any]


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
        extensions: tuple[str, ...],
        fps: float,
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


class RecordedVideoSource:
    """Replays a recording (a video **plus its sidecar**) with the capture's true clock.

    :class:`VideoFileSource` alone renumbers frames ``0..N`` and derives timestamps
    from the container PTS — dropping the real ``frame_id``/``timestamp`` that the
    ``--record`` sidecar (A5, :class:`~.recording.RecordingManifest`) preserved. This
    source is the reader half: it restamps each replayed frame from that sidecar so
    downstream world models carry the capture's own clock, which is the precondition
    for aligning them to an engine ground-truth export by ``frame_id``/``timestamp``
    (#2, D6). Pixels come from the video; only the clock (and source name) is restored.

    The inner video source and the manifest are injectable so the restamping is
    unit-testable without a codec; by default the manifest loads from the sibling
    ``.json`` (same stem as the video).
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        video: Any | None = None,
        manifest: RecordingManifest | None = None,
    ) -> None:
        self._video_path = Path(video_path)
        self._video = video if video is not None else VideoFileSource(self._video_path)
        self._manifest = (
            manifest
            if manifest is not None
            else RecordingManifest.load(self._video_path.with_suffix(".json"))
        )

    def frames(self) -> Iterator[Frame]:
        stamps = self._manifest.stamps
        source = self._manifest.source
        count = 0
        for frame in self._video.frames():
            if count >= len(stamps):
                raise OSError(
                    f"recording {self._video_path.name}: video has more frames than its "
                    f"sidecar lists ({len(stamps)}) — sidecar and video are out of step"
                )
            stamp = stamps[count]
            yield Frame(
                image=frame.image,
                meta=FrameMeta(
                    frame_id=stamp.frame_id,
                    timestamp=stamp.timestamp,
                    source=source,
                    width=frame.meta.width,
                    height=frame.meta.height,
                ),
                crop_offset=frame.crop_offset,
            )
            count += 1
        if count != len(stamps):
            raise OSError(
                f"recording {self._video_path.name}: sidecar lists {len(stamps)} frames "
                f"but the video had {count} — sidecar and video are out of step"
            )
