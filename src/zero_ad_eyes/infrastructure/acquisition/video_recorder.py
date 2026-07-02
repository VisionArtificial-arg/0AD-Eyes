"""Video recording mode (EPIC A / A5) — the ``--record`` sink.

``VideoFrameRecorder`` is a passthrough decorator around *any* ``FrameSource``: as
each frame flows through it is appended to a single video file while the frame
itself is forwarded unchanged. Because it is both a ``FrameSource`` and a wrapper,
it sits transparently in an acquisition chain — the ``run --record`` path wraps the
live source so the pipeline keeps consuming frames while they are persisted.

Contrast with :class:`~zero_ad_eyes.infrastructure.data.dataset_collection.DatasetCollector`,
which is a terminal *drain* (it exhausts the source and returns a manifest of
lossless ``.npy`` frames for labelling). A recorder cannot be a drain: the inference
path still needs the frames, so recording has to be a passthrough.

Design notes:
- The video sink has an open/write-many/release lifecycle, so it does not fit the
  per-frame ``FrameWriter`` callback of :class:`FrameRecorder`; we open lazily on the
  first frame (the frame supplies the ``(width, height)`` the writer needs) and
  release when the source is exhausted.
- The codec (``FFV1`` lossless by default) comes from ``acquisition`` config so the
  fidelity trade-off is a config decision, not a hidden default — lossy compression
  would inject artifacts into the very pixels the classical readers measure (NF3/#2).
- Alongside the video it writes a ``RecordingManifest`` sidecar (same stem, ``.json``)
  carrying each frame's true ``frame_id``/``timestamp`` — the temporal provenance the
  video drops but ground-truth alignment (#2) needs. See :mod:`.recording`.
- The writer construction is injected (``writer_factory``) so tests exercise the
  lifecycle without a real codec or display.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

import cv2

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import FrameSource
from zero_ad_eyes.application.settings import AcquisitionSettings

from .recording import FrameStamp, RecordingManifest


class VideoSink(Protocol):
    """The slice of ``cv2.VideoWriter`` the recorder relies on."""

    def isOpened(self) -> bool: ...  # noqa: N802 - mirrors cv2's method name

    def write(self, image: Any) -> None: ...

    def release(self) -> None: ...


# (path, fourcc, fps, (width, height)) -> an open video sink.
VideoSinkFactory = Callable[[str, str, float, tuple[int, int]], VideoSink]


class VideoFrameRecorder:
    """Persists every passing frame to one video file, forwarding it untouched."""

    def __init__(
        self,
        source: FrameSource,
        path: str | Path,
        *,
        fourcc: str,
        fps: float,
        writer_factory: VideoSinkFactory | None = None,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self._source = source
        self._path = Path(path)
        self._fourcc = fourcc
        self._fps = fps
        self._new_sink = writer_factory if writer_factory is not None else self._default_factory

    @classmethod
    def from_settings(
        cls,
        source: FrameSource,
        path: str | Path,
        settings: AcquisitionSettings,
        *,
        writer_factory: VideoSinkFactory | None = None,
    ) -> VideoFrameRecorder:
        """Build from the ``acquisition`` config: codec from ``record_fourcc``, pacing
        from ``live_fps``. The ``path`` is a composition-root concern (like the
        ``--recording`` input paths), not a tuning default, so it stays an argument."""

        return cls(
            source,
            path,
            fourcc=settings.record_fourcc,
            fps=settings.live_fps,
            writer_factory=writer_factory,
        )

    @staticmethod
    def _default_factory(path: str, fourcc: str, fps: float, size: tuple[int, int]) -> VideoSink:
        code = cv2.VideoWriter.fourcc(*fourcc)
        return cv2.VideoWriter(path, code, fps, size)

    @property
    def out_path(self) -> Path:
        """The video file frames are written to."""

        return self._path

    @property
    def manifest_path(self) -> Path:
        """The sidecar carrying each frame's true clock (video stem, ``.json``)."""

        return self._path.with_suffix(".json")

    def frames(self) -> Iterator[Frame]:
        sink: VideoSink | None = None
        stamps: list[FrameStamp] = []
        source_name: str | None = None
        try:
            for frame in self._source.frames():
                if sink is None:
                    height, width = frame.image.shape[:2]
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    candidate = self._new_sink(
                        str(self._path), self._fourcc, self._fps, (width, height)
                    )
                    if not candidate.isOpened():
                        candidate.release()
                        raise OSError(
                            f"cannot open video writer for {self._path} "
                            f"(codec {self._fourcc!r} unavailable?)"
                        )
                    sink = candidate  # marks a genuinely-open recording (manifest gets written)
                    source_name = frame.meta.source
                sink.write(frame.image)
                stamps.append(
                    FrameStamp(frame_id=frame.meta.frame_id, timestamp=frame.meta.timestamp)
                )
                yield frame
        finally:
            if sink is not None:
                sink.release()
                self._write_manifest(stamps, source_name)

    def _write_manifest(self, stamps: list[FrameStamp], source_name: str | None) -> None:
        """Persist the temporal sidecar for whatever frames were actually recorded.

        Runs in ``frames``' ``finally`` so an interrupted or bounded capture still
        indexes the frames it did write; only reached once the video opened, so a
        failed-to-open recording leaves no orphan manifest.
        """

        manifest = RecordingManifest(
            video=self._path.name,
            source=source_name if source_name is not None else "unknown",
            fps=self._fps,
            stamps=tuple(stamps),
        )
        manifest.save(self.manifest_path)
