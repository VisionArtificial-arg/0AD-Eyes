"""Frame acquisition adapters (EPIC A).

``InMemoryFrameSource`` is a trunk-provided minimal ``FrameSource`` used by tests
and the CLI smoke path. The live (screen-capture) and offline (video/image-folder)
sources are built by the acquisition feature agent.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from zero_ad_eyes.application.frames import Frame

from .backpressure import BoundedFrameBuffer, ThreadedFrameSource
from .offline import ImageFolderSource, RecordedVideoSource, VideoFileSource
from .recorder import FrameRecorder
from .recording import FrameStamp, RecordingManifest
from .screen import CaptureRegion, Grabber, MssGrabber, ScreenCaptureSource
from .timing import FramePacer, Tick
from .video_recorder import VideoFrameRecorder, VideoSink, VideoSinkFactory


class InMemoryFrameSource:
    """Yields a fixed sequence of frames (satisfies the ``FrameSource`` port)."""

    def __init__(self, frames: Sequence[Frame]) -> None:
        self._frames = tuple(frames)

    def frames(self) -> Iterator[Frame]:
        yield from self._frames


__all__ = [
    "BoundedFrameBuffer",
    "CaptureRegion",
    "FramePacer",
    "FrameRecorder",
    "FrameStamp",
    "Grabber",
    "ImageFolderSource",
    "InMemoryFrameSource",
    "MssGrabber",
    "RecordedVideoSource",
    "RecordingManifest",
    "ScreenCaptureSource",
    "ThreadedFrameSource",
    "Tick",
    "VideoFileSource",
    "VideoFrameRecorder",
    "VideoSink",
    "VideoSinkFactory",
]
