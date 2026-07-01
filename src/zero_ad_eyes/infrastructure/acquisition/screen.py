"""Live screen/window capture via ``mss`` (EPIC A / A1).

``ScreenCaptureSource`` grabs a monitor or an arbitrary region at a target FPS and
emits ``Frame`` objects tagged ``source="live"``. The actual pixel grab is behind
the tiny ``Grabber`` seam, so the source is unit-testable with a fake grabber and
never needs a display; the default ``MssGrabber`` is the only place that touches
``mss`` and it does so lazily (import + handle created on first grab).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.world_model import FrameMeta

from .timing import Clock, FramePacer, Sleep

RawImage = NDArray[Any]


@dataclass(frozen=True)
class CaptureRegion:
    """A rectangular capture area in screen pixels (``mss`` ``top/left`` origin)."""

    top: int
    left: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return {"top": self.top, "left": self.left, "width": self.width, "height": self.height}


class Grabber(Protocol):
    """Returns one raw screenshot (BGRA or BGR ``ndarray``) per call."""

    def grab(self) -> RawImage: ...


class MssGrabber:
    """Default ``Grabber``: captures a monitor or region with ``mss`` (lazy handle)."""

    def __init__(self, monitor: int = 1, region: CaptureRegion | None = None) -> None:
        self._monitor = monitor
        self._region = region
        self._sct: Any | None = None

    def _ensure_open(self) -> Any:
        if self._sct is None:
            import mss

            self._sct = mss.MSS()
        return self._sct

    def grab(self) -> RawImage:
        sct = self._ensure_open()
        area = self._region.as_mss() if self._region is not None else sct.monitors[self._monitor]
        return np.asarray(sct.grab(area))

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


def _to_bgr(raw: RawImage) -> RawImage:
    """Normalise a raw grab to 3-channel BGR (``mss`` yields BGRA)."""

    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr


class ScreenCaptureSource:
    """A ``FrameSource`` that captures the screen live at a target FPS."""

    def __init__(
        self,
        *,
        monitor: int = 1,
        region: CaptureRegion | None = None,
        target_fps: float = 30.0,
        grabber: Grabber | None = None,
        max_frames: int | None = None,
        source: str = "live",
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._grabber: Grabber = grabber if grabber is not None else MssGrabber(monitor, region)
        self._pacer = FramePacer(target_fps, clock=clock, sleep=sleep)
        self._max_frames = max_frames
        self._source = source

    @property
    def dropped_total(self) -> int:
        """Frame intervals missed so far (see :class:`FramePacer`)."""

        return self._pacer.dropped_total

    def frames(self) -> Iterator[Frame]:
        emitted = 0
        for tick in self._pacer.ticks():
            if self._max_frames is not None and emitted >= self._max_frames:
                return
            image = _to_bgr(self._grabber.grab())
            height, width = image.shape[:2]
            yield Frame(
                image=image,
                meta=FrameMeta(
                    frame_id=tick.frame_id,
                    timestamp=tick.timestamp,
                    source=self._source,
                    width=width,
                    height=height,
                ),
            )
            emitted += 1
