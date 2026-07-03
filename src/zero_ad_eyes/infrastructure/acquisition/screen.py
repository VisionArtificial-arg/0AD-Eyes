"""Live screen/window capture via ``mss`` (EPIC A / A1).

``ScreenCaptureSource`` grabs a monitor or an arbitrary region at a target FPS and
emits ``Frame`` objects tagged ``source="live"``. The actual pixel grab is behind
the tiny ``Grabber`` seam, so the source is unit-testable with a fake grabber and
never needs a display; the default ``MssGrabber`` is the only place that touches
``mss`` and it does so lazily (import + handle created on first grab).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import AcquisitionSettings
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

    def __init__(self, monitor: int, region: CaptureRegion | None = None) -> None:
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


CommandRunner = Callable[[Sequence[str]], bytes]


class WaylandGrabber:
    """``Grabber`` for Wayland sessions, via a screenshot CLI that writes one encoded
    image to stdout (e.g. ``grim -`` on wlroots/Hyprland).

    X11 ``GetImage`` (what ``mss`` uses) is blocked under Wayland/XWayland, so live
    capture there goes through the compositor's own screenshot tool over the Wayland
    protocol. The command is config-driven (``acquisition.wayland_capture_command``) so
    it adapts per compositor; the process runner is injected so the decode/crop path is
    unit-testable without spawning anything or needing a display.
    """

    def __init__(
        self,
        command: Sequence[str],
        region: CaptureRegion | None = None,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        if not command:
            raise ValueError("wayland capture command must not be empty")
        self._command = tuple(command)
        self._region = region
        self._runner = runner if runner is not None else self._default_runner

    def grab(self) -> RawImage:
        encoded = self._runner(self._command)
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot decode image from capture command {self._command!r}")
        if self._region is not None:
            r = self._region
            image = image[r.top : r.top + r.height, r.left : r.left + r.width]
        return image

    @staticmethod
    def _default_runner(command: Sequence[str]) -> bytes:
        import subprocess

        result = subprocess.run(tuple(command), capture_output=True, check=True)
        return result.stdout


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
        monitor: int,
        target_fps: float,
        region: CaptureRegion | None = None,
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

    @classmethod
    def from_settings(
        cls,
        settings: AcquisitionSettings,
        *,
        region: CaptureRegion | None = None,
        grabber: Grabber | None = None,
        max_frames: int | None = None,
    ) -> ScreenCaptureSource:
        """Build the live source from the ``acquisition`` config (Approach B).

        The tuning values (which monitor, target FPS) come from config; ``region`` and
        the run-control / test seams (``grabber``, ``max_frames``) are supplied by the
        composition root, since they are not tuning defaults. The grabber is chosen by
        ``capture_backend``: the default ``mss`` (X11) is built lazily in ``__init__``;
        ``wayland`` builds a :class:`WaylandGrabber` from ``wayland_capture_command``;
        ``portal`` builds a ``PortalPipeWireGrabber`` (window/screen capture via
        xdg-desktop-portal + PipeWire). An explicitly injected ``grabber`` overrides
        the backend selection (tests).
        """

        if grabber is None and settings.capture_backend == "wayland":
            grabber = WaylandGrabber(settings.wayland_capture_command, region)
        elif grabber is None and settings.capture_backend == "portal":
            from .portal import PortalPipeWireGrabber

            grabber = PortalPipeWireGrabber.from_settings(settings, region)
        return cls(
            monitor=settings.live_monitor,
            target_fps=settings.live_fps,
            region=region,
            grabber=grabber,
            max_frames=max_frames,
        )

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
