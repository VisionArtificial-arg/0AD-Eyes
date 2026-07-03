"""Portal + PipeWire capture backend (EPIC A1) — Discord-style window/screen capture.

``PortalPipeWireGrabber`` is a :class:`~.screen.Grabber` for Wayland sessions that,
unlike the grim backend (visible output only), can capture a **specific window**
(e.g. 0 A.D.) even when it is not the focused/visible workspace — because the
compositor does the capture and streams it over PipeWire.

The heavy stack (D-Bus portal negotiation + GStreamer ``pipewiresrc``) lives in a
standalone helper run under the **system** Python (see
``portal_pipewire_helper.py``); this grabber owns that helper as a long-lived
subprocess and reads its length-prefixed BGR frame stream. A background thread keeps
only the latest frame, so ``grab()`` returns the most recent frame at the pacer's
rate without the stream backing up.

One consent picker appears when the helper's portal session starts (once per capture
run, not per frame); on portal backends that support restore tokens (ScreenCast v4+)
that can be skipped on later runs.
"""

from __future__ import annotations

import struct
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO, Protocol

import numpy as np

from zero_ad_eyes.application.settings import AcquisitionSettings

from .screen import CaptureRegion, RawImage

_HELPER = Path(__file__).with_name("portal_pipewire_helper.py")
_HEADER = struct.Struct("<III")  # width, height, stride (little-endian)


def _read_exact(stream: IO[bytes], n: int) -> bytes | None:
    """Read exactly ``n`` bytes, or ``None`` if the stream ends first."""

    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_frame(stream: IO[bytes]) -> RawImage | None:
    """Read one length-prefixed BGR frame from the helper; de-pad row stride.

    Returns ``None`` at end-of-stream. The helper may emit rows padded to a stride
    wider than ``width*3``; we slice each row back to the packed width.
    """

    header = _read_exact(stream, _HEADER.size)
    if header is None:
        return None
    width, height, stride = _HEADER.unpack(header)
    payload = _read_exact(stream, height * stride)
    if payload is None:
        return None
    rows = np.frombuffer(payload, dtype=np.uint8).reshape(height, stride)
    return np.ascontiguousarray(rows[:, : width * 3]).reshape(height, width, 3)


class _Closable(Protocol):
    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = ...) -> int: ...


# Starts the capture helper and returns (process-handle, its stdout byte stream).
StreamFactory = Callable[[], tuple[_Closable, IO[bytes]]]


class PortalPipeWireGrabber:
    """A ``Grabber`` that streams a window/screen via xdg-desktop-portal + PipeWire."""

    def __init__(
        self,
        *,
        source_type: str = "window",
        cursor: str = "embedded",
        helper_python: str = "/usr/bin/python",
        restore_token_file: str | None = None,
        region: CaptureRegion | None = None,
        startup_timeout: float = 60.0,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self._source_type = source_type
        self._cursor = cursor
        self._helper_python = helper_python
        self._restore_token_file = restore_token_file
        self._region = region
        self._startup_timeout = startup_timeout
        self._new_stream = stream_factory if stream_factory is not None else self._spawn_helper

        self._proc: _Closable | None = None
        self._stream: IO[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: RawImage | None = None
        self._first = threading.Event()
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: AcquisitionSettings,
        region: CaptureRegion | None = None,
        *,
        stream_factory: StreamFactory | None = None,
    ) -> PortalPipeWireGrabber:
        token = settings.portal_restore_token_file or None
        return cls(
            source_type=settings.portal_source_type,
            cursor=settings.portal_cursor,
            helper_python=settings.portal_helper_python,
            restore_token_file=token,
            region=region,
            stream_factory=stream_factory,
        )

    def _spawn_helper(self) -> tuple[_Closable, IO[bytes]]:
        cmd = [
            self._helper_python,
            str(_HELPER),
            "--source-type",
            self._source_type,
            "--cursor",
            self._cursor,
        ]
        if self._restore_token_file:
            cmd += ["--restore-token-file", self._restore_token_file]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert proc.stdout is not None
        return proc, proc.stdout

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        self._proc, self._stream = self._new_stream()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        stream = self._stream
        assert stream is not None
        while not self._closed:
            frame = read_frame(stream)
            if frame is None:
                break
            with self._lock:
                self._latest = frame
            self._first.set()

    def grab(self) -> RawImage:
        self._ensure_started()
        if not self._first.wait(timeout=self._startup_timeout):
            raise OSError(
                "portal capture produced no frame within "
                f"{self._startup_timeout:.0f}s (consent dialog not approved?)"
            )
        with self._lock:
            assert self._latest is not None
            image = self._latest
        if self._region is not None:
            r = self._region
            image = image[r.top : r.top + r.height, r.left : r.left + r.width]
        return image

    def close(self) -> None:
        self._closed = True
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
