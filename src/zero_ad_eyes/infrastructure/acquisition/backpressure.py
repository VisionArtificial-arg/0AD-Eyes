"""Bounded buffering and backpressure shared across sources (EPIC A / A4).

Live capture is a producer/consumer problem: the grabber runs at the target FPS
while the pipeline consumes at its own (possibly slower) rate. ``BoundedFrameBuffer``
bounds memory by dropping the *oldest* frame when full — the honest backpressure
policy for real-time perception, where a stale frame is worthless.

``ThreadedFrameSource`` is a decorator that wraps *any* ``FrameSource``: it drains
the wrapped source on a background thread into a bounded buffer, so a slow consumer
never blocks the producer. It works with the live source or the offline readers
identically, keeping backpressure a shared, source-agnostic concern.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import FrameSource

_CLOSED = object()


class BoundedFrameBuffer:
    """Thread-safe, bounded frame queue with a drop-oldest overflow policy."""

    def __init__(self, maxsize: int = 8) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._items: deque[Frame] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """How many frames were discarded because the buffer was full."""

        return self._dropped

    def put(self, frame: Frame) -> None:
        """Append a frame, dropping the oldest if the buffer is already full."""

        with self._not_empty:
            if len(self._items) >= self._maxsize:
                self._items.popleft()
                self._dropped += 1
            self._items.append(frame)
            self._not_empty.notify()

    def get(self, timeout: float | None = None) -> Frame | object:
        """Pop the oldest frame; return the closed sentinel when drained and closed."""

        with self._not_empty:
            while not self._items:
                if self._closed:
                    return _CLOSED
                if not self._not_empty.wait(timeout=timeout):
                    if not self._items and self._closed:
                        return _CLOSED
            return self._items.popleft()

    def close(self) -> None:
        """Signal that no more frames will arrive; wakes any blocked consumer."""

        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()


class ThreadedFrameSource:
    """Backpressure decorator: run any ``FrameSource`` on a background thread.

    The wrapped source is drained into a ``BoundedFrameBuffer``; this object's
    ``frames()`` yields from that buffer, so a slow consumer causes stale frames to
    be dropped rather than blocking the producer.
    """

    def __init__(self, source: FrameSource, *, maxsize: int = 8) -> None:
        self._source = source
        self._maxsize = maxsize
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Frames dropped by backpressure during the most recent ``frames()`` run."""

        return self._dropped

    def frames(self) -> Iterator[Frame]:
        buffer = BoundedFrameBuffer(self._maxsize)

        def _produce() -> None:
            try:
                for frame in self._source.frames():
                    buffer.put(frame)
            finally:
                buffer.close()

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()
        try:
            while True:
                item = buffer.get(timeout=0.1)
                if item is _CLOSED:
                    break
                assert isinstance(item, Frame)
                yield item
        finally:
            producer.join()
            self._dropped = buffer.dropped
