"""Shared timestamping, pacing and frame-drop detection (EPIC A / A4).

``FramePacer`` centralises three concerns every live source shares:

- *timestamping*: a monotonic capture time (seconds) per frame, from an injected
  clock so tests are deterministic;
- *pacing*: sleeping to hold a target FPS cadence;
- *frame-drop detection*: when the consumer falls behind the cadence, the pacer
  reports how many whole frame intervals were missed (telemetry, not a gap in the
  emitted ``frame_id`` — that stays a simple monotonic counter).

The clock and sleep are injectable so the whole thing is unit-testable without a
real wall clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

Clock = Callable[[], float]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class Tick:
    """One cadence step: the id/timestamp to stamp a frame with, plus drop telemetry."""

    frame_id: int
    timestamp: float
    dropped: int


class FramePacer:
    """Paces a capture loop to ``target_fps`` and stamps each frame.

    A ``target_fps`` of ``None`` disables pacing (run as fast as possible); the
    pacer then only supplies timestamps and a monotonic ``frame_id``.
    """

    def __init__(
        self,
        target_fps: float | None = None,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ) -> None:
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be positive or None")
        self._interval: float | None = (1.0 / target_fps) if target_fps else None
        self._clock = clock
        self._sleep = sleep
        self._frame_id = -1
        self._deadline: float | None = None
        self._dropped_total = 0

    @property
    def dropped_total(self) -> int:
        """Cumulative number of missed frame intervals detected so far."""

        return self._dropped_total

    def tick(self) -> Tick:
        """Advance one frame: pace, timestamp and report any drop."""

        now = self._clock()
        dropped = 0

        if self._interval is None:
            pass
        elif self._deadline is None:
            # First frame: establish the cadence, do not wait.
            self._deadline = now + self._interval
        elif now < self._deadline:
            self._sleep(self._deadline - now)
            now = self._deadline
            self._deadline += self._interval
        else:
            behind = now - self._deadline
            dropped = int(behind / self._interval)
            self._dropped_total += dropped
            self._deadline += self._interval * (dropped + 1)

        self._frame_id += 1
        return Tick(frame_id=self._frame_id, timestamp=now, dropped=dropped)

    def ticks(self) -> Iterator[Tick]:
        """An unbounded stream of ticks (one per frame)."""

        while True:
            yield self.tick()
