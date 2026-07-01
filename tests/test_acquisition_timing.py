"""Tests for the shared pacer/timestamping/drop-detection (A4)."""

from __future__ import annotations

import pytest

from zero_ad_eyes.infrastructure.acquisition.timing import FramePacer


class ScriptedClock:
    """A deterministic clock returning a scripted sequence of monotonic times."""

    def __init__(self, times: list[float]) -> None:
        self._times = list(times)
        self._i = 0

    def __call__(self) -> float:
        value = self._times[self._i]
        self._i += 1
        return value


def test_unpaced_pacer_only_stamps_and_counts() -> None:
    clock = ScriptedClock([10.0, 10.5, 11.0])
    pacer = FramePacer(None, clock=clock, sleep=lambda _: None)

    ticks = [pacer.tick() for _ in range(3)]

    assert [t.frame_id for t in ticks] == [0, 1, 2]
    assert [t.timestamp for t in ticks] == [10.0, 10.5, 11.0]
    assert all(t.dropped == 0 for t in ticks)
    assert pacer.dropped_total == 0


def test_pacer_sleeps_to_hold_cadence_when_ahead() -> None:
    slept: list[float] = []
    # 10 FPS -> 0.1s interval. First frame at t=0 (deadline=0.1); second arrives
    # early at t=0.05 -> must sleep 0.05s.
    clock = ScriptedClock([0.0, 0.05])
    pacer = FramePacer(10.0, clock=clock, sleep=slept.append)

    first = pacer.tick()
    second = pacer.tick()

    assert first.dropped == 0 and second.dropped == 0
    assert slept == [pytest.approx(0.05)]
    assert second.timestamp == pytest.approx(0.1)  # snapped to the deadline


def test_pacer_detects_dropped_frames_when_behind() -> None:
    slept: list[float] = []
    # 10 FPS. First frame at t=0 (deadline=0.1). Second arrives late at t=0.35:
    # that is 0.25s past the deadline -> two whole intervals missed.
    clock = ScriptedClock([0.0, 0.35])
    pacer = FramePacer(10.0, clock=clock, sleep=slept.append)

    pacer.tick()
    second = pacer.tick()

    assert second.dropped == 2
    assert pacer.dropped_total == 2
    assert slept == []  # never sleeps when already behind


def test_pacer_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError):
        FramePacer(0.0)
