"""Per-stage timing accumulator (REQUIREMENTS.md NF6) — a ``StageProfiler`` adapter.

Wraps each pipeline stage the orchestrator demarcates, records its wall duration, and
aggregates the per-stage distributions into :class:`LatencyStats`. Offline/dev tooling
(paired with the benchmark harness); it never sits in the production inference path.

The clock is injected (defaults to ``time.perf_counter``) so the accumulation is
unit-testable without the wall clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .benchmark import LatencyStats, latency_stats


class StageTimings:
    """Accumulates per-stage durations; satisfies the ``StageProfiler`` port."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._durations_ms: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            elapsed_ms = (self._clock() - start) * 1000.0
            self._durations_ms.setdefault(stage, []).append(elapsed_ms)

    def stats(self) -> dict[str, LatencyStats]:
        """Per-stage latency distribution, in the order stages were first seen."""

        return {stage: latency_stats(durations) for stage, durations in self._durations_ms.items()}
