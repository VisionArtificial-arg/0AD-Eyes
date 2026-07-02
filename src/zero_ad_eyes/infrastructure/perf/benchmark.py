"""Performance benchmark harness (REQUIREMENTS.md T6 / NF1 / NF2).

Measures the pipeline's **end-to-end** per-frame latency and sustained throughput,
so the NF1 (≤ ~66 ms) and NF2 (15–30 FPS) targets can be checked with numbers rather
than assumed. It treats the pipeline as a black box: it times each world model the
pipeline yields and aggregates the durations.

Honest by construction, like the ML8 eval harness: on the **stub** model the latency
is *not representative of live NF1* (the learned inference that will dominate the
budget is absent), so a report built without ``model_available`` is flagged
**provisional** — the real NF1 gate closes once the model adapter (MP4) lands and the
benchmark runs on a controlled machine over real frames. Throughput of the classical
path is meaningful now and catches gross regressions.

The clock is injected (defaults to ``time.perf_counter``) so the aggregation is unit
-testable with deterministic timestamps — the wall clock never enters a test's asserts.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.application.pipeline import PerceptionPipeline

# NF1/NF2 targets (OQ-2, Aggressive). Overridable per run.
DEFAULT_LATENCY_TARGET_MS = 66.0
DEFAULT_THROUGHPUT_TARGET_FPS = 15.0


def _percentile(ordered_ms: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty sequence."""

    rank = math.ceil(pct / 100.0 * len(ordered_ms))
    rank = min(max(rank, 1), len(ordered_ms))
    return ordered_ms[rank - 1]


class LatencyStats(BaseModel):
    """Distribution of per-frame latencies, in milliseconds."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=0)
    min_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0


def latency_stats(durations_ms: Sequence[float]) -> LatencyStats:
    """Aggregate raw per-frame durations into a :class:`LatencyStats`."""

    if not durations_ms:
        return LatencyStats(count=0)
    ordered = sorted(durations_ms)
    return LatencyStats(
        count=len(ordered),
        min_ms=ordered[0],
        mean_ms=sum(ordered) / len(ordered),
        p50_ms=_percentile(ordered, 50.0),
        p95_ms=_percentile(ordered, 95.0),
        max_ms=ordered[-1],
    )


class BenchmarkReport(BaseModel):
    """The measured perf scorecard (T6): latency distribution + throughput vs targets."""

    model_config = ConfigDict(frozen=True)

    frames: int = Field(ge=0)  # frames measured (warm-up excluded)
    warmup: int = Field(ge=0)
    wall_seconds: float = Field(ge=0.0)
    throughput_fps: float = Field(ge=0.0)
    latency: LatencyStats
    latency_target_ms: float = DEFAULT_LATENCY_TARGET_MS
    throughput_target_fps: float = DEFAULT_THROUGHPUT_TARGET_FPS
    model_available: bool = False

    @property
    def meets_latency(self) -> bool:
        """p95 within the NF1 budget."""

        return self.latency.count > 0 and self.latency.p95_ms <= self.latency_target_ms

    @property
    def meets_throughput(self) -> bool:
        """Sustained rate at/above the NF2 floor."""

        return self.throughput_fps >= self.throughput_target_fps

    @property
    def is_provisional(self) -> bool:
        """Latency is only representative of live NF1 once the real model is in the loop."""

        return not self.model_available


def benchmark(
    pipeline: PerceptionPipeline,
    *,
    warmup: int = 1,
    max_frames: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
    latency_target_ms: float = DEFAULT_LATENCY_TARGET_MS,
    throughput_target_fps: float = DEFAULT_THROUGHPUT_TARGET_FPS,
    model_available: bool = False,
) -> BenchmarkReport:
    """Run ``pipeline`` over its source, timing each yielded world model.

    The first ``warmup`` frames are timed but excluded from the stats (they pay
    one-off costs: lazy imports, first calibration, JIT-warm caches). Throughput is
    the measured frames over the summed processing time (back-to-back), so it reports
    *processing* capacity independent of inter-frame idle. ``max_frames`` caps the
    measured window; ``None`` consumes the whole source.
    """

    iterator = pipeline.run()
    durations_ms: list[float] = []
    index = 0
    while max_frames is None or index < warmup + max_frames:
        start = clock()
        try:
            next(iterator)
        except StopIteration:
            break
        elapsed_ms = (clock() - start) * 1000.0
        if index >= warmup:
            durations_ms.append(elapsed_ms)
        index += 1

    wall_seconds = sum(durations_ms) / 1000.0
    throughput = (len(durations_ms) / wall_seconds) if wall_seconds > 0.0 else 0.0
    return BenchmarkReport(
        frames=len(durations_ms),
        warmup=warmup,
        wall_seconds=wall_seconds,
        throughput_fps=throughput,
        latency=latency_stats(durations_ms),
        latency_target_ms=latency_target_ms,
        throughput_target_fps=throughput_target_fps,
        model_available=model_available,
    )
