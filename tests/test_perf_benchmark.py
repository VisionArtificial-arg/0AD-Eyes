"""T6 — perf benchmark harness: latency percentiles + throughput, deterministic."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.pipeline import PerceptionPipeline
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource
from zero_ad_eyes.infrastructure.model import StubPerceptionModel
from zero_ad_eyes.infrastructure.perf import benchmark, latency_stats


def _fake_clock(values: Sequence[float]) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)


def _pipeline(n_frames: int) -> PerceptionPipeline:
    frames = [
        Frame(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            meta=FrameMeta(frame_id=i, timestamp=float(i), source="test", width=4, height=4),
        )
        for i in range(n_frames)
    ]
    return PerceptionPipeline(
        InMemoryFrameSource(frames), StubPerceptionModel(), recalibrate_interval=30
    )


def test_latency_stats_math() -> None:
    stats = latency_stats([20.0, 60.0, 40.0])  # unsorted on purpose
    assert stats.count == 3
    assert stats.min_ms == 20.0
    assert stats.max_ms == 60.0
    assert stats.mean_ms == 40.0
    assert stats.p50_ms == 40.0  # nearest-rank
    assert stats.p95_ms == 60.0


def test_latency_stats_empty_is_zeroed() -> None:
    stats = latency_stats([])
    assert stats.count == 0
    assert stats.p95_ms == 0.0


def test_benchmark_aggregates_with_injected_clock() -> None:
    # warm-up frame is timed but excluded; measured durations are 20, 40, 60 ms.
    clock = _fake_clock([0.0, 0.010, 0.010, 0.030, 0.030, 0.070, 0.070, 0.130])
    report = benchmark(_pipeline(4), warmup=1, max_frames=3, clock=clock)

    assert report.frames == 3
    assert report.latency.p50_ms == pytest.approx(40.0)  # nearest-rank of [20, 40, 60]
    assert report.latency.p95_ms == pytest.approx(60.0)
    assert report.throughput_fps == pytest.approx(25.0)  # 3 frames / 0.120 s
    assert report.meets_latency  # p95 60 <= 66
    assert report.meets_throughput  # 25 >= 15
    assert report.is_provisional  # stub model → latency not representative of live NF1


def test_provisional_flag_clears_when_model_available() -> None:
    clock = _fake_clock([0.0, 0.010, 0.010, 0.030, 0.030, 0.070, 0.070, 0.130])
    report = benchmark(_pipeline(4), warmup=1, max_frames=3, clock=clock, model_available=True)
    assert not report.is_provisional


def test_benchmark_runs_on_wall_clock() -> None:
    report = benchmark(_pipeline(3), warmup=1)  # whole source, real clock
    assert report.frames == 2  # 3 minus the warm-up frame
    assert report.latency.count == 2
    assert report.throughput_fps >= 0.0
    assert report.latency.min_ms <= report.latency.p50_ms <= report.latency.max_ms


def test_stage_timings_measure_with_injected_clock() -> None:
    from zero_ad_eyes.infrastructure.perf import StageTimings

    clock = _fake_clock([1.0, 1.020, 5.0, 5.010])  # two measures: 20 ms then 10 ms
    timings = StageTimings(clock=clock)
    with timings.measure("infer"):
        pass
    with timings.measure("infer"):
        pass

    stats = timings.stats()
    assert stats["infer"].count == 2
    assert stats["infer"].min_ms == pytest.approx(10.0)
    assert stats["infer"].max_ms == pytest.approx(20.0)


def test_pipeline_records_per_stage_timings() -> None:
    from zero_ad_eyes.infrastructure.perf import StageTimings
    from zero_ad_eyes.infrastructure.tracking import IouTracker

    frames = [
        Frame(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            meta=FrameMeta(frame_id=i, timestamp=float(i), source="test", width=4, height=4),
        )
        for i in range(3)
    ]
    timings = StageTimings()
    pipeline = PerceptionPipeline(
        InMemoryFrameSource(frames),
        StubPerceptionModel(),
        recalibrate_interval=30,
        tracker=IouTracker(iou_threshold=0.3, min_hits=1, max_staleness=15, decay=0.85),
        profiler=timings,
    )

    list(pipeline.run())

    stats = timings.stats()
    assert stats["infer"].count == 3  # infer runs every frame
    assert stats["track"].count == 3  # tracker present → track stage runs
    assert stats["infer"].min_ms >= 0.0


def test_no_profiler_is_zero_overhead_and_unchanged() -> None:
    # Without a profiler the pipeline still yields the same number of world models.
    assert len(list(_pipeline(3).run())) == 3
