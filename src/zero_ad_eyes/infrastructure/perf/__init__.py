"""Performance measurement (REQUIREMENTS.md T6 / NF1 / NF2 / NF6).

The benchmark harness times the pipeline end-to-end and reports latency percentiles
+ throughput against the NF1/NF2 targets. Offline dev/CI tooling: it never sits in
the inference path. On the stub model the latency figure is provisional (see
:mod:`.benchmark`) — the real NF1 gate closes at MP4 on a controlled machine.
"""

from __future__ import annotations

from .benchmark import (
    DEFAULT_LATENCY_TARGET_MS,
    DEFAULT_THROUGHPUT_TARGET_FPS,
    BenchmarkReport,
    LatencyStats,
    benchmark,
    latency_stats,
)

__all__ = [
    "DEFAULT_LATENCY_TARGET_MS",
    "DEFAULT_THROUGHPUT_TARGET_FPS",
    "BenchmarkReport",
    "LatencyStats",
    "benchmark",
    "latency_stats",
]
