"""Config guards for perf targets (NF1/NF2) and the pipeline recalibrate interval."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from zero_ad_eyes.application.settings import PerfSettings
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.interface.cli import _build_offline_pipeline
from zero_ad_eyes.interface.default_config import default_config


def test_perf_and_pipeline_defaults_match_historical() -> None:
    # Guard the generated defaults against the frozen historical NF1/NF2 + interval.
    config = default_config()
    assert (config.perf.latency_target_ms, config.perf.throughput_target_fps) == (66.0, 15.0)
    assert config.pipeline.recalibrate_interval == 30


def test_config_file_threads_recalibrate_interval(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "frame_000.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('{"pipeline": {"recalibrate_interval": 7}}', encoding="utf-8")

    config = load_config(default_config(), cfg_path, env={})
    pipeline = _build_offline_pipeline(str(tmp_path), detector="stub", config=config)

    assert pipeline._recalibrate_interval == 7


def test_config_file_threads_perf_targets_into_benchmark() -> None:
    from zero_ad_eyes.application.frames import Frame
    from zero_ad_eyes.application.pipeline import PerceptionPipeline
    from zero_ad_eyes.domain.world_model import FrameMeta
    from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource
    from zero_ad_eyes.infrastructure.model.stub_adapter import StubPerceptionModel
    from zero_ad_eyes.infrastructure.perf import benchmark

    frames = [
        Frame(
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            meta=FrameMeta(frame_id=i, timestamp=float(i), source="syn", width=16, height=16),
        )
        for i in range(3)
    ]
    pipeline = PerceptionPipeline(
        InMemoryFrameSource(frames), StubPerceptionModel(), recalibrate_interval=30
    )

    perf = PerfSettings(latency_target_ms=10.0, throughput_target_fps=99.0)
    report = benchmark(
        pipeline,
        warmup=0,
        model_available=False,
        latency_target_ms=perf.latency_target_ms,
        throughput_target_fps=perf.throughput_target_fps,
    )

    assert report.latency_target_ms == 10.0
    assert report.throughput_target_fps == 99.0
