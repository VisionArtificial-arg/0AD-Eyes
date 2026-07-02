"""Command-line entry point (``zero-ad-eyes``).

Trunk scope: a ``run`` command that drives the pipeline with the stub model over a
synthetic in-memory source, emitting world models as JSON. This proves the seam
end-to-end headlessly. Feature agents extend it with real sources (``--recording``),
live capture, and the overlay window.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from zero_ad_eyes.application.pipeline import PerceptionPipeline
from zero_ad_eyes.application.ports import FrameSource, PerceptionModel, StageProfiler
from zero_ad_eyes.infrastructure.model.stub_adapter import StubPerceptionModel


def _offline_source(recording: str) -> FrameSource:
    """A recording is an image folder or a video file — pick the matching source."""

    from zero_ad_eyes.infrastructure.acquisition import ImageFolderSource, VideoFileSource

    path = Path(recording)
    return ImageFolderSource(path) if path.is_dir() else VideoFileSource(path)


def _build_offline_pipeline(
    recording: str, *, detector: str = "stub", profiler: StageProfiler | None = None
) -> PerceptionPipeline:
    """Wire the real classical chain over a recording (EPIC A→G integration).

    Everything except the detection *model* is a real classical adapter: offline
    source, preprocessing, HUD calibration + reader, minimap reader, IoU tracker,
    and the classical entity enricher (ownership/health/state). The model seam stays
    plugged with the stub (🔌) by default so the learned detector remains the only
    thing to swap in at MP4; ``--detector classical`` opts into the classical
    detection baseline (E6a) meanwhile.
    """

    from zero_ad_eyes.infrastructure.calibration import HudCalibrator
    from zero_ad_eyes.infrastructure.hud.reader import ClassicalHudReader
    from zero_ad_eyes.infrastructure.minimap.reader import ClassicalMinimapReader
    from zero_ad_eyes.infrastructure.perception import (
        ClassicalEntityEnricher,
        ClassicalPerceptionModel,
    )
    from zero_ad_eyes.infrastructure.preprocessing.pipeline import PreprocessingPipeline
    from zero_ad_eyes.infrastructure.tracking import ClassicalEventDetector, IouTracker

    model: PerceptionModel = (
        ClassicalPerceptionModel() if detector == "classical" else StubPerceptionModel()
    )
    return PerceptionPipeline(
        _offline_source(recording),
        model,
        preprocessor=PreprocessingPipeline(),
        calibrator=HudCalibrator(),
        hud_reader=ClassicalHudReader(),
        minimap_reader=ClassicalMinimapReader(),
        tracker=IouTracker(),
        enricher=ClassicalEntityEnricher(),
        event_detector=ClassicalEventDetector(),
        profiler=profiler,
    )


def _synthetic_source(n_frames: int, width: int, height: int) -> object:
    import numpy as np

    from zero_ad_eyes.application.frames import Frame
    from zero_ad_eyes.domain.world_model import FrameMeta
    from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource

    frames = [
        Frame(
            image=np.zeros((height, width, 3), dtype=np.uint8),
            meta=FrameMeta(
                frame_id=i,
                timestamp=float(i),
                source="synthetic",
                width=width,
                height=height,
            ),
        )
        for i in range(n_frames)
    ]
    return InMemoryFrameSource(frames)


def _format_metric(metric: object) -> str:
    from zero_ad_eyes.infrastructure.data import MetricResult

    assert isinstance(metric, MetricResult)
    if metric.is_pending:
        return f"  {metric.name}: pending-model (needs the trained model, MP4)"
    verdict = {True: "PASS", False: "FAIL", None: "n/a"}[metric.passed]
    bound = "<=" if not metric.higher_is_better else ">="
    threshold = "" if metric.threshold is None else f" (target {bound} {metric.threshold})"
    return f"  {metric.name}: {metric.value:.4f} [{verdict}]{threshold}"


def _run_eval(dataset: str | None) -> int:
    """ML8 accuracy gate. Honest by construction: with no ground-truth dataset it
    reports the classical metrics as unmeasured and detection mAP as pending-model,
    rather than scoring empty inputs. With a dataset it runs the real harness and
    exits non-zero only on a measured failure."""

    from zero_ad_eyes.infrastructure.data import evaluate

    if not dataset or not Path(dataset).exists():
        print("eval: no ground-truth dataset supplied (pass --dataset PATH once recordings exist)")
        print("  detection_map: pending-model (needs the trained model, MP4)")
        print("  hud_read_error / ownership_accuracy / tracking_mota: unmeasured (no ground truth)")
        return 0

    raw = json.loads(Path(dataset).read_text(encoding="utf-8"))
    from zero_ad_eyes.domain.world_model import WorldModel

    predicted = [WorldModel.model_validate(item) for item in raw.get("predicted", [])]
    truth = [WorldModel.model_validate(item) for item in raw.get("truth", [])]
    report = evaluate(predicted, truth)
    for metric in report.metrics:
        print(_format_metric(metric))
    verdict = report.passed
    print(f"eval: {'PENDING' if verdict is None else ('PASS' if verdict else 'FAIL')}")
    return 1 if verdict is False else 0


def _run_bench(
    *,
    recording: str | None,
    detector: str,
    frames: int,
    width: int,
    height: int,
    warmup: int,
) -> int:
    """T6 perf harness. Honest like eval: on the stub/classical path the latency is
    provisional (the learned inference that dominates NF1 is absent), so it reports
    but never gates. A real measured failure (model in the loop) exits non-zero."""

    from zero_ad_eyes.infrastructure.perf import StageTimings, benchmark

    timings = StageTimings()
    if recording is not None:
        pipeline = _build_offline_pipeline(recording, detector=detector, profiler=timings)
    else:
        source = _synthetic_source(frames, width, height)
        pipeline = PerceptionPipeline(source, StubPerceptionModel(), profiler=timings)  # type: ignore[arg-type]

    report = benchmark(pipeline, warmup=warmup, model_available=False)
    latency = report.latency
    print(f"bench: {report.frames} frames measured (warmup {report.warmup})")
    print(
        f"  latency ms  p50={latency.p50_ms:.2f}  p95={latency.p95_ms:.2f}  "
        f"max={latency.max_ms:.2f}  (target <= {report.latency_target_ms:.0f})"
    )
    print(
        f"  throughput  {report.throughput_fps:.1f} fps  "
        f"(target >= {report.throughput_target_fps:.0f})"
    )
    per_stage = timings.stats()
    if per_stage:
        print("  per-stage (NF6):")
        for stage, stats in per_stage.items():
            print(f"    {stage:10s} p95={stats.p95_ms:6.2f}  mean={stats.mean_ms:6.2f}  ms")
    if report.is_provisional:
        print("  verdict: PROVISIONAL (stub/classical path; real NF1 gate closes at MP4)")
        return 0
    passed = report.meets_latency and report.meets_throughput
    print(f"  verdict: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zero-ad-eyes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline (synthetic source, stub model)")
    run.add_argument("--frames", type=int, default=3)
    run.add_argument("--width", type=int, default=1280)
    run.add_argument("--height", type=int, default=720)
    run.add_argument(
        "--recording",
        default=None,
        help="path to a recording (image folder or video); drives the real classical chain",
    )
    run.add_argument(
        "--detector",
        choices=("stub", "classical"),
        default="stub",
        help="detection source for --recording; stub keeps the model seam plugged (default)",
    )

    ev = sub.add_parser("eval", help="run the ML8 accuracy harness (NF3 metrics)")
    ev.add_argument("--dataset", default=None, help="JSON with {predicted, truth} world models")

    bench = sub.add_parser("bench", help="benchmark latency + throughput (T6/NF1/NF2)")
    bench.add_argument("--frames", type=int, default=100)
    bench.add_argument("--width", type=int, default=1280)
    bench.add_argument("--height", type=int, default=720)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument(
        "--recording", default=None, help="benchmark the real classical chain over a recording"
    )
    bench.add_argument("--detector", choices=("stub", "classical"), default="stub")

    args = parser.parse_args(argv)

    if args.command == "run":
        if args.recording is not None:
            pipeline = _build_offline_pipeline(args.recording, detector=args.detector)
        else:
            source = _synthetic_source(args.frames, args.width, args.height)
            pipeline = PerceptionPipeline(source, StubPerceptionModel())  # type: ignore[arg-type]
        for world_model in pipeline.run():
            print(world_model.model_dump_json())
        return 0

    if args.command == "eval":
        return _run_eval(args.dataset)

    if args.command == "bench":
        return _run_bench(
            recording=args.recording,
            detector=args.detector,
            frames=args.frames,
            width=args.width,
            height=args.height,
            warmup=args.warmup,
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
