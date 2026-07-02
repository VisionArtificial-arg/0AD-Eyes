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
from zero_ad_eyes.application.settings import Config
from zero_ad_eyes.infrastructure.config.config import load_config
from zero_ad_eyes.infrastructure.model.stub_adapter import StubPerceptionModel


def _load_config(path: str | None) -> Config:
    """Load the typed config from ``path`` (defaults + JSON + env), or pure defaults."""

    return load_config(path)


def _offline_source(recording: str) -> FrameSource:
    """A recording is an image folder or a video file — pick the matching source."""

    from zero_ad_eyes.infrastructure.acquisition import ImageFolderSource, VideoFileSource

    path = Path(recording)
    return ImageFolderSource(path) if path.is_dir() else VideoFileSource(path)


def _perception_model(detector: str) -> PerceptionModel:
    """The detection model behind the seam. ``classical`` is the v1 default — the
    E6a/E11 baseline that actually perceives from pixels; ``stub`` emits nothing
    (plumbing only). The learned adapter (MP4) would slot in here unchanged."""

    if detector == "classical":
        from zero_ad_eyes.infrastructure.perception import ClassicalPerceptionModel

        return ClassicalPerceptionModel()
    return StubPerceptionModel()


def _build_offline_pipeline(
    recording: str,
    *,
    detector: str = "classical",
    profiler: StageProfiler | None = None,
    config: Config | None = None,
) -> PerceptionPipeline:
    """Wire the real classical chain over a recording (EPIC A→G integration).

    Every stage is a real classical adapter: offline source, preprocessing, HUD
    calibration + self-check, HUD/minimap readers, IoU tracker, entity enricher, and
    the classical detection baseline (E6a/E11) — the v1 default. ``--detector stub``
    swaps in the empty stub for plumbing-only runs; a learned adapter (MP4) would
    swap in here identically.

    This is the composition root where config is read (NF7): each adapter is built
    ``from_settings`` of the typed :class:`Config` (defaults when none supplied), so
    the pipeline itself stays config-free — it only ever sees ports.
    """

    from zero_ad_eyes.infrastructure.calibration import HudCalibrator, LayoutSelfCheck
    from zero_ad_eyes.infrastructure.hud.reader import ClassicalHudReader
    from zero_ad_eyes.infrastructure.minimap.reader import ClassicalMinimapReader
    from zero_ad_eyes.infrastructure.perception import ClassicalEntityEnricher
    from zero_ad_eyes.infrastructure.preprocessing.pipeline import PreprocessingPipeline
    from zero_ad_eyes.infrastructure.tracking import ClassicalEventDetector, IouTracker

    cfg = config or Config()

    return PerceptionPipeline(
        _offline_source(recording),
        _perception_model(detector),
        preprocessor=PreprocessingPipeline(),
        calibrator=HudCalibrator(),
        self_check=LayoutSelfCheck(),
        hud_reader=ClassicalHudReader.from_settings(cfg.hud),
        minimap_reader=ClassicalMinimapReader.from_settings(cfg.minimap),
        tracker=IouTracker.from_settings(cfg.tracking),
        enricher=ClassicalEntityEnricher.from_settings(cfg.perception),
        event_detector=ClassicalEventDetector.from_settings(cfg.tracking),
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


def _print_report(report: object) -> int:
    from zero_ad_eyes.infrastructure.data import EvaluationReport

    assert isinstance(report, EvaluationReport)
    for metric in report.metrics:
        print(_format_metric(metric))
    verdict = report.passed
    print(f"eval: {'PENDING' if verdict is None else ('PASS' if verdict else 'FAIL')}")
    return 1 if verdict is False else 0


def _eval_from_recording(
    recording: str,
    engine_export: str,
    *,
    detector: str,
    align_by: str,
    time_tolerance: float,
    config: Config,
) -> int:
    """Run the classical chain over a recording and score it against an engine export.

    This is the offline accuracy loop end-to-end: the pipeline produces predicted
    world models from pixels, the engine export (ML2/D6) provides ground truth, and
    the harness aligns and scores the classical NF3 metrics. Detection mAP stays
    pending-model until the learned adapter (MP4) lands.
    """

    from zero_ad_eyes.infrastructure.data import (
        AlignBy,
        EngineStateExport,
        EvalConfig,
        evaluate_against_engine,
    )

    pipeline = _build_offline_pipeline(recording, detector=detector, config=config)
    predicted = list(pipeline.run())
    export = EngineStateExport.load(Path(engine_export))
    report = evaluate_against_engine(
        predicted,
        export,
        align_by=AlignBy(align_by),
        time_tolerance=time_tolerance,
        config=EvalConfig.from_thresholds(config.thresholds),
    )
    return _print_report(report)


def _run_eval(
    *,
    dataset: str | None,
    recording: str | None,
    engine_export: str | None,
    detector: str,
    align_by: str,
    time_tolerance: float,
    config: Config,
) -> int:
    """ML8 accuracy gate. Honest by construction: with no ground truth it reports the
    classical metrics as unmeasured and detection mAP as pending-model, rather than
    scoring empty inputs. Two ground-truth sources are supported: a pre-serialized
    ``--dataset`` of ``{predicted, truth}`` world models, or a live ``--recording``
    scored against an ``--engine-export`` (the full offline loop). NF3 targets come
    from the loaded config (NF7). Either way it exits non-zero only on a *measured*
    failure."""

    from zero_ad_eyes.infrastructure.data import EvalConfig, evaluate

    if recording is not None or engine_export is not None:
        if recording is None or engine_export is None:
            print("eval: --recording and --engine-export must be supplied together")
            return 2
        return _eval_from_recording(
            recording,
            engine_export,
            detector=detector,
            align_by=align_by,
            time_tolerance=time_tolerance,
            config=config,
        )

    if not dataset or not Path(dataset).exists():
        print("eval: no ground-truth dataset supplied (pass --dataset PATH once recordings exist)")
        print("  or score a recording directly: --recording PATH --engine-export PATH")
        print("  detection_map: pending-model (needs the trained model, MP4)")
        print("  hud_read_error / ownership_accuracy / tracking_mota: unmeasured (no ground truth)")
        return 0

    raw = json.loads(Path(dataset).read_text(encoding="utf-8"))
    from zero_ad_eyes.domain.world_model import WorldModel

    predicted = [WorldModel.model_validate(item) for item in raw.get("predicted", [])]
    truth = [WorldModel.model_validate(item) for item in raw.get("truth", [])]
    report = evaluate(predicted, truth, config=EvalConfig.from_thresholds(config.thresholds))
    return _print_report(report)


def _run_bench(
    *,
    recording: str | None,
    detector: str,
    frames: int,
    width: int,
    height: int,
    warmup: int,
    config: Config,
) -> int:
    """T6 perf harness. Honest like eval: on the stub/classical path the latency is
    provisional (the learned inference that dominates NF1 is absent), so it reports
    but never gates. A real measured failure (model in the loop) exits non-zero."""

    from zero_ad_eyes.infrastructure.perf import StageTimings, benchmark

    timings = StageTimings()
    if recording is not None:
        pipeline = _build_offline_pipeline(
            recording, detector=detector, profiler=timings, config=config
        )
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

    run = sub.add_parser("run", help="run the pipeline (classical detection by default)")
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
        choices=("classical", "stub"),
        default="classical",
        help="detection model behind the seam (v1 default: classical; stub = plumbing only)",
    )
    run.add_argument(
        "--config", default=None, help="path to a JSON config file (NF7); env ZAE_* overrides it"
    )

    ev = sub.add_parser("eval", help="run the ML8 accuracy harness (NF3 metrics)")
    ev.add_argument("--dataset", default=None, help="JSON with {predicted, truth} world models")
    ev.add_argument(
        "--recording",
        default=None,
        help="score a recording (image folder or video) end-to-end; needs --engine-export",
    )
    ev.add_argument(
        "--engine-export",
        default=None,
        help="engine ground-truth JSON (ML2/D6) to score the --recording against",
    )
    ev.add_argument(
        "--detector",
        choices=("classical", "stub"),
        default="classical",
        help="detection model behind the seam for --recording (default: classical)",
    )
    ev.add_argument(
        "--align-by",
        choices=("frame_id", "timestamp"),
        default="frame_id",
        help="pair predicted frames to engine frames by exact id or nearest timestamp (R5)",
    )
    ev.add_argument(
        "--time-tolerance",
        type=float,
        default=0.0,
        help="max seconds between paired frames when --align-by timestamp",
    )
    ev.add_argument(
        "--config", default=None, help="path to a JSON config file (NF7); env ZAE_* overrides it"
    )

    bench = sub.add_parser("bench", help="benchmark latency + throughput (T6/NF1/NF2)")
    bench.add_argument("--frames", type=int, default=100)
    bench.add_argument("--width", type=int, default=1280)
    bench.add_argument("--height", type=int, default=720)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument(
        "--recording", default=None, help="benchmark the real classical chain over a recording"
    )
    bench.add_argument("--detector", choices=("classical", "stub"), default="classical")
    bench.add_argument(
        "--config", default=None, help="path to a JSON config file (NF7); env ZAE_* overrides it"
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        config = _load_config(args.config)
        if args.recording is not None:
            pipeline = _build_offline_pipeline(
                args.recording, detector=args.detector, config=config
            )
        else:
            source = _synthetic_source(args.frames, args.width, args.height)
            model = _perception_model(args.detector)
            pipeline = PerceptionPipeline(source, model)  # type: ignore[arg-type]
        for world_model in pipeline.run():
            print(world_model.model_dump_json())
        return 0

    if args.command == "eval":
        return _run_eval(
            dataset=args.dataset,
            recording=args.recording,
            engine_export=args.engine_export,
            detector=args.detector,
            align_by=args.align_by,
            time_tolerance=args.time_tolerance,
            config=_load_config(args.config),
        )

    if args.command == "bench":
        return _run_bench(
            recording=args.recording,
            detector=args.detector,
            frames=args.frames,
            width=args.width,
            height=args.height,
            warmup=args.warmup,
            config=_load_config(args.config),
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
