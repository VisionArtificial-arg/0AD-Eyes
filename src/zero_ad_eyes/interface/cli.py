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
from zero_ad_eyes.application.settings import AcquisitionSettings, Config
from zero_ad_eyes.infrastructure.config.config import load_config, save_config
from zero_ad_eyes.infrastructure.model.stub_adapter import StubPerceptionModel
from zero_ad_eyes.interface.default_config import default_config


def _load_config(path: str | None) -> Config:
    """Load the typed config: the generated defaults, then the file, then env (NF7).

    The models carry no defaults; :func:`default_config` is the single source of
    default values and is injected as the base the file/env layer onto. With no
    ``--config`` and no env overrides this is exactly the generated defaults, in
    memory — nothing is written to disk.
    """

    return load_config(default_config(), path)


def _offline_source(recording: str, acquisition: AcquisitionSettings) -> FrameSource:
    """A recording is an image folder or a video file — pick the matching source.

    Image-folder replay is config-paced (offline_fps + accepted extensions); a video
    file carries its own frame clock, so only the folder source takes config."""

    from zero_ad_eyes.infrastructure.acquisition import ImageFolderSource, VideoFileSource

    path = Path(recording)
    if path.is_dir():
        return ImageFolderSource(
            path, extensions=acquisition.image_extensions, fps=acquisition.offline_fps
        )
    return VideoFileSource(path)


def _perception_model(detector: str, config: Config) -> PerceptionModel:
    """The detection model behind the seam. ``classical`` is the v1 default — the
    E6a/E11 baseline that actually perceives from pixels, built from cfg.perception
    (resource cues + toggle); ``stub`` emits nothing (plumbing only). The learned
    adapter (MP4) would slot in here unchanged."""

    if detector == "classical":
        from zero_ad_eyes.infrastructure.perception import ClassicalPerceptionModel

        return ClassicalPerceptionModel.from_settings(config.perception)
    return StubPerceptionModel()


def _build_pipeline(
    source: FrameSource,
    *,
    detector: str = "classical",
    profiler: StageProfiler | None = None,
    config: Config | None = None,
) -> PerceptionPipeline:
    """Wire the real classical chain over any ``FrameSource`` (EPIC A→G integration).

    Every stage is a real classical adapter: preprocessing, HUD calibration +
    self-check, HUD/minimap readers, IoU tracker, entity enricher, minimap fusion
    (G4/G5), and the classical detection baseline (E6a/E11) — the v1 default.
    ``--detector stub`` swaps in the empty stub for plumbing-only runs; a learned
    adapter (MP4) would swap in here identically. The ``source`` (offline recording
    or live capture) is chosen by the caller.

    This is the composition root where config is read (NF7): each adapter is built
    ``from_settings`` of the typed :class:`Config` (the generated defaults when none
    supplied), so the pipeline itself stays config-free — it only ever sees ports.
    """

    from zero_ad_eyes.infrastructure.calibration import HudCalibrator, LayoutSelfCheck
    from zero_ad_eyes.infrastructure.hud.reader import ClassicalHudReader
    from zero_ad_eyes.infrastructure.minimap.reader import ClassicalMinimapReader
    from zero_ad_eyes.infrastructure.perception import ClassicalEntityEnricher
    from zero_ad_eyes.infrastructure.preprocessing.pipeline import PreprocessingPipeline
    from zero_ad_eyes.infrastructure.tracking import (
        ClassicalEntityFuser,
        ClassicalEventDetector,
        IouTracker,
    )

    cfg = config if config is not None else default_config()

    return PerceptionPipeline(
        source,
        _perception_model(detector, cfg),
        preprocessor=PreprocessingPipeline(),
        calibrator=HudCalibrator.from_settings(cfg.calibration),
        self_check=LayoutSelfCheck.from_settings(cfg.calibration),
        hud_reader=ClassicalHudReader.from_settings(cfg.hud),
        minimap_reader=ClassicalMinimapReader.from_settings(cfg.minimap),
        recalibrate_interval=cfg.pipeline.recalibrate_interval,
        tracker=IouTracker.from_settings(cfg.tracking),
        enricher=ClassicalEntityEnricher.from_settings(cfg.perception),
        fuser=ClassicalEntityFuser.from_settings(cfg.geometry),
        event_detector=ClassicalEventDetector.from_settings(cfg.tracking),
        profiler=profiler,
    )


def _build_offline_pipeline(
    recording: str,
    *,
    detector: str = "classical",
    profiler: StageProfiler | None = None,
    config: Config | None = None,
) -> PerceptionPipeline:
    """The classical chain over a recording (image folder or video)."""

    cfg = config if config is not None else default_config()
    return _build_pipeline(
        _offline_source(recording, cfg.acquisition),
        detector=detector,
        profiler=profiler,
        config=cfg,
    )


def _build_live_pipeline(
    *,
    detector: str = "classical",
    profiler: StageProfiler | None = None,
    config: Config | None = None,
    max_frames: int | None = None,
) -> PerceptionPipeline:
    """The classical chain over live screen capture (EPIC A1), built from config.

    ``cfg.acquisition`` supplies the monitor + target FPS; ``max_frames`` bounds an
    otherwise-endless capture (the ``run`` command maps ``--frames`` onto it) so the
    command terminates. Needs a display and the ``mss`` backend to actually grab.
    """

    from zero_ad_eyes.infrastructure.acquisition import ScreenCaptureSource

    cfg = config if config is not None else default_config()
    source = ScreenCaptureSource.from_settings(cfg.acquisition, max_frames=max_frames)
    return _build_pipeline(source, detector=detector, profiler=profiler, config=cfg)


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
        pipeline = PerceptionPipeline(
            source,  # type: ignore[arg-type]
            StubPerceptionModel(),
            recalibrate_interval=config.pipeline.recalibrate_interval,
            profiler=timings,
        )

    report = benchmark(
        pipeline,
        warmup=warmup,
        model_available=False,
        latency_target_ms=config.perf.latency_target_ms,
        throughput_target_fps=config.perf.throughput_target_fps,
    )
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


def _run_config(
    *,
    action: str,
    path: str | None,
    config: str | None,
    force: bool,
) -> int:
    """The ``config`` command group: surface the generated defaults as an editable file.

    The system holds no defaults except :func:`default_config`; these subcommands are
    the UI over it — ``init`` writes it, ``show`` prints the effective tree, ``validate``
    checks a file against the schema. Users discover and edit defaults as data, never
    by reading source (NF7).
    """

    from pydantic import ValidationError

    if action == "init":
        destination = Path(path) if path is not None else Path("config.json")
        if destination.exists() and not force:
            print(f"config: {destination} already exists (pass --force to overwrite)")
            return 1
        save_config(default_config(), destination)
        print(f"config: wrote default config to {destination}")
        return 0

    if action == "show":
        # Effective config: defaults, then the file (if any), then env overrides.
        print(_load_config(config).model_dump_json(indent=2))
        return 0

    if action == "validate":
        assert path is not None  # required by the argument parser
        source = Path(path)
        if not source.exists():
            print(f"config: {source} does not exist")
            return 2
        try:
            _load_config(str(source))
        except ValidationError as error:
            print(f"config: {source} is invalid")
            print(str(error))
            return 1
        except json.JSONDecodeError as error:
            print(f"config: {source} is not valid JSON: {error}")
            return 1
        print(f"config: {source} is valid")
        return 0

    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zero-ad-eyes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline (classical detection by default)")
    run.add_argument(
        "--frames",
        type=int,
        default=3,
        help="frames to emit (synthetic source; also bounds --live so it terminates)",
    )
    run.add_argument("--width", type=int, default=1280)
    run.add_argument("--height", type=int, default=720)
    run.add_argument(
        "--recording",
        default=None,
        help="path to a recording (image folder or video); drives the real classical chain",
    )
    run.add_argument(
        "--live",
        action="store_true",
        help="capture the screen live (EPIC A1) from cfg.acquisition; needs a display + mss",
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

    config_cmd = sub.add_parser(
        "config", help="generate / inspect / validate the config file (NF7)"
    )
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    cfg_init = config_sub.add_parser("init", help="write the default config to a file")
    cfg_init.add_argument(
        "path", nargs="?", default=None, help="destination file (default: config.json)"
    )
    cfg_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    cfg_show = config_sub.add_parser("show", help="print the effective config as JSON")
    cfg_show.add_argument(
        "--config",
        default=None,
        help="a JSON config to layer on the defaults (else the pure defaults); env ZAE_* applies",
    )
    cfg_validate = config_sub.add_parser(
        "validate", help="validate a config file against the schema"
    )
    cfg_validate.add_argument("path", help="the JSON config file to validate")

    args = parser.parse_args(argv)

    if args.command == "run":
        if args.live and args.recording is not None:
            parser.error("choose one source: --live or --recording, not both")
        config = _load_config(args.config)
        if args.live:
            pipeline = _build_live_pipeline(
                detector=args.detector, config=config, max_frames=args.frames
            )
        elif args.recording is not None:
            pipeline = _build_offline_pipeline(
                args.recording, detector=args.detector, config=config
            )
        else:
            source = _synthetic_source(args.frames, args.width, args.height)
            model = _perception_model(args.detector, config)
            pipeline = PerceptionPipeline(
                source,  # type: ignore[arg-type]
                model,
                recalibrate_interval=config.pipeline.recalibrate_interval,
            )
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

    if args.command == "config":
        return _run_config(
            action=args.config_command,
            path=getattr(args, "path", None),
            config=getattr(args, "config", None),
            force=getattr(args, "force", False),
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
