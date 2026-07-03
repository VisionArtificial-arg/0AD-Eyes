"""Command-line entry point (``zero-ad-eyes``).

Trunk scope: a ``run`` command that drives the pipeline with the stub model over a
synthetic in-memory source, emitting world models as JSON. This proves the seam
end-to-end headlessly. Feature agents extend it with real sources (``--recording``),
live capture, and the overlay window.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.pipeline import PerceptionPipeline
from zero_ad_eyes.application.ports import (
    FrameSource,
    PerceptionModel,
    StageProfiler,
    WorldModelSink,
)
from zero_ad_eyes.application.settings import AcquisitionSettings, Config
from zero_ad_eyes.domain.world_model import WorldModel
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
    file carries its own frame clock. A video written by ``--record`` has a sibling
    ``.json`` sidecar (A5): when present we replay through :class:`RecordedVideoSource`
    so frames regain the capture's true ``frame_id``/``timestamp`` — which is what lets
    ``eval --recording`` align them to an engine ground-truth export (#2/D6)."""

    from zero_ad_eyes.infrastructure.acquisition import (
        ImageFolderSource,
        RecordedVideoSource,
        VideoFileSource,
    )

    path = Path(recording)
    if path.is_dir():
        return ImageFolderSource(
            path, extensions=acquisition.image_extensions, fps=acquisition.offline_fps
        )
    if path.with_suffix(".json").exists():
        return RecordedVideoSource(path)
    return VideoFileSource(path)


def _perception_model(detector: str, config: Config) -> PerceptionModel:
    """The main-viewport detector behind the model seam.

    ``stub`` is the default until the model-team adapter lands: it emits no
    main-scene detections, while HUD/minimap/calibration still run classically.
    ``classical`` remains an explicit debug baseline only.
    """

    if detector == "classical":
        from zero_ad_eyes.infrastructure.perception import ClassicalPerceptionModel

        return ClassicalPerceptionModel.from_settings(config.perception)
    if detector == "learned":
        from zero_ad_eyes.infrastructure.model import SegmentationPerceptionModel

        return SegmentationPerceptionModel.from_weights()
    return StubPerceptionModel()


class _LatestFrameSource:
    """FrameSource decorator that exposes the most recent frame for overlay output."""

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self.latest: Frame | None = None

    def frames(self) -> Iterator[Frame]:
        for frame in self._source.frames():
            self.latest = frame
            yield frame


class _OverlaySink:
    """World-model sink that annotates the latest frame seen by ``_LatestFrameSource``."""

    def __init__(self, frames: _LatestFrameSource, output: Any) -> None:
        self._frames = frames
        self._output = output

    def publish(self, world_model: WorldModel) -> None:
        frame = self._frames.latest
        if frame is not None:
            self._output.publish(frame, world_model)


def _build_pipeline(
    source: FrameSource,
    *,
    detector: str = "stub",
    profiler: StageProfiler | None = None,
    config: Config | None = None,
    sink: WorldModelSink | None = None,
) -> PerceptionPipeline:
    """Wire the HUD/minimap classical chain over any ``FrameSource``.

    Classical CV owns preprocessing, HUD calibration + self-check, HUD/minimap
    readers, post-detection enrichment, tracking, minimap fusion (G4/G5), and
    events. Main-viewport detection is deliberately behind ``PerceptionModel``:
    the default stub emits no scene detections until the learned adapter lands.
    ``--detector classical`` is kept only as a noisy debug baseline.

    This is the composition root where config is read (NF7): each adapter is built
    ``from_settings`` of the typed :class:`Config` (the generated defaults when none
    supplied), so the pipeline itself stays config-free — it only ever sees ports.
    """

    from zero_ad_eyes.infrastructure.calibration import (
        CalibrationProfileStore,
        HudCalibrator,
        LayoutSelfCheck,
    )
    from zero_ad_eyes.infrastructure.geometry import ViewportCameraProjector
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

    # Manual calibration profiles are always readable from cfg.paths.calibration_dir.
    # Automatic writes still require cfg.calibration.persist_profiles, enforced by
    # HudCalibrator.from_settings, so a normal run does not create surprise files.
    profile_store = CalibrationProfileStore(cfg.paths.calibration_dir)

    return PerceptionPipeline(
        source,
        _perception_model(detector, cfg),
        preprocessor=PreprocessingPipeline(),
        calibrator=HudCalibrator.from_settings(cfg.calibration, store=profile_store),
        self_check=LayoutSelfCheck.from_settings(cfg.calibration),
        hud_reader=ClassicalHudReader.from_settings(cfg.hud),
        minimap_reader=ClassicalMinimapReader.from_settings(cfg.minimap),
        recalibrate_interval=cfg.pipeline.recalibrate_interval,
        tracker=IouTracker.from_settings(cfg.tracking),
        enricher=ClassicalEntityEnricher.from_settings(cfg.perception),
        projector=ViewportCameraProjector.from_settings(cfg.geometry),
        fuser=ClassicalEntityFuser.from_settings(cfg.geometry),
        event_detector=ClassicalEventDetector.from_settings(cfg.tracking),
        sink=sink,
        profiler=profiler,
    )


def _build_offline_pipeline(
    recording: str,
    *,
    detector: str = "stub",
    profiler: StageProfiler | None = None,
    config: Config | None = None,
    sink: WorldModelSink | None = None,
) -> PerceptionPipeline:
    """The HUD/minimap classical chain over a recording (image folder or video)."""

    cfg = config if config is not None else default_config()
    return _build_pipeline(
        _offline_source(recording, cfg.acquisition),
        detector=detector,
        profiler=profiler,
        config=cfg,
        sink=sink,
    )


def _build_live_pipeline(
    *,
    detector: str = "stub",
    profiler: StageProfiler | None = None,
    config: Config | None = None,
    max_frames: int | None = None,
    record_path: Path | None = None,
    overlay_output: Any | None = None,
    sink: WorldModelSink | None = None,
) -> PerceptionPipeline:
    """The HUD/minimap classical chain over live screen capture (EPIC A1).

    ``cfg.acquisition`` supplies the monitor + target FPS; ``max_frames`` bounds an
    otherwise-endless capture (the ``run`` command maps ``--frames`` onto it) so the
    command terminates. Needs a display and the ``mss`` backend to actually grab.

    When ``record_path`` is given (``--record``, A5) the live source is wrapped in a
    :class:`VideoFrameRecorder` passthrough so every captured frame is persisted to a
    video file while the pipeline still consumes it — the dataset-collection hand-off
    for the real-frame validation chain (#1/#2).
    """

    from zero_ad_eyes.infrastructure.acquisition import ScreenCaptureSource, VideoFrameRecorder

    cfg = config if config is not None else default_config()
    source: FrameSource = ScreenCaptureSource.from_settings(cfg.acquisition, max_frames=max_frames)
    if record_path is not None:
        source = VideoFrameRecorder.from_settings(source, record_path, cfg.acquisition)
    output_sink = sink
    if overlay_output is not None:
        observed = _LatestFrameSource(source)
        source = observed
        overlay_sink = _OverlaySink(observed, overlay_output)
        if output_sink is None:
            output_sink = overlay_sink
        else:
            from zero_ad_eyes.infrastructure.contract import CompositeWorldModelSink

            output_sink = CompositeWorldModelSink(output_sink, overlay_sink)
    return _build_pipeline(
        source,
        detector=detector,
        profiler=profiler,
        config=cfg,
        sink=output_sink,
    )


def _recording_path(config: Config) -> Path:
    """A timestamped video file under ``cfg.paths.recordings_dir`` for ``--record``.

    The directory is config-driven (where captures are stored, per the readiness
    backlog); the container suffix comes from ``cfg.acquisition.record_container``.
    """

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.paths.recordings_dir / f"live_{stamp}{config.acquisition.record_container}"


def _first_live_frame(config: Config) -> Frame:
    """Capture one live frame for manual calibration."""

    from zero_ad_eyes.infrastructure.acquisition import ScreenCaptureSource

    return next(ScreenCaptureSource.from_settings(config.acquisition, max_frames=1).frames())


def _first_recorded_frame(recording: str, config: Config) -> Frame:
    """Read one recording frame for manual calibration."""

    return next(_offline_source(recording, config.acquisition).frames())


def _overlay_recording_path(record_path: Path, config: Config) -> Path:
    """Sibling path for an annotated video produced from the same live run."""

    return record_path.with_name(f"{record_path.stem}_overlay{config.acquisition.record_container}")


def _world_model_output_path(config: Config) -> Path:
    """Timestamped JSONL file for ``run`` output."""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.paths.recordings_dir / f"world_models_{stamp}.jsonl"


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
    if verdict is False:
        label = "FAIL"  # a measured metric fell short — surfaced even if mAP is pending
    elif verdict is True:
        label = "PASS"
    else:  # None: every measured metric passed; only model-dependent metric(s) pend
        pending = ", ".join(metric.name for metric in report.metrics if metric.is_pending)
        label = f"PASS (classical); pending-model: {pending}"
    print(f"eval: {label}")
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

    run = sub.add_parser("run", help="run the pipeline (model seam stub by default)")
    run.add_argument(
        "--frames",
        type=int,
        default=None,
        help="cap the number of frames; omit on --live to capture until Ctrl-C "
        "(the synthetic source falls back to 3 frames when omitted)",
    )
    run.add_argument("--width", type=int, default=1280)
    run.add_argument("--height", type=int, default=720)
    run.add_argument(
        "--recording",
        default=None,
        help="path to a recording (image folder or video); drives the classical HUD/minimap chain",
    )
    run.add_argument(
        "--live",
        action="store_true",
        help="capture the screen live (EPIC A1) from cfg.acquisition; needs a display + mss",
    )
    run.add_argument(
        "--record",
        action="store_true",
        help="persist the live capture to a video file in cfg.paths.recordings_dir "
        "(A5; codec cfg.acquisition.record_fourcc); requires --live",
    )
    run.add_argument(
        "--overlay",
        action="store_true",
        help="show the annotated debug overlay while running live capture",
    )
    run.add_argument(
        "--record-overlay",
        action="store_true",
        help="write an annotated overlay video next to the raw --record video",
    )
    run.add_argument(
        "--overlay-output",
        default=None,
        help="path for the annotated overlay video (implies --record-overlay)",
    )
    run.add_argument(
        "--output",
        default=None,
        help="JSONL world-model output path (default: recordings/world_models_<timestamp>.jsonl)",
    )
    run.add_argument(
        "--stdout",
        action="store_true",
        help="also write world models to stdout as JSON lines",
    )
    run.add_argument(
        "--detector",
        choices=("classical", "stub", "learned"),
        default="stub",
        help=(
            "main-viewport detector behind the model seam "
            "(default: stub; classical = debug baseline)"
        ),
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
        choices=("classical", "stub", "learned"),
        default="stub",
        help="main-viewport detector behind the model seam for --recording (default: stub)",
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
    bench.add_argument("--recording", default=None, help="benchmark the recording pipeline")
    bench.add_argument("--detector", choices=("classical", "stub", "learned"), default="stub")
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

    calibrate = sub.add_parser(
        "calibrate", help="manually draw HUD boxes and save a calibration profile"
    )
    calibrate.add_argument(
        "--recording",
        default=None,
        help="image folder or video to take the calibration frame from",
    )
    calibrate.add_argument(
        "--live",
        action="store_true",
        help="capture one live frame to calibrate from",
    )
    calibrate.add_argument(
        "--config", default=None, help="path to a JSON config file (NF7); env ZAE_* overrides it"
    )
    calibrate.add_argument(
        "--output-dir",
        default=None,
        help="directory for the saved profile (default: cfg.paths.calibration_dir)",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        if args.live and args.recording is not None:
            parser.error("choose one source: --live or --recording, not both")
        if args.record and not args.live:
            parser.error("--record requires --live (there is nothing to record otherwise)")
        if args.overlay and not args.live:
            parser.error("--overlay requires --live")
        if (args.record_overlay or args.overlay_output is not None) and not args.record:
            parser.error("--record-overlay/--overlay-output require --record")
        config = _load_config(args.config)
        overlay_output = None
        closers: list[Any] = []
        sinks: list[WorldModelSink] = []
        from zero_ad_eyes.infrastructure.contract import (
            CallbackWorldModelSink,
            CompositeWorldModelSink,
            JsonlFileWorldModelSink,
        )

        output_path = (
            Path(args.output) if args.output is not None else _world_model_output_path(config)
        )
        output_sink = JsonlFileWorldModelSink(output_path)
        closers.append(output_sink)
        sinks.append(output_sink)
        print(f"writing world models to {output_path}", file=sys.stderr)
        if args.stdout:
            sinks.append(CallbackWorldModelSink(lambda wm: print(wm.model_dump_json())))
        world_model_sink: WorldModelSink = (
            sinks[0] if len(sinks) == 1 else CompositeWorldModelSink(*sinks)
        )
        if args.live:
            record_path = _recording_path(config) if args.record else None
            if record_path is not None:
                print(f"recording live capture to {record_path}", file=sys.stderr)
            if args.overlay or args.record_overlay or args.overlay_output is not None:
                from zero_ad_eyes.interface.overlay import OverlayVideoSink

                overlay_path = None
                if args.record_overlay or args.overlay_output is not None:
                    assert record_path is not None
                    overlay_path = (
                        Path(args.overlay_output)
                        if args.overlay_output is not None
                        else _overlay_recording_path(record_path, config)
                    )
                    print(f"recording overlay video to {overlay_path}", file=sys.stderr)
                overlay_output = OverlayVideoSink(
                    settings=config.overlay,
                    video_path=overlay_path,
                    fps=config.acquisition.live_fps,
                    fourcc=config.acquisition.record_fourcc,
                    show=args.overlay,
                )
            pipeline = _build_live_pipeline(
                detector=args.detector,
                config=config,
                max_frames=args.frames,
                record_path=record_path,
                overlay_output=overlay_output,
                sink=world_model_sink,
            )
        elif args.recording is not None:
            pipeline = _build_offline_pipeline(
                args.recording,
                detector=args.detector,
                config=config,
                sink=world_model_sink,
            )
        else:
            n_frames = args.frames if args.frames is not None else 3
            source = _synthetic_source(n_frames, args.width, args.height)
            model = _perception_model(args.detector, config)
            pipeline = PerceptionPipeline(
                source,  # type: ignore[arg-type]
                model,
                recalibrate_interval=config.pipeline.recalibrate_interval,
                sink=world_model_sink,
            )
        try:
            for _world_model in pipeline.run():
                pass
        except KeyboardInterrupt:
            # Unbounded --live runs until interrupted; unwinding the pipeline
            # generator finalizes the recording (video + sidecar), then the
            # finally below flushes the overlay and JSONL sinks. Exit cleanly.
            print("stopping — finalizing capture…", file=sys.stderr)
        finally:
            if overlay_output is not None:
                overlay_output.close()
            for closer in closers:
                closer.close()
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

    if args.command == "calibrate":
        if args.live == (args.recording is not None):
            parser.error("choose exactly one calibration source: --live or --recording PATH")
        config = _load_config(args.config)
        if args.live:
            from zero_ad_eyes.infrastructure.acquisition import ScreenCaptureSource

            source = ScreenCaptureSource.from_settings(config.acquisition)
        else:
            source = _offline_source(args.recording, config.acquisition)
        from zero_ad_eyes.interface.manual_calibration import save_manual_calibration_from_source

        output_dir = (
            Path(args.output_dir) if args.output_dir is not None else config.paths.calibration_dir
        )
        path = save_manual_calibration_from_source(
            source,
            directory=output_dir,
            theme=config.calibration.theme,
        )
        print(f"calibrate: wrote manual profile to {path}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
