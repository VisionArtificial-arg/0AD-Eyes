"""Pipeline orchestration — wires the ports into the runtime data flow.

    acquire -> preprocess -> (calibrate -> hud, minimap) + perceive -> track -> world model -> sink

The pipeline holds only ``Protocol`` references, so it runs end-to-end today with
the stub ``PerceptionModel`` and grows real behaviour as each feature adapter is
merged in — with no change here (REQUIREMENTS.md §5.10, §2 D-decisions).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.world_model import WorldModel

from .frames import Frame
from .ports import (
    Calibrator,
    EntityEnricher,
    EntityFuser,
    EventSource,
    FrameSource,
    HudReader,
    LayoutChecker,
    MinimapReader,
    PerceptionModel,
    Preprocessor,
    StageProfiler,
    Tracker,
    WorldModelSink,
)


def _same_resolution(frame: Frame, calibration: Calibration) -> bool:
    """Whether ``frame``'s resolution matches the one a calibration was built for.

    Prefers the actual pixel-buffer shape, falling back to declared metadata — the
    same rule the calibrator uses to read resolution, so reuse and re-detection agree.
    """

    shape = getattr(frame.image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]) == calibration.width and int(shape[0]) == calibration.height
    return frame.meta.width == calibration.width and frame.meta.height == calibration.height


class PerceptionPipeline:
    """Composes the ports; every collaborator except source+model is optional."""

    def __init__(
        self,
        source: FrameSource,
        model: PerceptionModel,
        *,
        preprocessor: Preprocessor | None = None,
        calibrator: Calibrator | None = None,
        self_check: LayoutChecker | None = None,
        recalibrate_interval: int,
        hud_reader: HudReader | None = None,
        minimap_reader: MinimapReader | None = None,
        tracker: Tracker | None = None,
        enricher: EntityEnricher | None = None,
        fuser: EntityFuser | None = None,
        event_detector: EventSource | None = None,
        sink: WorldModelSink | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        self._source = source
        self._model = model
        self._preprocessor = preprocessor
        self._calibrator = calibrator
        self._self_check = self_check
        self._recalibrate_interval = recalibrate_interval
        self._hud_reader = hud_reader
        self._minimap_reader = minimap_reader
        self._tracker = tracker
        self._enricher = enricher
        self._fuser = fuser
        self._event_detector = event_detector
        self._sink = sink
        self._profiler = profiler

    def _stage(self, name: str) -> AbstractContextManager[None]:
        """Demarcate a timed stage (NF6). No profiler → a zero-cost null context."""

        return self._profiler.measure(name) if self._profiler is not None else nullcontext()

    def run(self) -> Iterator[WorldModel]:
        """Process every frame from the source into a world model."""

        calibration: Calibration | None = None
        frames_since_check = 0
        for raw in self._source.frames():
            if self._preprocessor is not None:
                with self._stage("preprocess"):
                    frame = self._preprocessor.process(raw)
            else:
                frame = raw
            calibration, frames_since_check = self._resolve_calibration(
                frame, calibration, frames_since_check
            )
            world_model = self._perceive(frame, calibration)
            if self._sink is not None:
                with self._stage("sink"):
                    self._sink.publish(world_model)
            yield world_model

    def _resolve_calibration(
        self, frame: Frame, previous: Calibration | None, frames_since_check: int
    ) -> tuple[Calibration | None, int]:
        """Reuse (B3) vs re-detect, with a periodic layout self-check (B4).

        Calibration is session-stable and keyed by resolution (A4), so the prior
        profile is reused in-memory — HUD calibration is the dominant classical-path
        cost. A fresh detection happens only when: there is none yet, the resolution
        changed, or (every ``recalibrate_interval`` frames) the self-check reports the
        live layout has drifted. The self-check re-runs anchor detection, so it is
        amortised over the interval rather than paid per frame. Returns the calibration
        to use and the updated frames-since-check counter.
        """

        if self._calibrator is None:
            return previous, frames_since_check
        if previous is None or not _same_resolution(frame, previous):
            return self._detect(frame), 0

        if self._self_check is None or frames_since_check + 1 < self._recalibrate_interval:
            return previous, frames_since_check + 1

        with self._stage("selfcheck"):
            verdict = self._self_check.verify(frame, previous)
        if verdict.matches:
            return previous, 0  # layout confirmed; restart the interval
        return self._detect(frame), 0

    def _detect(self, frame: Frame) -> Calibration:
        assert self._calibrator is not None  # only reached when a calibrator is present
        with self._stage("calibrate"):
            return self._calibrator.calibrate(frame)

    def _perceive(self, frame: Frame, calibration: Calibration | None) -> WorldModel:
        hud = None
        minimap = None
        if calibration is not None:
            if self._hud_reader is not None:
                with self._stage("hud"):
                    hud = self._hud_reader.read(frame, calibration)
            if self._minimap_reader is not None:
                with self._stage("minimap"):
                    minimap = self._minimap_reader.read(frame, calibration)

        with self._stage("infer"):
            detections = self._model.infer(frame)
        if self._tracker is not None:
            with self._stage("track"):
                entities = self._tracker.update(detections, frame)
        else:
            entities = ()
        if self._enricher is not None:
            with self._stage("enrich"):
                entities = self._enricher.enrich(entities, frame)

        if self._fuser is not None and minimap is not None:
            with self._stage("fuse"):
                entities = self._fuser.fuse(entities, minimap)

        if self._event_detector is not None:
            with self._stage("events"):
                events = self._event_detector.detect(entities, frame.meta.frame_id)
        else:
            events = ()

        return WorldModel(
            meta=frame.meta,
            hud=hud,
            minimap=minimap,
            entities=entities,
            events=events,
        )
