"""Pipeline orchestration — wires the ports into the runtime data flow.

    acquire -> preprocess -> (calibrate -> hud, minimap) + perceive -> track -> world model -> sink

The pipeline holds only ``Protocol`` references, so it runs end-to-end today with
the stub ``PerceptionModel`` and grows real behaviour as each feature adapter is
merged in — with no change here (REQUIREMENTS.md §5.10, §2 D-decisions).
"""

from __future__ import annotations

from collections.abc import Iterator

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.world_model import WorldModel

from .frames import Frame
from .ports import (
    Calibrator,
    EntityEnricher,
    FrameSource,
    HudReader,
    MinimapReader,
    PerceptionModel,
    Preprocessor,
    Tracker,
    WorldModelSink,
)


class PerceptionPipeline:
    """Composes the ports; every collaborator except source+model is optional."""

    def __init__(
        self,
        source: FrameSource,
        model: PerceptionModel,
        *,
        preprocessor: Preprocessor | None = None,
        calibrator: Calibrator | None = None,
        hud_reader: HudReader | None = None,
        minimap_reader: MinimapReader | None = None,
        tracker: Tracker | None = None,
        enricher: EntityEnricher | None = None,
        sink: WorldModelSink | None = None,
    ) -> None:
        self._source = source
        self._model = model
        self._preprocessor = preprocessor
        self._calibrator = calibrator
        self._hud_reader = hud_reader
        self._minimap_reader = minimap_reader
        self._tracker = tracker
        self._enricher = enricher
        self._sink = sink

    def run(self) -> Iterator[WorldModel]:
        """Process every frame from the source into a world model."""

        calibration: Calibration | None = None
        for raw in self._source.frames():
            frame = self._preprocessor.process(raw) if self._preprocessor else raw
            calibration = self._calibrate(frame, calibration)
            world_model = self._perceive(frame, calibration)
            if self._sink is not None:
                self._sink.publish(world_model)
            yield world_model

    def _calibrate(self, frame: Frame, previous: Calibration | None) -> Calibration | None:
        if self._calibrator is None:
            return previous
        return self._calibrator.calibrate(frame)

    def _perceive(self, frame: Frame, calibration: Calibration | None) -> WorldModel:
        hud = None
        minimap = None
        if calibration is not None:
            if self._hud_reader is not None:
                hud = self._hud_reader.read(frame, calibration)
            if self._minimap_reader is not None:
                minimap = self._minimap_reader.read(frame, calibration)

        detections = self._model.infer(frame)
        entities = self._tracker.update(detections, frame) if self._tracker else ()
        if self._enricher is not None:
            entities = self._enricher.enrich(entities, frame)

        return WorldModel(
            meta=frame.meta,
            hud=hud,
            minimap=minimap,
            entities=entities,
        )
