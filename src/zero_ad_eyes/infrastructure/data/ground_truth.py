"""ML2 — Ground-truth alignment scaffolding (REQUIREMENTS.md §7 ML2, D6).

0 A.D. is open source and can emit authoritative simulation state (replays / a
debug build / the AI-interface JSON). Decision **D6** says that engine-derived
state MAY be used as ground truth for building datasets and measuring accuracy —
and that it is **NEVER** available at inference (production reads pixels only,
D1). This module lives entirely on the offline side of that boundary.

WARNING — offline-only. Nothing here may be imported by the inference pipeline.
The types below model an *engine export*; linking them into live perception would
violate the pure-pixel boundary and fail the §11 Definition-of-Done audit.

Documented engine-state JSON schema
-----------------------------------
An export is one JSON document per match::

    {
      "match_id": "demo-001",
      "self_player": 1,           # engine player number we are playing
      "ally_players": [2],        # players treated as allies
      "frames": [
        {
          "frame_id": 0,
          "timestamp": 0.0,       # seconds, same clock as capture (R5)
          "phase": "village",
          "resources": {"food": 300, "wood": 200, "stone": 100, "metal": 50},
          "population_current": 8,
          "population_cap": 20,
          "entities": [
            {
              "entity_id": 42,
              "kind": "unit",
              "entity_type": "female_citizen",
              "owner": 1,                      # engine player number
              "health_current": 40.0,
              "health_max": 40.0,
              "world_x": 12.5, "world_y": 30.0,
              "bbox": {"x": 100, "y": 120, "width": 24, "height": 40}  # optional
            }
          ]
        }
      ]
    }

Alignment maps each engine frame to captured frames by **frame_id** (exact) or by
**timestamp** (nearest within a tolerance, R5), producing domain labels: a
``WorldModel`` (full state) and a ``Detections`` (only entities with a projected
screen ``bbox``). Ownership is derived from ``owner`` against ``self_player`` /
``ally_players``; player ``0`` is Gaia (neutral).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.hud import HudState, Population
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership, Phase, ResourceType
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel

GAIA_PLAYER = 0


class EngineEntityState(BaseModel):
    """One entity as the engine reports it (world space, engine player number)."""

    model_config = ConfigDict(frozen=True)

    entity_id: int = Field(ge=0)
    kind: EntityKind
    owner: int = Field(ge=0)
    entity_type: str | None = None
    health_current: float = Field(default=0.0, ge=0.0)
    health_max: float = Field(default=0.0, ge=0.0)
    world_x: float | None = None
    world_y: float | None = None
    bbox: ScreenBBox | None = None  # present only if the exporter projected it

    def health_fraction(self) -> float | None:
        if self.health_max <= 0.0:
            return None
        return min(1.0, self.health_current / self.health_max)

    def world_pos(self) -> WorldPoint | None:
        if self.world_x is None or self.world_y is None:
            return None
        return WorldPoint(x=self.world_x, y=self.world_y)


class EngineFrameState(BaseModel):
    """Authoritative simulation state for a single frame."""

    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp: float
    phase: Phase = Phase.UNKNOWN
    resources: dict[ResourceType, int] = Field(default_factory=dict)
    population_current: int | None = Field(default=None, ge=0)
    population_cap: int | None = Field(default=None, ge=0)
    entities: tuple[EngineEntityState, ...] = ()


class EngineStateExport(BaseModel):
    """A whole match's engine ground truth, as loaded from JSON."""

    model_config = ConfigDict(frozen=True)

    match_id: str
    self_player: int = Field(ge=0)
    ally_players: tuple[int, ...] = ()
    frames: tuple[EngineFrameState, ...] = ()

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EngineStateExport:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class AlignedGroundTruth(BaseModel):
    """Domain labels for one captured frame: full world model + detections."""

    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    world_model: WorldModel
    detections: Detections


class GroundTruthAligner:
    """Maps an engine export to domain labels, aligned to captured frames (D6).

    Ownership is resolved once, from the export's ``self_player`` / ``ally_players``
    roster, so every produced label agrees on who is self / ally / enemy / gaia.
    """

    def __init__(self, export: EngineStateExport) -> None:
        self._export = export
        self._by_frame_id = {frame.frame_id: frame for frame in export.frames}
        self._ordered = tuple(sorted(export.frames, key=lambda f: f.timestamp))

    def ownership_of(self, owner: int) -> Ownership:
        if owner == GAIA_PLAYER:
            return Ownership.GAIA
        if owner == self._export.self_player:
            return Ownership.SELF
        if owner in self._export.ally_players:
            return Ownership.ALLY
        return Ownership.ENEMY

    def frame_for_id(self, frame_id: int) -> EngineFrameState | None:
        return self._by_frame_id.get(frame_id)

    def frame_for_timestamp(self, timestamp: float, tolerance: float) -> EngineFrameState | None:
        """Nearest engine frame within ``tolerance`` seconds, or ``None`` (R5)."""

        best: EngineFrameState | None = None
        best_delta = tolerance
        for frame in self._ordered:
            delta = abs(frame.timestamp - timestamp)
            if delta <= best_delta:
                best_delta = delta
                best = frame
        return best

    def labels_for(self, engine_frame: EngineFrameState, meta: FrameMeta) -> AlignedGroundTruth:
        """Convert one aligned engine frame into domain labels under ``meta``."""

        return AlignedGroundTruth(
            frame_id=meta.frame_id,
            world_model=self._to_world_model(engine_frame, meta),
            detections=self._to_detections(engine_frame, meta.frame_id),
        )

    def align_by_frame_id(self, metas: Iterable[FrameMeta]) -> tuple[AlignedGroundTruth, ...]:
        """Align each captured-frame ``meta`` to its engine frame by exact id."""

        aligned: list[AlignedGroundTruth] = []
        for meta in metas:
            engine_frame = self.frame_for_id(meta.frame_id)
            if engine_frame is not None:
                aligned.append(self.labels_for(engine_frame, meta))
        return tuple(aligned)

    def align_by_timestamp(
        self, metas: Iterable[FrameMeta], tolerance: float
    ) -> tuple[AlignedGroundTruth, ...]:
        """Align each captured-frame ``meta`` to the nearest engine frame in time."""

        aligned: list[AlignedGroundTruth] = []
        for meta in metas:
            engine_frame = self.frame_for_timestamp(meta.timestamp, tolerance)
            if engine_frame is not None:
                aligned.append(self.labels_for(engine_frame, meta))
        return tuple(aligned)

    def _to_detections(self, engine_frame: EngineFrameState, frame_id: int) -> Detections:
        items = tuple(
            Detection(
                kind=entity.kind,
                bbox=entity.bbox,
                entity_type=entity.entity_type,
                confidence=Confidence.certain(Provenance.ENGINE_GT),
            )
            for entity in engine_frame.entities
            if entity.bbox is not None
        )
        return Detections(frame_id=frame_id, items=items)

    def _to_world_model(self, engine_frame: EngineFrameState, meta: FrameMeta) -> WorldModel:
        entities = tuple(
            Entity(
                entity_id=entity.entity_id,
                kind=entity.kind,
                ownership=self.ownership_of(entity.owner),
                entity_type=entity.entity_type,
                world_pos=entity.world_pos(),
                screen_bbox=entity.bbox,
                health=entity.health_fraction(),
                staleness=0,
                confidence=Confidence.certain(Provenance.ENGINE_GT),
            )
            for entity in engine_frame.entities
        )
        return WorldModel(meta=meta, hud=self._to_hud(engine_frame), entities=entities)

    def _to_hud(self, engine_frame: EngineFrameState) -> HudState:
        population: Population | None = None
        if engine_frame.population_current is not None and engine_frame.population_cap is not None:
            population = Population(
                current=engine_frame.population_current,
                cap=engine_frame.population_cap,
            )
        return HudState(
            stockpiles=dict(engine_frame.resources),
            population=population,
            phase=engine_frame.phase,
            confidence=Confidence.certain(Provenance.ENGINE_GT),
        )
