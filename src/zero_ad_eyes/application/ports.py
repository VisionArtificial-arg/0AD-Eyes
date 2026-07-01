"""Ports — the interfaces every feature agent implements (onion boundaries).

These Protocols are the contract between the pipeline and its adapters. They are
the *seam*: an adapter can be swapped (e.g. the stub ``PerceptionModel`` for the
real learned one, MP3→MP4) with zero change to the pipeline or to any other adapter.

Design rule: a port is expressed only in domain terms plus ``Frame``. No port
mentions OpenCV, a model framework, or a screen-capture library.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.detections import Detections
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.hud import HudState
from zero_ad_eyes.domain.minimap import MinimapModel
from zero_ad_eyes.domain.world_model import WorldModel

from .frames import Frame


@runtime_checkable
class FrameSource(Protocol):
    """EPIC A — yields frames from live capture or a recording, identically."""

    def frames(self) -> Iterator[Frame]: ...


@runtime_checkable
class Preprocessor(Protocol):
    """EPIC P — conditions a raw frame for downstream perception."""

    def process(self, frame: Frame) -> Frame: ...


@runtime_checkable
class Calibrator(Protocol):
    """EPIC B — locates HUD regions in the current frame."""

    def calibrate(self, frame: Frame) -> Calibration: ...


@runtime_checkable
class HudReader(Protocol):
    """EPIC C — parses the top bar / self-identification."""

    def read(self, frame: Frame, calibration: Calibration) -> HudState: ...


@runtime_checkable
class MinimapReader(Protocol):
    """EPIC D — interprets the minimap region."""

    def read(self, frame: Frame, calibration: Calibration) -> MinimapModel: ...


@runtime_checkable
class PerceptionModel(Protocol):
    """MP1 — the model seam. Stub and real adapters both satisfy this.

    Implementations MUST populate confidence and provenance on every detection
    so consumers cannot tell stub from real (REQUIREMENTS.md §5.10.2 / MP2).
    """

    def infer(self, frame: Frame, roi: ScreenBBox | None = None) -> Detections: ...


@runtime_checkable
class Tracker(Protocol):
    """EPIC G — turns per-frame detections into temporally-stable entities."""

    def update(self, detections: Detections, frame: Frame) -> tuple[Entity, ...]: ...


@runtime_checkable
class EntityEnricher(Protocol):
    """EPIC E — attaches per-entity attributes (ownership E3, health E4, state E5).

    Runs *after* tracking, reading each entity's crop from the frame to fill
    attributes the detector/tracker left absent. Fill-if-absent by contract: an
    enricher must not clobber a value a more authoritative source already set.
    """

    def enrich(self, entities: tuple[Entity, ...], frame: Frame) -> tuple[Entity, ...]: ...


@runtime_checkable
class WorldModelSink(Protocol):
    """EPIC H — receives each published world model (the decision-layer boundary)."""

    def publish(self, world_model: WorldModel) -> None: ...
