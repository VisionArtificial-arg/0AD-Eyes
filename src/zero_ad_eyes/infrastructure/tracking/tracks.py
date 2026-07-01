"""The mutable ``Track`` — a single object's evolving state (REQUIREMENTS.md G1/G2).

A ``Track`` is the tracker's private, stateful collaborator (Alan-Kay style: the
behaviour that mutates a track lives *on* the track, not in the tracker). It carries
the persistent ``track_id``, the latest box/kind/confidence, and the accumulated
trajectory of centroids. Domain ``Entity`` objects are *projected out* of a track
via :meth:`to_entity`; the track itself never leaves this package.

Lifecycle (G2): a track is born ``TENTATIVE``, promoted to ``CONFIRMED`` once it has
been seen enough (``min_hits``), turns ``LOST`` when a frame misses it (enter fog /
leave viewport are modelled identically as a disappearance), and finally ``DEAD``
when its memory budget is exhausted. G3 adds the staleness increment + confidence
decay that governs how long a ``LOST`` track is remembered.
"""

from __future__ import annotations

from enum import StrEnum

from zero_ad_eyes.domain.confidence import Confidence
from zero_ad_eyes.domain.detections import Detection
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox, ScreenPoint
from zero_ad_eyes.domain.taxonomy import EntityKind


class TrackStatus(StrEnum):
    """Where a track sits in its birth→death lifecycle (G2)."""

    TENTATIVE = "tentative"  # just born, not yet confirmed by repeated hits
    CONFIRMED = "confirmed"  # seen enough to be trusted
    LOST = "lost"  # missed this frame; retained in memory (G3) until dead
    DEAD = "dead"  # memory exhausted; scheduled for removal


class Track:
    """One tracked object across frames. Owns its own lifecycle behaviour."""

    def __init__(self, track_id: int, detection: Detection, frame_id: int) -> None:
        self.track_id = track_id
        self.kind: EntityKind = detection.kind
        self.entity_type: str | None = detection.entity_type
        self.bbox: ScreenBBox = detection.bbox
        self.confidence: Confidence = detection.confidence
        self.staleness: int = 0
        self.hits: int = 1
        self.age: int = 1
        self.last_seen_frame: int = frame_id
        self.status: TrackStatus = TrackStatus.TENTATIVE
        self.trajectory: list[ScreenPoint] = [detection.bbox.center]

    @property
    def centroid(self) -> ScreenPoint:
        return self.bbox.center

    def register_hit(self, detection: Detection, frame_id: int, *, min_hits: int) -> None:
        """Absorb a matched detection: refresh state, promote, extend trajectory."""

        self.kind = detection.kind
        self.entity_type = detection.entity_type
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.staleness = 0
        self.hits += 1
        self.age += 1
        self.last_seen_frame = frame_id
        self.trajectory.append(detection.bbox.center)
        if self.hits >= min_hits:
            self.status = TrackStatus.CONFIRMED

    def mark_missed(self, *, max_staleness: int, decay: float) -> None:
        """No detection matched this frame: age the track and decay its belief (G3).

        Staleness counts frames since the last real observation; confidence is
        multiplied by ``decay`` so a remembered fact grows less trustworthy the
        longer it goes unseen. Once staleness exceeds the memory budget the track
        is declared ``DEAD``.
        """

        self.staleness += 1
        self.age += 1
        self.confidence = Confidence(
            value=self.confidence.value * decay,
            provenance=self.confidence.provenance,
        )
        self.status = TrackStatus.DEAD if self.staleness > max_staleness else TrackStatus.LOST

    @property
    def is_dead(self) -> bool:
        return self.status is TrackStatus.DEAD

    def to_entity(self) -> Entity:
        """Project the current track state into an immutable domain ``Entity``."""

        return Entity(
            entity_id=self.track_id,
            kind=self.kind,
            entity_type=self.entity_type,
            screen_bbox=self.bbox,
            staleness=self.staleness,
            confidence=self.confidence,
        )
