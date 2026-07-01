"""``IouTracker`` — the ``Tracker`` port adapter (REQUIREMENTS.md G1/G2, CV-15/16).

Multi-object tracking by greedy IoU association: each frame's ``Detections`` are
matched to the surviving tracks; matched tracks keep their id and extend their
trajectory, unmatched detections spawn fresh ids, and unmatched tracks are aged
through the lifecycle (G2) toward death. The adapter satisfies the ``Tracker``
protocol structurally (``update(detections, frame) -> tuple[Entity]``) and carries
detection confidence/provenance straight through onto the emitted entities.

G1 delivers stable ids + trajectories; G2 the birth→death lifecycle; G3 the memory
that keeps a recently-lost track alive (off-screen / in explored-but-not-visible
fog, §4.1) with growing staleness and decaying confidence until its budget runs
out. Motion (G6), stabilisation (G7) and events (G8) extend this same adapter.
"""

from __future__ import annotations

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.motion import Motion as EntityMotion

from .association import greedy_match
from .motion import Motion, motion_from_trajectory
from .tracks import Track, TrackStatus


class IouTracker:
    """Greedy-IoU multi-object tracker producing stable, lifecycle-aware entities."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        min_hits: int = 1,
        max_staleness: int = 15,
        decay: float = 0.85,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._min_hits = min_hits
        self._max_staleness = max_staleness
        self._decay = decay
        self._tracks: list[Track] = []
        self._next_id = 0

    def update(self, detections: Detections, frame: Frame) -> tuple[Entity, ...]:
        """Associate this frame's detections, age the misses, emit live entities."""

        items = detections.items
        assignment = greedy_match(
            [t.bbox for t in self._tracks],
            [d.bbox for d in items],
            iou_threshold=self._iou_threshold,
        )

        for track_idx, det_idx in assignment.matches:
            self._tracks[track_idx].register_hit(
                items[det_idx], detections.frame_id, min_hits=self._min_hits
            )
        for track_idx in assignment.unmatched_tracks:
            self._tracks[track_idx].mark_missed(
                max_staleness=self._max_staleness, decay=self._decay
            )

        survivors = [t for t in self._tracks if not t.is_dead]
        spawned = [
            self._spawn(items[det_idx], detections.frame_id)
            for det_idx in assignment.unmatched_detections
        ]
        self._tracks = survivors + spawned

        return tuple(self._emit(t) for t in self._tracks)

    def _emit(self, track: Track) -> Entity:
        """Build the entity and attach its v0.2 motion estimate (G6) when known.

        Motion is derived deterministically from the track's own trajectory, so it
        is a ``CLASSICAL`` fact and only as sure as the entity it belongs to. A track
        with fewer than two observations has no velocity yet — motion stays ``None``
        (honest "unknown"), never a fabricated "still".
        """

        entity = track.to_entity()
        if len(track.trajectory) < 2:
            return entity
        velocity = motion_from_trajectory(track.trajectory)
        motion = EntityMotion(
            dx=velocity.dx,
            dy=velocity.dy,
            confidence=Confidence(value=entity.confidence.value, provenance=Provenance.CLASSICAL),
        )
        return entity.model_copy(update={"motion": motion})

    def statuses(self) -> dict[int, TrackStatus]:
        """Observability (NF6): current lifecycle status keyed by entity id."""

        return {t.track_id: t.status for t in self._tracks}

    def motions(self) -> dict[int, Motion]:
        """Per-entity direction/speed from each track's trajectory (G6, CV-18)."""

        return {t.track_id: motion_from_trajectory(t.trajectory) for t in self._tracks}

    def _spawn(self, detection: Detection, frame_id: int) -> Track:
        track = Track(self._next_id, detection, frame_id)
        self._next_id += 1
        return track
