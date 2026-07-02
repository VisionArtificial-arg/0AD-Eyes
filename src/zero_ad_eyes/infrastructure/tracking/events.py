"""Event detection from temporal state changes (REQUIREMENTS.md G8, CV-34).

``EventDetector`` compares the entity set of the current frame against the previous
one (keyed by the tracker's stable ``entity_id``) and emits a list of discrete
``TrackingEvent`` values. It reads only the domain ``Entity`` contract, so whichever
upstream stage populated a field (health from E4, type from E2/E6) drives the event
identically — the detector never touches pixels.

Detected changes:

- **appearance / disappearance** — an id enters or leaves the emitted set (which,
  with G3 memory, means true birth and post-memory death).
- **combat** — an entity's health fraction drops by more than a threshold.
- **depletion** — a resource node's health reaches (near) zero.
- **state change** — the fine ``entity_type`` changes (e.g. a construction scaffold
  becoming its finished building: "build complete").
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.application.settings import TrackingSettings
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.events import Event
from zero_ad_eyes.domain.events import EventKind as DomainEventKind
from zero_ad_eyes.domain.taxonomy import EntityKind


class EventKind(StrEnum):
    """A discrete world change worth reporting to the decision layer."""

    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    COMBAT = "combat"  # health dropping (CV-34)
    DEPLETED = "depleted"  # resource node exhausted
    STATE_CHANGED = "state_changed"  # fine-type change, e.g. build-complete


class TrackingEvent(BaseModel):
    """One detected event, tied to the entity and frame it occurred on."""

    model_config = ConfigDict(frozen=True)

    kind: EventKind
    entity_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    detail: str | None = None


class EventDetector:
    """Stateful frame-over-frame differ producing an events list (G8)."""

    def __init__(self, *, combat_drop: float = 0.05, depletion_health: float = 0.02) -> None:
        self._combat_drop = combat_drop
        self._depletion_health = depletion_health
        self._previous: dict[int, Entity] = {}

    @classmethod
    def from_settings(cls, settings: TrackingSettings) -> EventDetector:
        """Build from pure config (Approach B boundary mapping)."""

        return cls(
            combat_drop=settings.combat_drop,
            depletion_health=settings.depletion_health,
        )

    def detect(self, entities: Sequence[Entity], frame_id: int) -> tuple[TrackingEvent, ...]:
        """Diff this frame's entities against the last, returning ordered events."""

        current = {e.entity_id: e for e in entities}
        events: list[TrackingEvent] = []

        for entity_id, entity in current.items():
            previous = self._previous.get(entity_id)
            if previous is None:
                events.append(self._event(EventKind.APPEARED, entity, frame_id))
                continue
            events.extend(self._transitions(previous, entity, frame_id))

        for entity_id, previous in self._previous.items():
            if entity_id not in current:
                events.append(self._event(EventKind.DISAPPEARED, previous, frame_id))

        self._previous = current
        events.sort(key=lambda e: (e.entity_id, e.kind))
        return tuple(events)

    def _transitions(self, previous: Entity, current: Entity, frame_id: int) -> list[TrackingEvent]:
        events: list[TrackingEvent] = []

        if previous.health is not None and current.health is not None:
            depleted = current.health <= self._depletion_health < previous.health
            dropped = current.health < previous.health - self._combat_drop
            if depleted and current.kind is EntityKind.RESOURCE_NODE:
                events.append(self._event(EventKind.DEPLETED, current, frame_id))
            elif dropped:
                events.append(
                    self._event(
                        EventKind.COMBAT,
                        current,
                        frame_id,
                        detail=f"{previous.health:.2f}->{current.health:.2f}",
                    )
                )

        if (
            previous.entity_type is not None
            and current.entity_type is not None
            and previous.entity_type != current.entity_type
        ):
            events.append(
                self._event(
                    EventKind.STATE_CHANGED,
                    current,
                    frame_id,
                    detail=f"{previous.entity_type}->{current.entity_type}",
                )
            )
        return events

    @staticmethod
    def _event(
        kind: EventKind, entity: Entity, frame_id: int, *, detail: str | None = None
    ) -> TrackingEvent:
        return TrackingEvent(
            kind=kind, entity_id=entity.entity_id, frame_id=frame_id, detail=detail
        )


_DOMAIN_KIND: dict[EventKind, DomainEventKind] = {
    EventKind.APPEARED: DomainEventKind.UNIT_APPEARED,
    EventKind.DISAPPEARED: DomainEventKind.UNIT_DISAPPEARED,
    EventKind.COMBAT: DomainEventKind.COMBAT,
    EventKind.DEPLETED: DomainEventKind.RESOURCE_DEPLETED,
    EventKind.STATE_CHANGED: DomainEventKind.STATE_CHANGED,
}


class ClassicalEventDetector:
    """``EventSource`` adapter: wraps :class:`EventDetector`, mapping to domain events.

    The transition is inferred deterministically from the entity contract, so it is
    ``CLASSICAL``; its sureness inherits the current entity's confidence value where
    the entity is still present (appear/combat/state-change), and is taken as certain
    for a disappearance (the id definitively left the tracked set). The infra event's
    ``detail`` string is a debug aid and is dropped from the domain contract.
    """

    def __init__(self, detector: EventDetector | None = None) -> None:
        self._detector = detector if detector is not None else EventDetector()

    @classmethod
    def from_settings(cls, settings: TrackingSettings) -> ClassicalEventDetector:
        """Build from pure config (Approach B boundary mapping)."""

        return cls(EventDetector.from_settings(settings))

    def detect(self, entities: Sequence[Entity], frame_id: int) -> tuple[Event, ...]:
        by_id = {entity.entity_id: entity for entity in entities}
        events: list[Event] = []
        for raw in self._detector.detect(entities, frame_id):
            source = by_id.get(raw.entity_id)
            value = source.confidence.value if source is not None else 1.0
            events.append(
                Event(
                    kind=_DOMAIN_KIND[raw.kind],
                    frame_id=raw.frame_id,
                    entity_id=raw.entity_id,
                    confidence=Confidence(value=value, provenance=Provenance.CLASSICAL),
                )
            )
        return tuple(events)
