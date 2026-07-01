"""G8 — event detection: appear/disappear, combat, depletion, state change."""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.tracking import EventDetector, EventKind


def _entity(
    entity_id: int,
    *,
    kind: EntityKind = EntityKind.UNIT,
    health: float | None = None,
    entity_type: str | None = None,
) -> Entity:
    return Entity(
        entity_id=entity_id,
        kind=kind,
        entity_type=entity_type,
        health=health,
        confidence=Confidence(value=1.0, provenance=Provenance.CLASSICAL),
    )


def test_appearance_and_disappearance() -> None:
    detector = EventDetector()

    born = detector.detect((_entity(0),), frame_id=0)
    assert [e.kind for e in born] == [EventKind.APPEARED]

    steady = detector.detect((_entity(0),), frame_id=1)
    assert steady == ()

    gone = detector.detect((), frame_id=2)
    assert [(e.kind, e.entity_id) for e in gone] == [(EventKind.DISAPPEARED, 0)]


def test_combat_on_health_drop() -> None:
    detector = EventDetector(combat_drop=0.05)
    detector.detect((_entity(0, health=1.0),), frame_id=0)
    events = detector.detect((_entity(0, health=0.7),), frame_id=1)

    assert len(events) == 1
    assert events[0].kind is EventKind.COMBAT
    assert events[0].detail == "1.00->0.70"


def test_small_health_jitter_is_not_combat() -> None:
    detector = EventDetector(combat_drop=0.05)
    detector.detect((_entity(0, health=1.0),), frame_id=0)
    assert detector.detect((_entity(0, health=0.98),), frame_id=1) == ()


def test_resource_depletion() -> None:
    detector = EventDetector(depletion_health=0.02)
    detector.detect((_entity(0, kind=EntityKind.RESOURCE_NODE, health=0.3),), frame_id=0)
    events = detector.detect((_entity(0, kind=EntityKind.RESOURCE_NODE, health=0.0),), frame_id=1)

    assert [e.kind for e in events] == [EventKind.DEPLETED]


def test_state_change_models_build_complete() -> None:
    detector = EventDetector()
    detector.detect((_entity(0, kind=EntityKind.BUILDING, entity_type="scaffold"),), frame_id=0)
    events = detector.detect(
        (_entity(0, kind=EntityKind.BUILDING, entity_type="house"),), frame_id=1
    )

    assert len(events) == 1
    assert events[0].kind is EventKind.STATE_CHANGED
    assert events[0].detail == "scaffold->house"


def test_events_are_deterministically_ordered() -> None:
    detector = EventDetector()
    detector.detect((_entity(2), _entity(5)), frame_id=0)
    # id 2 leaves, id 1 appears → sorted by (entity_id, kind)
    events = detector.detect((_entity(1), _entity(5)), frame_id=1)

    assert [(e.entity_id, e.kind) for e in events] == [
        (1, EventKind.APPEARED),
        (2, EventKind.DISAPPEARED),
    ]


def test_classical_adapter_maps_to_domain_events() -> None:
    from zero_ad_eyes.domain.events import EventKind as DomainEventKind
    from zero_ad_eyes.infrastructure.tracking import ClassicalEventDetector

    adapter = ClassicalEventDetector()

    born = adapter.detect((_entity(0),), frame_id=0)
    assert [e.kind for e in born] == [DomainEventKind.UNIT_APPEARED]
    assert born[0].entity_id == 0
    assert born[0].confidence.provenance is Provenance.CLASSICAL

    gone = adapter.detect((), frame_id=1)
    assert [e.kind for e in gone] == [DomainEventKind.UNIT_DISAPPEARED]
