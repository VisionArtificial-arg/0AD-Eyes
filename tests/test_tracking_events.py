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
    detector = EventDetector(combat_drop=0.05, depletion_health=0.02)

    born = detector.detect((_entity(0),), frame_id=0)
    assert [e.kind for e in born] == [EventKind.APPEARED]

    steady = detector.detect((_entity(0),), frame_id=1)
    assert steady == ()

    gone = detector.detect((), frame_id=2)
    assert [(e.kind, e.entity_id) for e in gone] == [(EventKind.DISAPPEARED, 0)]


def test_combat_on_health_drop() -> None:
    detector = EventDetector(combat_drop=0.05, depletion_health=0.02)
    detector.detect((_entity(0, health=1.0),), frame_id=0)
    events = detector.detect((_entity(0, health=0.7),), frame_id=1)

    assert len(events) == 1
    assert events[0].kind is EventKind.COMBAT
    assert events[0].detail == "1.00->0.70"


def test_small_health_jitter_is_not_combat() -> None:
    detector = EventDetector(combat_drop=0.05, depletion_health=0.02)
    detector.detect((_entity(0, health=1.0),), frame_id=0)
    assert detector.detect((_entity(0, health=0.98),), frame_id=1) == ()


def test_resource_depletion() -> None:
    detector = EventDetector(combat_drop=0.05, depletion_health=0.02)
    detector.detect((_entity(0, kind=EntityKind.RESOURCE_NODE, health=0.3),), frame_id=0)
    events = detector.detect((_entity(0, kind=EntityKind.RESOURCE_NODE, health=0.0),), frame_id=1)

    assert [e.kind for e in events] == [EventKind.DEPLETED]


def test_state_change_models_build_complete() -> None:
    detector = EventDetector(combat_drop=0.05, depletion_health=0.02)
    detector.detect((_entity(0, kind=EntityKind.BUILDING, entity_type="scaffold"),), frame_id=0)
    events = detector.detect(
        (_entity(0, kind=EntityKind.BUILDING, entity_type="house"),), frame_id=1
    )

    assert len(events) == 1
    assert events[0].kind is EventKind.STATE_CHANGED
    assert events[0].detail == "scaffold->house"


def test_events_are_deterministically_ordered() -> None:
    detector = EventDetector(combat_drop=0.05, depletion_health=0.02)
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

    adapter = ClassicalEventDetector(EventDetector(combat_drop=0.05, depletion_health=0.02))

    born = adapter.detect((_entity(0),), frame_id=0)
    assert [e.kind for e in born] == [DomainEventKind.UNIT_APPEARED]
    assert born[0].entity_id == 0
    assert born[0].confidence.provenance is Provenance.CLASSICAL

    gone = adapter.detect((), frame_id=1)
    assert [e.kind for e in gone] == [DomainEventKind.UNIT_DISAPPEARED]


def test_domain_events_carry_origin_details_and_position() -> None:
    from zero_ad_eyes.domain.events import EventKind as DomainEventKind
    from zero_ad_eyes.domain.events import EventOrigin
    from zero_ad_eyes.domain.geometry import ScreenBBox, ScreenPoint, WorldPoint
    from zero_ad_eyes.domain.taxonomy import Ownership
    from zero_ad_eyes.infrastructure.tracking import ClassicalEventDetector

    def placed(health: float) -> Entity:
        return _entity(3, health=health, entity_type="spearman").model_copy(
            update={
                "ownership": Ownership.ENEMY,
                "screen_bbox": ScreenBBox(x=10, y=20, width=4, height=6),
                "world_pos": WorldPoint(x=100.0, y=50.0),
            }
        )

    adapter = ClassicalEventDetector(EventDetector(combat_drop=0.05, depletion_health=0.02))
    adapter.detect((placed(1.0),), frame_id=0)
    (combat,) = adapter.detect((placed(0.6),), frame_id=1)
    assert combat.kind is DomainEventKind.COMBAT
    assert combat.origin is EventOrigin.ENTITY
    assert combat.details == {
        "entity_kind": "unit",
        "ownership": "enemy",
        "entity_type": "spearman",
        "health": 0.6,
        "health_before": 1.0,
        "health_after": 0.6,
    }
    assert combat.screen_pos == ScreenPoint(x=12.0, y=23.0)
    assert combat.world_pos == WorldPoint(x=100.0, y=50.0)

    (gone,) = adapter.detect((), frame_id=2)
    assert gone.kind is DomainEventKind.UNIT_DISAPPEARED
    assert gone.world_pos == WorldPoint(x=100.0, y=50.0)  # where it was last seen
    assert gone.details["ownership"] == "enemy"


def test_state_change_details_name_both_types() -> None:
    from zero_ad_eyes.infrastructure.tracking import ClassicalEventDetector

    adapter = ClassicalEventDetector(EventDetector(combat_drop=0.05, depletion_health=0.02))
    adapter.detect((_entity(0, entity_type="scaffold"),), frame_id=0)
    (event,) = adapter.detect((_entity(0, entity_type="house"),), frame_id=1)
    assert event.details["type_before"] == "scaffold"
    assert event.details["type_after"] == "house"


def test_v02_events_without_payload_still_validate() -> None:
    from zero_ad_eyes.domain.events import Event, EventOrigin

    event = Event.model_validate({"kind": "combat", "frame_id": 4, "entity_id": 1})
    assert event.origin is EventOrigin.UNKNOWN
    assert event.details == {} and event.screen_pos is None and event.world_pos is None
