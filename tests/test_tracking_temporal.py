"""G7 — temporal stabilisation: majority-voted classes, averaged confidence."""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.tracking import TemporalStabilizer, majority


def _entity(kind: EntityKind, value: float, *, entity_type: str | None = None) -> Entity:
    return Entity(
        entity_id=0,
        kind=kind,
        entity_type=entity_type,
        confidence=Confidence(value=value, provenance=Provenance.LEARNED),
    )


def test_majority_breaks_ties_toward_latest() -> None:
    assert majority([1, 1, 2, 2]) == 2
    assert majority(["a", "b", "a"]) == "a"


def test_flickering_kind_is_stabilised_to_the_majority() -> None:
    stabilizer = TemporalStabilizer(window=5)
    kinds = [EntityKind.UNIT, EntityKind.UNIT, EntityKind.RESOURCE_NODE, EntityKind.UNIT]

    out_kinds = []
    for k in kinds:
        (entity,) = stabilizer.stabilize((_entity(k, 0.9),))
        out_kinds.append(entity.kind)

    # the single RESOURCE_NODE frame never wins over the UNIT majority
    assert out_kinds == [
        EntityKind.UNIT,
        EntityKind.UNIT,
        EntityKind.UNIT,
        EntityKind.UNIT,
    ]


def test_confidence_is_averaged_over_the_window() -> None:
    stabilizer = TemporalStabilizer(window=3)
    (e1,) = stabilizer.stabilize((_entity(EntityKind.UNIT, 1.0),))
    (e2,) = stabilizer.stabilize((_entity(EntityKind.UNIT, 0.0),))

    assert e1.confidence.value == 1.0  # first frame: just itself
    assert e2.confidence.value == 0.5  # mean of [1.0, 0.0]
    assert e2.confidence.provenance is Provenance.LEARNED  # latest provenance kept


def test_entity_type_majority_ignores_missing_values() -> None:
    stabilizer = TemporalStabilizer(window=5)
    stabilizer.stabilize((_entity(EntityKind.UNIT, 0.9, entity_type="house"),))
    stabilizer.stabilize((_entity(EntityKind.UNIT, 0.9, entity_type=None),))
    (entity,) = stabilizer.stabilize((_entity(EntityKind.UNIT, 0.9, entity_type="house"),))

    assert entity.entity_type == "house"
