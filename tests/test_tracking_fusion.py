"""G4/G5 — fusion of entities with hints and confidence-weighted conflict resolution."""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership
from zero_ad_eyes.infrastructure.tracking import fuse_entities, resolve_conflict


def _entity(
    entity_id: int,
    *,
    kind: EntityKind = EntityKind.UNIT,
    x: float = 0.0,
    y: float = 0.0,
    value: float = 0.5,
    ownership: Ownership = Ownership.UNKNOWN,
    health: float | None = None,
) -> Entity:
    return Entity(
        entity_id=entity_id,
        kind=kind,
        ownership=ownership,
        world_pos=WorldPoint(x=x, y=y),
        health=health,
        confidence=Confidence(value=value, provenance=Provenance.CLASSICAL),
    )


def test_resolve_conflict_prefers_higher_confidence_and_fills_gaps() -> None:
    primary = _entity(0, value=0.4, ownership=Ownership.UNKNOWN, health=None)
    secondary = _entity(9, value=0.9, ownership=Ownership.ENEMY, health=0.5)

    merged = resolve_conflict(primary, secondary, agreement_scale=1.0)

    assert merged.entity_id == 0  # tracked identity preserved
    assert merged.ownership is Ownership.ENEMY  # winner supplies ownership
    assert merged.health == 0.5  # gap filled from the more-confident source
    assert merged.confidence.value == 0.9
    assert merged.confidence.provenance is Provenance.FUSED


def test_resolve_conflict_backfills_from_lower_confidence_when_winner_unknown() -> None:
    primary = _entity(0, value=0.9, ownership=Ownership.UNKNOWN, health=None)
    secondary = _entity(1, value=0.2, ownership=Ownership.ALLY, health=0.7)

    merged = resolve_conflict(primary, secondary, agreement_scale=1.0)

    assert merged.ownership is Ownership.ALLY  # winner unknown → backfilled
    assert merged.health == 0.7
    assert merged.confidence.value == 0.9


def test_resolve_conflict_blends_world_pos_via_geometry_reconciler() -> None:
    # G5 delegates the spatial attribute to the F3 reconciler: the fused position is
    # the confidence-weighted blend of both, not the winner's coordinate.
    primary = _entity(0, x=0.0, y=0.0, value=0.4)
    secondary = _entity(9, x=10.0, y=0.0, value=0.9)

    merged = resolve_conflict(primary, secondary, agreement_scale=1.0)

    assert merged.world_pos is not None
    # blended toward the more-confident source (10.0) but strictly between the two
    assert 0.0 < merged.world_pos.x < 10.0
    assert merged.world_pos.x > 5.0
    # entity confidence stays winner-take-all (attribute policy), only re-tagged
    assert merged.confidence.value == 0.9
    assert merged.confidence.provenance is Provenance.FUSED


def test_fuse_matches_nearby_hint_and_keeps_faraway_hint_separate() -> None:
    primary = (_entity(0, x=0.0, y=0.0, value=0.5, ownership=Ownership.UNKNOWN),)
    hints = (
        _entity(100, x=1.0, y=1.0, value=0.9, ownership=Ownership.ENEMY),  # same thing
        _entity(200, x=500.0, y=500.0, value=0.8, ownership=Ownership.GAIA),  # elsewhere
    )

    fused = fuse_entities(primary, hints, match_radius=10.0, agreement_scale=1.0)

    assert len(fused) == 2
    reconciled = next(e for e in fused if e.entity_id == 0)
    assert reconciled.ownership is Ownership.ENEMY
    assert reconciled.confidence.provenance is Provenance.FUSED
    assert any(e.entity_id == 200 for e in fused)  # far hint kept as its own entity


def test_fuse_without_hints_is_identity() -> None:
    primary = (_entity(0), _entity(1, x=50.0))
    assert fuse_entities(primary, (), match_radius=20.0, agreement_scale=1.0) == primary


def test_fuse_matches_each_primary_at_most_once() -> None:
    primary = (_entity(0, x=0.0),)
    hints = (_entity(10, x=1.0, value=0.9), _entity(11, x=2.0, value=0.8))

    fused = fuse_entities(primary, hints, match_radius=10.0, agreement_scale=1.0)

    # one hint merges into primary 0; the second has no free primary → appended
    assert len(fused) == 2
    assert {e.entity_id for e in fused} == {0, 11}
