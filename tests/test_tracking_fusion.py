"""G4/G5 — fusion of entities with hints and confidence-weighted conflict resolution."""

from __future__ import annotations

from zero_ad_eyes.application.settings import GeometrySettings
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.domain.minimap import Blip, MinimapModel, ViewportQuad
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership
from zero_ad_eyes.infrastructure.tracking import (
    ClassicalEntityFuser,
    fuse_entities,
    resolve_conflict,
)


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


# --- ClassicalEntityFuser: minimap blips folded into the entity set (G4/G5) -------


def _blip(
    x: float, y: float, *, ownership: Ownership = Ownership.ENEMY, value: float = 0.8
) -> Blip:
    return Blip(
        world_pos=WorldPoint(x=x, y=y),
        ownership=ownership,
        confidence=Confidence(value=value, provenance=Provenance.CLASSICAL),
    )


def _viewport(x0: float, y0: float, x1: float, y1: float) -> ViewportQuad:
    # An axis-aligned quad is the degenerate (top-down) footprint; enough to exercise
    # the on/off-viewport containment logic the fuser relies on.
    return ViewportQuad(
        top_left=WorldPoint(x=x0, y=y0),
        top_right=WorldPoint(x=x1, y=y0),
        bottom_right=WorldPoint(x=x1, y=y1),
        bottom_left=WorldPoint(x=x0, y=y1),
    )


_FUSER = ClassicalEntityFuser(match_radius=20.0, agreement_scale=1.0)


def test_fuser_surfaces_off_viewport_blip_as_world_entity() -> None:
    entities = (_entity(0, x=0.0, y=0.0, value=0.5),)
    minimap = MinimapModel(blips=(_blip(500.0, 500.0),), viewport=_viewport(0.0, 0.0, 100.0, 100.0))

    fused = _FUSER.fuse(entities, minimap)

    extra = [e for e in fused if e.entity_id != 0]
    assert len(extra) == 1
    assert extra[0].world_pos == WorldPoint(x=500.0, y=500.0)
    assert extra[0].ownership is Ownership.ENEMY
    assert extra[0].entity_id != 0  # fresh id, no collision with the tracked entity


def test_fuser_skips_in_viewport_blip_to_avoid_double_count() -> None:
    entities = (_entity(0, x=0.0, y=0.0),)
    minimap = MinimapModel(blips=(_blip(50.0, 50.0),), viewport=_viewport(0.0, 0.0, 100.0, 100.0))

    # The blip is on-screen (inside the viewport); without a screen→world projector we
    # cannot merge it with the tracked entity, so it is dropped rather than duplicated.
    assert _FUSER.fuse(entities, minimap) == entities


def test_fuser_without_viewport_emits_no_hints() -> None:
    entities = (_entity(0),)
    minimap = MinimapModel(blips=(_blip(500.0, 500.0),), viewport=None)
    assert _FUSER.fuse(entities, minimap) == entities


def test_fuser_from_settings_reads_geometry() -> None:
    fuser = ClassicalEntityFuser.from_settings(
        GeometrySettings(
            camera_error_tolerance=1.0, fusion_agreement_scale=2.0, fusion_match_radius=15.0
        )
    )
    assert fuser._match_radius == 15.0
    assert fuser._agreement_scale == 2.0
