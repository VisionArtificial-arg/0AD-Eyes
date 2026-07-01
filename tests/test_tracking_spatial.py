"""G9 — spatial reasoning: distance, proximity pairs, neighbours, occupancy."""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.tracking import (
    OccupancyGrid,
    distance,
    neighbours,
    proximity_pairs,
)


def _entity(entity_id: int, x: float, y: float, *, located: bool = True) -> Entity:
    return Entity(
        entity_id=entity_id,
        kind=EntityKind.UNIT,
        world_pos=WorldPoint(x=x, y=y) if located else None,
        confidence=Confidence(value=1.0, provenance=Provenance.CLASSICAL),
    )


def test_distance_uses_world_position() -> None:
    assert distance(_entity(0, 0, 0), _entity(1, 3, 4)) == 5.0


def test_distance_falls_back_to_screen_centre() -> None:
    a = Entity(
        entity_id=0,
        kind=EntityKind.UNIT,
        screen_bbox=ScreenBBox(x=0, y=0, width=10, height=10),
        confidence=Confidence.certain(),
    )
    b = Entity(
        entity_id=1,
        kind=EntityKind.UNIT,
        screen_bbox=ScreenBBox(x=10, y=0, width=10, height=10),
        confidence=Confidence.certain(),
    )
    # centres are (5,5) and (15,5) → distance 10
    assert distance(a, b) == 10.0


def test_distance_is_none_when_unlocated() -> None:
    assert distance(_entity(0, 0, 0, located=False), _entity(1, 1, 1)) is None


def test_proximity_pairs_are_ordered_and_thresholded() -> None:
    entities = [_entity(0, 0, 0), _entity(1, 1, 0), _entity(2, 100, 0)]
    assert proximity_pairs(entities, radius=5.0) == ((0, 1),)


def test_neighbours_excludes_self_and_faraway() -> None:
    target = _entity(0, 0, 0)
    others = [target, _entity(1, 2, 0), _entity(2, 50, 0)]
    assert neighbours(target, others, radius=5.0) == (1,)


def test_occupancy_grid_buckets_entities_into_cells() -> None:
    entities = [_entity(0, 1, 1), _entity(1, 2, 2), _entity(2, 25, 5)]
    grid = OccupancyGrid(entities, cell_size=10.0)

    assert grid.is_occupied((0, 0))
    assert grid.occupants((0, 0)) == (0, 1)  # both in the first 10x10 cell
    assert grid.occupants((2, 0)) == (2,)
    assert grid.occupied_cells() == ((0, 0), (2, 0))
    assert not grid.is_occupied((5, 5))
