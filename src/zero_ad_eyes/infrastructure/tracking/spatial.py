"""Spatial reasoning between entities (REQUIREMENTS.md G9, CV-33).

Pure helpers that answer proximity/occupancy questions over a set of domain
``Entity`` values: how far apart two things are, which pairs are close enough to
interact, which entities neighbour a target, and how entities bucket into a coarse
grid (occupancy). Positions are read from ``world_pos`` when available, else the
screen-bbox centre; both operands are assumed to share a coordinate frame.

These functions carry no state and no pixels; they are the geometric vocabulary the
world model and, later, the decision layer reason with.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from zero_ad_eyes.domain.entities import Entity


def _position(entity: Entity) -> tuple[float, float] | None:
    if entity.world_pos is not None:
        return (entity.world_pos.x, entity.world_pos.y)
    if entity.screen_bbox is not None:
        centre = entity.screen_bbox.center
        return (centre.x, centre.y)
    return None


def distance(a: Entity, b: Entity) -> float | None:
    """Euclidean distance between two entities, or ``None`` if either is unlocated."""

    pa, pb = _position(a), _position(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def proximity_pairs(entities: Sequence[Entity], radius: float) -> tuple[tuple[int, int], ...]:
    """All id pairs within ``radius`` of each other, ordered ``(low_id, high_id)``."""

    pairs: list[tuple[int, int]] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            d = distance(entities[i], entities[j])
            if d is not None and d <= radius:
                a_id, b_id = entities[i].entity_id, entities[j].entity_id
                pairs.append((min(a_id, b_id), max(a_id, b_id)))
    pairs.sort()
    return tuple(pairs)


def neighbours(target: Entity, entities: Sequence[Entity], radius: float) -> tuple[int, ...]:
    """Ids of entities within ``radius`` of ``target`` (excluding the target id)."""

    found: list[int] = []
    for entity in entities:
        if entity.entity_id == target.entity_id:
            continue
        d = distance(target, entity)
        if d is not None and d <= radius:
            found.append(entity.entity_id)
    found.sort()
    return tuple(found)


class OccupancyGrid:
    """Buckets located entities into square cells for coarse occupancy queries."""

    def __init__(self, entities: Sequence[Entity], *, cell_size: float) -> None:
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        self._cell_size = cell_size
        self._cells: dict[tuple[int, int], list[int]] = {}
        for entity in entities:
            position = _position(entity)
            if position is None:
                continue
            self._cells.setdefault(self.cell_of(position), []).append(entity.entity_id)

    def cell_of(self, position: tuple[float, float]) -> tuple[int, int]:
        return (
            math.floor(position[0] / self._cell_size),
            math.floor(position[1] / self._cell_size),
        )

    def occupants(self, cell: tuple[int, int]) -> tuple[int, ...]:
        return tuple(sorted(self._cells.get(cell, ())))

    def is_occupied(self, cell: tuple[int, int]) -> bool:
        return bool(self._cells.get(cell))

    def occupied_cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._cells))
