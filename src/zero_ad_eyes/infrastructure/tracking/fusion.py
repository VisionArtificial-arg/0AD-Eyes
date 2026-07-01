"""Fusion & conflict resolution (REQUIREMENTS.md G4/G5, CV-20/CV-21).

Combine the tracker's per-frame ``Entity`` set with optional *hints* derived from
other surfaces (minimap blips, HUD-selected entity) into one coherent world view.
When two sources describe the same thing but disagree, the conflict is resolved by
**confidence weighting**: each attribute is taken from whichever source is more
sure, and the result is stamped ``Provenance.FUSED`` so downstream knows it is a
reconciliation, not a raw observation.

These are pure functions over domain ``Entity`` values — no tracker state, no
pixels — so the same helper fuses stub-fed and real-fed entities identically. Both
entity sets are assumed to share a coordinate frame (both world, or both screen);
cross-frame reconciliation (main-view ⇄ minimap) is wired in at integration (F3).
"""

from __future__ import annotations

from collections.abc import Sequence

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.taxonomy import Ownership


def _position(entity: Entity) -> tuple[float, float] | None:
    """Comparable (x, y) for matching: world position first, else screen centre."""

    if entity.world_pos is not None:
        return (entity.world_pos.x, entity.world_pos.y)
    if entity.screen_bbox is not None:
        centre = entity.screen_bbox.center
        return (centre.x, centre.y)
    return None


def _sq_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy


def resolve_conflict(primary: Entity, secondary: Entity) -> Entity:
    """Confidence-weighted merge of two entities held to be the same thing (G5).

    Each attribute is taken from the more-confident source, falling back to the
    other when the winner's value is missing/unknown. The persistent ``entity_id``
    is always the primary's (the tracked identity); ``staleness`` takes the freshest
    (minimum) of the two; the fused confidence keeps the winning value but is
    re-tagged ``FUSED``.
    """

    higher, lower = (
        (primary, secondary)
        if primary.confidence.value >= secondary.confidence.value
        else (secondary, primary)
    )

    ownership = higher.ownership if higher.ownership is not Ownership.UNKNOWN else lower.ownership
    return Entity(
        entity_id=primary.entity_id,
        kind=higher.kind,
        ownership=ownership,
        entity_type=higher.entity_type if higher.entity_type is not None else lower.entity_type,
        world_pos=higher.world_pos if higher.world_pos is not None else lower.world_pos,
        screen_bbox=higher.screen_bbox if higher.screen_bbox is not None else lower.screen_bbox,
        health=higher.health if higher.health is not None else lower.health,
        selected=higher.selected or lower.selected,
        staleness=min(primary.staleness, secondary.staleness),
        confidence=Confidence(value=higher.confidence.value, provenance=Provenance.FUSED),
    )


def fuse_entities(
    primary: Sequence[Entity],
    hints: Sequence[Entity],
    *,
    match_radius: float = 20.0,
) -> tuple[Entity, ...]:
    """Fuse ``primary`` entities with ``hints`` into one coherent set (G4).

    A hint is greedily matched to the nearest primary within ``match_radius`` (same
    coordinate frame) and merged via :func:`resolve_conflict`. Hints that match no
    primary are appended unchanged — they represent things known only to the other
    source (e.g. a minimap blip currently off the main viewport).
    """

    radius_sq = match_radius * match_radius
    primary_positions = [_position(p) for p in primary]
    merged = list(primary)
    consumed: set[int] = set()

    for hint in hints:
        hint_pos = _position(hint)
        best_idx: int | None = None
        best_dist = float("inf")
        if hint_pos is not None:
            for idx, pos in enumerate(primary_positions):
                if idx in consumed or pos is None:
                    continue
                dist = _sq_distance(hint_pos, pos)
                if dist <= radius_sq and dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        if best_idx is None:
            merged.append(hint)
        else:
            merged[best_idx] = resolve_conflict(primary[best_idx], hint)
            consumed.add(best_idx)

    return tuple(merged)
