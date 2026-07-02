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

from zero_ad_eyes.application.settings import GeometrySettings
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.domain.minimap import MinimapModel, ViewportRect
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership
from zero_ad_eyes.infrastructure.geometry.fusion import reconcile


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


def _fuse_world_pos(higher: Entity, lower: Entity, agreement_scale: float) -> WorldPoint | None:
    """Fuse the two world positions (the one spatial attribute).

    When both sources carry a position, delegate the blend to the F3 geometry
    reconciler (:func:`geometry.fusion.reconcile`) — G5 is the single home for
    fusion and treats the two-estimate spatial case as F3's pure-geometry special
    case, rather than winner-take-all. ``agreement_scale`` (the G5 1/e distance
    discount) comes from the ``geometry`` config. Only the reconciled *point* is used
    here; the entity's overall confidence stays the winner's value (attribute-level G5
    policy, below). When only one source has a position, backfill from whichever has it.

    Eventual target: inverse-variance weighting once each source exposes a position
    covariance; ``reconcile``'s confidence-weighted mean is the documented heuristic
    until then (REQUIREMENTS.md G5).
    """

    if higher.world_pos is not None and lower.world_pos is not None:
        fused_point, _ = reconcile(
            higher.world_pos,
            higher.confidence,
            lower.world_pos,
            lower.confidence,
            agreement_scale=agreement_scale,
        )
        return fused_point
    return higher.world_pos if higher.world_pos is not None else lower.world_pos


def resolve_conflict(primary: Entity, secondary: Entity, *, agreement_scale: float) -> Entity:
    """Confidence-weighted merge of two entities held to be the same thing (G5).

    Each attribute is taken from the more-confident source, falling back to the
    other when the winner's value is missing/unknown — except the world position,
    which is *blended* via the F3 reconciler (see :func:`_fuse_world_pos`), using the
    ``geometry`` config's ``agreement_scale``. The persistent ``entity_id`` is always
    the primary's (the tracked identity); ``staleness`` takes the freshest (minimum)
    of the two; the fused confidence keeps the winning value but is re-tagged ``FUSED``.
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
        world_pos=_fuse_world_pos(higher, lower, agreement_scale),
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
    match_radius: float,
    agreement_scale: float,
) -> tuple[Entity, ...]:
    """Fuse ``primary`` entities with ``hints`` into one coherent set (G4).

    A hint is greedily matched to the nearest primary within ``match_radius`` (same
    coordinate frame) and merged via :func:`resolve_conflict`. Hints that match no
    primary are appended unchanged — they represent things known only to the other
    source (e.g. a minimap blip currently off the main viewport). Both ``match_radius``
    (G4) and ``agreement_scale`` (G5) come from the ``geometry`` config.
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
            merged[best_idx] = resolve_conflict(
                primary[best_idx], hint, agreement_scale=agreement_scale
            )
            consumed.add(best_idx)

    return tuple(merged)


def _inside_viewport(point: WorldPoint, viewport: ViewportRect) -> bool:
    """Whether a world point falls within the camera viewport rect (order-agnostic)."""

    x_lo, x_hi = sorted((viewport.top_left.x, viewport.bottom_right.x))
    y_lo, y_hi = sorted((viewport.top_left.y, viewport.bottom_right.y))
    return x_lo <= point.x <= x_hi and y_lo <= point.y <= y_hi


class ClassicalEntityFuser:
    """``EntityFuser`` adapter (G4/G5): folds minimap blips into the entity set.

    The tracked entities and the minimap blips do not yet share a coordinate frame —
    tracker entities carry a *screen* bbox while blips carry a *world* position, and
    the offline path has no screen→world projector (F1) to bridge them. So this adapter
    does the half of G4 that IS well-defined today: it surfaces the blips that lie
    **outside the camera viewport** — units the main view cannot see — as their own
    world-space entities, and deliberately drops in-viewport blips (which would be
    double-counts of on-screen tracked entities it cannot yet merge with). It still
    routes everything through :func:`fuse_entities` with ``cfg.geometry`` (match_radius,
    agreement_scale), so once entities gain world positions (F1 wired) the positional
    match-and-merge activates with no change here. When the viewport is unknown it emits
    no hints, rather than risk double-counting.
    """

    def __init__(self, *, match_radius: float, agreement_scale: float) -> None:
        self._match_radius = match_radius
        self._agreement_scale = agreement_scale

    @classmethod
    def from_settings(cls, geometry: GeometrySettings) -> ClassicalEntityFuser:
        """Build from the ``geometry`` config (Approach B boundary mapping)."""

        return cls(
            match_radius=geometry.fusion_match_radius,
            agreement_scale=geometry.fusion_agreement_scale,
        )

    def fuse(self, entities: Sequence[Entity], minimap: MinimapModel) -> tuple[Entity, ...]:
        hints = self._blip_hints(entities, minimap)
        return fuse_entities(
            entities,
            hints,
            match_radius=self._match_radius,
            agreement_scale=self._agreement_scale,
        )

    def _blip_hints(self, entities: Sequence[Entity], minimap: MinimapModel) -> tuple[Entity, ...]:
        if minimap.viewport is None:
            return ()  # no viewport → cannot tell on- from off-screen; emit nothing
        base = max((e.entity_id for e in entities), default=-1) + 1
        hints: list[Entity] = []
        for offset, blip in enumerate(minimap.blips):
            if _inside_viewport(blip.world_pos, minimap.viewport):
                continue  # on-screen: already tracked from the main view; skip (no double-count)
            hints.append(
                Entity(
                    entity_id=base + offset,
                    kind=EntityKind.OTHER,  # a blip is a coloured dot: coarse class unknown
                    ownership=blip.ownership,
                    world_pos=blip.world_pos,
                    confidence=blip.confidence,
                )
            )
        return tuple(hints)
