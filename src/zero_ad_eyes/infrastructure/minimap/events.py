"""Minimap event source (EVT-4): transitions read off the minimap.

``MinimapEventSource`` diffs each frame's ``MinimapModel`` against the last one it saw
and emits ``origin=minimap`` events:

- **blips:** ``blip_appeared`` / ``blip_vanished``. Blips carry no id, so each one is
  matched to the nearest same-owner blip of the previous frame within
  ``events.minimap_blip_match_radius``; unmatched blips are new or gone.
- **threat:** ``enemy_in_territory`` when an enemy blip enters our territory. A territory
  region is known only by centroid and coverage, so it is taken as a disc of that area.
- **territory:** ``territory_gained`` / ``territory_lost`` per owner, when its share of
  the map moves by at least ``events.minimap_territory_min_change``.
- **fog:** ``area_explored`` (unexplored → seen), ``area_visible`` (shroud → in sight)
  and ``area_fogged`` (in sight → shroud). One event per kind per frame, carrying the
  number of cells, so the start of a match does not flood the stream.
- **camera:** ``camera_moved`` when the viewport centre moves at least
  ``events.minimap_camera_move_min`` world units.

Policy: events over silence, with recall-biased defaults. A part of the minimap that
was not read this frame (no fog grid, no viewport) is not a change: the last reading
is kept.
"""

from __future__ import annotations

import math
from typing import Protocol

from zero_ad_eyes.application.frames import PerceptionContext
from zero_ad_eyes.application.settings import EventSettings, WorldExtentSettings
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.events import DetailValue, Event, EventKind, EventOrigin
from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.domain.minimap import Blip, FogGrid, FogState, MinimapModel, TerritoryMap
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.domain.world_model import WorldModel

_SEEN = (FogState.EXPLORED, FogState.VISIBLE)


class MinimapEventSource:
    """``EventSource`` for the minimap surface."""

    def __init__(
        self,
        *,
        blip_match_radius: float,
        camera_move_min: float,
        territory_min_change: float,
        map_area: float,
    ) -> None:
        self._radius = blip_match_radius
        self._camera_min = camera_move_min
        self._territory_min = territory_min_change
        self._map_area = map_area
        self._blips: tuple[Blip, ...] | None = None
        self._inside: set[int] = set()  # indices into self._blips of enemies in our land
        self._fog: FogGrid | None = None
        self._coverage: dict[Ownership, float] | None = None
        self._center: WorldPoint | None = None

    @classmethod
    def from_settings(
        cls, settings: EventSettings, world_extent: WorldExtentSettings
    ) -> MinimapEventSource:
        return cls(
            blip_match_radius=settings.minimap_blip_match_radius,
            camera_move_min=settings.minimap_camera_move_min,
            territory_min_change=settings.minimap_territory_min_change,
            map_area=world_extent.width * world_extent.height,
        )

    def events(self, world_model: WorldModel, context: PerceptionContext) -> tuple[Event, ...]:
        minimap = world_model.minimap
        if minimap is None:
            return ()
        frame_id = world_model.meta.frame_id
        surface = Confidence(value=minimap.confidence.value, provenance=Provenance.CLASSICAL)

        def event(
            kind: EventKind,
            *,
            confidence: Confidence = surface,
            world_pos: WorldPoint | None = None,
            **details: DetailValue,
        ) -> Event:
            return Event(
                kind=kind,
                frame_id=frame_id,
                confidence=confidence,
                origin=EventOrigin.MINIMAP,
                details=details,
                world_pos=world_pos,
            )

        found = self._blip_events(minimap, event)
        found += self._territory_events(minimap.territory, event)
        found += self._fog_events(minimap.fog, event)
        found += self._camera_events(minimap, event)
        return tuple(found)

    # -- blips ------------------------------------------------------------- #

    def _blip_events(self, minimap: MinimapModel, event: _Factory) -> list[Event]:
        current = minimap.blips
        inside = {
            i
            for i, blip in enumerate(current)
            if blip.ownership is Ownership.ENEMY and self._in_own_territory(blip, minimap.territory)
        }
        previous, was_inside = self._blips, self._inside
        self._blips, self._inside = current, inside
        if previous is None:
            return []

        matched = self._match(previous, current)
        found: list[Event] = []
        for i, blip in enumerate(current):
            confidence = Confidence(value=blip.confidence.value, provenance=Provenance.CLASSICAL)
            if i not in matched:
                found.append(
                    event(
                        EventKind.BLIP_APPEARED,
                        confidence=confidence,
                        world_pos=blip.world_pos,
                        ownership=blip.ownership.value,
                    )
                )
            if i in inside and matched.get(i) not in was_inside:
                found.append(
                    event(
                        EventKind.ENEMY_IN_TERRITORY,
                        confidence=confidence,
                        world_pos=blip.world_pos,
                        ownership=blip.ownership.value,
                    )
                )
        kept = set(matched.values())
        for j, blip in enumerate(previous):
            if j not in kept:
                found.append(
                    event(
                        EventKind.BLIP_VANISHED,
                        confidence=Confidence(
                            value=blip.confidence.value, provenance=Provenance.CLASSICAL
                        ),
                        world_pos=blip.world_pos,
                        ownership=blip.ownership.value,
                    )
                )
        return found

    def _match(self, previous: tuple[Blip, ...], current: tuple[Blip, ...]) -> dict[int, int]:
        """Greedy nearest same-owner pairing within the radius: current index → previous."""

        pairs = sorted(
            (_distance(c.world_pos, p.world_pos), i, j)
            for i, c in enumerate(current)
            for j, p in enumerate(previous)
            if c.ownership == p.ownership and _distance(c.world_pos, p.world_pos) <= self._radius
        )
        matched: dict[int, int] = {}
        used: set[int] = set()
        for _, i, j in pairs:
            if i not in matched and j not in used:
                matched[i] = j
                used.add(j)
        return matched

    def _in_own_territory(self, blip: Blip, territory: TerritoryMap | None) -> bool:
        if territory is None:
            return False
        for region in territory.regions:
            if region.ownership is not Ownership.SELF:
                continue
            radius = math.sqrt(region.coverage * self._map_area / math.pi)
            if _distance(blip.world_pos, region.centroid) <= radius:
                return True
        return False

    # -- territory, fog, camera ------------------------------------------- #

    def _territory_events(self, territory: TerritoryMap | None, event: _Factory) -> list[Event]:
        if territory is None:
            return []
        coverage: dict[Ownership, float] = {}
        for region in territory.regions:
            coverage[region.ownership] = coverage.get(region.ownership, 0.0) + region.coverage
        # Compare with the last *reported* share per owner, so slow growth still adds up.
        reported = self._coverage
        if reported is None:
            self._coverage = coverage
            return []
        found = []
        for owner in sorted(set(reported) | set(coverage), key=lambda o: o.value):
            before, after = reported.get(owner, 0.0), coverage.get(owner, 0.0)
            if abs(after - before) < self._territory_min or after == before:
                continue
            kind = EventKind.TERRITORY_GAINED if after > before else EventKind.TERRITORY_LOST
            found.append(event(kind, ownership=owner.value, before=before, after=after))
            reported[owner] = after
        return found

    def _fog_events(self, fog: FogGrid | None, event: _Factory) -> list[Event]:
        if fog is None or not fog.cells:
            return []
        previous, self._fog = self._fog, fog
        if previous is None or (previous.rows, previous.cols) != (fog.rows, fog.cols):
            return []
        changes: dict[EventKind, list[tuple[int, int]]] = {
            EventKind.AREA_EXPLORED: [],
            EventKind.AREA_VISIBLE: [],
            EventKind.AREA_FOGGED: [],
        }
        for r, (old_row, new_row) in enumerate(zip(previous.cells, fog.cells, strict=False)):
            for c, (old, new) in enumerate(zip(old_row, new_row, strict=False)):
                if old is FogState.UNEXPLORED and new in _SEEN:
                    changes[EventKind.AREA_EXPLORED].append((r, c))
                elif old is FogState.EXPLORED and new is FogState.VISIBLE:
                    changes[EventKind.AREA_VISIBLE].append((r, c))
                elif old is FogState.VISIBLE and new is FogState.EXPLORED:
                    changes[EventKind.AREA_FOGGED].append((r, c))
        total = fog.rows * fog.cols or 1
        explored = sum(cell in _SEEN for row in fog.cells for cell in row)
        found = []
        for kind, cells in changes.items():
            if cells:
                found.append(
                    event(
                        kind,
                        cells=len(cells),
                        row=sum(r for r, _ in cells) / len(cells),
                        col=sum(c for _, c in cells) / len(cells),
                        explored_fraction=explored / total,
                    )
                )
        return found

    def _camera_events(self, minimap: MinimapModel, event: _Factory) -> list[Event]:
        if minimap.viewport is None:
            return []
        corners = minimap.viewport.corners()
        center = WorldPoint(x=sum(p.x for p in corners) / 4, y=sum(p.y for p in corners) / 4)
        # Compare with the last *reported* position, so a slow pan still adds up.
        previous = self._center
        if previous is None:
            self._center = center
            return []
        moved = _distance(previous, center)
        if moved < self._camera_min or moved == 0:
            return []
        self._center = center
        return [
            event(
                EventKind.CAMERA_MOVED,
                world_pos=center,
                dx=center.x - previous.x,
                dy=center.y - previous.y,
                distance=moved,
            )
        ]


def _distance(a: WorldPoint, b: WorldPoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


class _Factory(Protocol):
    """Annotation-only type of the per-frame ``event(...)`` helper."""

    def __call__(
        self,
        kind: EventKind,
        *,
        confidence: Confidence = ...,
        world_pos: WorldPoint | None = None,
        **details: DetailValue,
    ) -> Event: ...
