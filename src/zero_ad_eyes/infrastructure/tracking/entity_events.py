"""Entity state event source (EVT-5): what happens to tracked entities between frames.

``EntityStateEventSource`` adds to the classical entity events (appeared, combat, …)
everything else the entity stream lets us observe. All events are ``origin=entity``:

- **sight:** an entity stops being observed (``staleness`` goes above 0). If its last
  world position is outside the camera's footprint (the minimap viewport) it
  ``entity_left_view``; otherwise it was lost inside the view (fog, occlusion) and is
  ``entity_hidden``. Seen again: ``entity_entered_view`` or ``entity_revealed``, with
  how many frames it was unseen.
- **health:** ``health_recovered`` when health rises by ``events.entity_heal_min``
  (repair, healing, a building going up). Drops are the classical ``combat``.
- **motion:** ``started_moving`` / ``stopped_moving`` from the tracker's velocity, with
  hysteresis between ``events.entity_stop_max_speed`` and
  ``events.entity_move_min_speed`` (pixels per frame).
- **selection:** ``entity_selected`` / ``entity_deselected``.
- **spawns:** ``building_seen`` for every new building, and ``possible_unit_trained``
  when a new own unit appears within ``events.entity_spawn_radius`` px of an own
  building. That one is a guess and says so in its name; it is not emitted on the first
  frame, when everything is new.

Construction progress and garrison counts are not on the entity contract (the state
cues stay inside the enricher), so they have no events here.
"""

from __future__ import annotations

import math
from typing import Protocol

from zero_ad_eyes.application.frames import PerceptionContext
from zero_ad_eyes.application.settings import EventSettings
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.events import DetailValue, Event, EventKind, EventOrigin
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.minimap import ViewportQuad
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership
from zero_ad_eyes.domain.world_model import WorldModel

LEFT_VIEW, HIDDEN = "left_view", "hidden"


class EntityStateEventSource:
    """``EventSource`` for entity sight, health recovery, motion, selection and spawns."""

    def __init__(
        self, *, heal_min: float, move_min_speed: float, stop_max_speed: float, spawn_radius: float
    ) -> None:
        self._heal_min = heal_min
        self._move_min = move_min_speed
        self._stop_max = stop_max_speed
        self._spawn_radius = spawn_radius
        self._previous: dict[int, Entity] | None = None
        self._lost_reason: dict[int, str] = {}
        self._moving: dict[int, bool] = {}

    @classmethod
    def from_settings(cls, settings: EventSettings) -> EntityStateEventSource:
        return cls(
            heal_min=settings.entity_heal_min,
            move_min_speed=settings.entity_move_min_speed,
            stop_max_speed=settings.entity_stop_max_speed,
            spawn_radius=settings.entity_spawn_radius,
        )

    def events(self, world_model: WorldModel, context: PerceptionContext) -> tuple[Event, ...]:
        viewport = world_model.minimap.viewport if world_model.minimap else None
        frame_id = world_model.meta.frame_id
        current = {e.entity_id: e for e in world_model.entities}
        first_frame = self._previous is None
        previous = self._previous or {}
        self._previous = current

        found: list[Event] = []
        for entity_id in sorted(current):
            entity, before = current[entity_id], previous.get(entity_id)

            def event(kind: EventKind, entity: Entity = entity, **details: DetailValue) -> Event:
                return _entity_event(kind, entity, frame_id, details)

            if before is None:
                found += self._spawn_events(entity, current, first_frame, event)
            else:
                found += self._sight_events(before, entity, viewport, event)
                found += self._health_events(before, entity, event)
                found += self._selection_events(before, entity, event)
            found += self._motion_events(entity, event)
        for entity_id in set(previous) - set(current):
            self._lost_reason.pop(entity_id, None)
            self._moving.pop(entity_id, None)
        return tuple(found)

    def _sight_events(
        self, before: Entity, now: Entity, viewport: ViewportQuad | None, event: _Factory
    ) -> list[Event]:
        if before.staleness == 0 and now.staleness > 0:
            where = now.world_pos or before.world_pos
            reason = (
                LEFT_VIEW
                if viewport is not None and where is not None and not viewport.contains(where)
                else HIDDEN
            )
            self._lost_reason[now.entity_id] = reason
            kind = EventKind.ENTITY_LEFT_VIEW if reason == LEFT_VIEW else EventKind.ENTITY_HIDDEN
            return [event(kind, viewport_known=viewport is not None)]
        if before.staleness > 0 and now.staleness == 0:
            reason = self._lost_reason.pop(now.entity_id, HIDDEN)
            kind = (
                EventKind.ENTITY_ENTERED_VIEW if reason == LEFT_VIEW else EventKind.ENTITY_REVEALED
            )
            return [event(kind, frames_unseen=before.staleness)]
        return []

    def _health_events(self, before: Entity, now: Entity, event: _Factory) -> list[Event]:
        if before.health is None or now.health is None or now.staleness > 0:
            return []
        if now.health >= before.health + self._heal_min:
            return [
                event(
                    EventKind.HEALTH_RECOVERED, health_before=before.health, health_after=now.health
                )
            ]
        return []

    def _selection_events(self, before: Entity, now: Entity, event: _Factory) -> list[Event]:
        if before.selected == now.selected:
            return []
        return [event(EventKind.ENTITY_SELECTED if now.selected else EventKind.ENTITY_DESELECTED)]

    def _motion_events(self, entity: Entity, event: _Factory) -> list[Event]:
        if entity.motion is None or entity.staleness > 0:
            return []
        speed = math.hypot(entity.motion.dx, entity.motion.dy)
        was_moving = self._moving.get(entity.entity_id, False)
        if not was_moving and speed >= self._move_min:
            self._moving[entity.entity_id] = True
            return [
                event(
                    EventKind.STARTED_MOVING,
                    speed=speed,
                    dx=entity.motion.dx,
                    dy=entity.motion.dy,
                )
            ]
        if was_moving and speed <= self._stop_max:
            self._moving[entity.entity_id] = False
            return [event(EventKind.STOPPED_MOVING, speed=speed)]
        return []

    def _spawn_events(
        self, entity: Entity, current: dict[int, Entity], first_frame: bool, event: _Factory
    ) -> list[Event]:
        if entity.kind is EntityKind.BUILDING:
            return [event(EventKind.BUILDING_SEEN)]
        if (
            first_frame
            or entity.kind is not EntityKind.UNIT
            or entity.ownership is not Ownership.SELF
        ):
            return []
        if entity.screen_bbox is None:
            return []
        center = entity.screen_bbox.center
        nearest: tuple[float, Entity] | None = None
        for other in current.values():
            if (
                other.kind is EntityKind.BUILDING
                and other.ownership is Ownership.SELF
                and other.screen_bbox is not None
                and other.staleness == 0
            ):
                gap = _gap(center.x, center.y, other.screen_bbox)
                if nearest is None or gap < nearest[0]:
                    nearest = (gap, other)
        if nearest is None or nearest[0] > self._spawn_radius:
            return []
        gap, building = nearest
        return [
            event(
                EventKind.POSSIBLE_UNIT_TRAINED,
                building_entity_id=building.entity_id,
                building_type=building.entity_type,
                distance_px=gap,
            )
        ]


def _gap(x: float, y: float, box: ScreenBBox) -> float:
    """Distance from a point to a box (0 inside)."""

    dx = max(box.x - x, 0.0, x - (box.x + box.width))
    dy = max(box.y - y, 0.0, y - (box.y + box.height))
    return math.hypot(dx, dy)


def _entity_event(
    kind: EventKind, entity: Entity, frame_id: int, details: dict[str, DetailValue]
) -> Event:
    context: dict[str, DetailValue] = {
        "entity_kind": entity.kind.value,
        "ownership": entity.ownership.value,
        "entity_type": entity.entity_type,
        "health": entity.health,
    }
    return Event(
        kind=kind,
        frame_id=frame_id,
        entity_id=entity.entity_id,
        confidence=Confidence(value=entity.confidence.value, provenance=Provenance.CLASSICAL),
        origin=EventOrigin.ENTITY,
        details={**context, **details},
        screen_pos=entity.screen_bbox.center if entity.screen_bbox else None,
        world_pos=entity.world_pos,
    )


class _Factory(Protocol):
    """Annotation-only: the per-entity ``event(kind, **details)`` helper."""

    def __call__(self, kind: EventKind, entity: Entity = ..., **details: DetailValue) -> Event: ...
