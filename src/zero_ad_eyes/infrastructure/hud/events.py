"""HUD event source (EVT-3): transitions read off the top bar and selection panel.

``HudEventSource`` diffs each frame's ``HudState`` against the last one it saw and
emits a self-describing event (``origin=hud``) for every change:

- **resources:** ``resource_gained`` / ``resource_spent`` per resource, with the amount;
- **population:** ``population_changed``, ``population_cap_reached``,
  ``population_cap_raised`` / ``population_cap_lowered``;
- **phase:** ``phase_changed``;
- **selection panel:** ``selection_changed``, ``selection_health_changed``,
  ``production_queue_changed``, ``garrison_changed``.

Policy: events over silence. Thresholds default to the smallest observable change and
every event carries the HUD read's confidence, so the consumer can filter OCR noise.
A value the HUD did not read this frame (an absent resource, an ``unknown`` phase, no
HUD at all) is *not* a change: the source keeps the last known value and compares
against it once the value is read again. Read failures are the system source's job.
"""

from __future__ import annotations

from typing import Protocol

from zero_ad_eyes.application.frames import PerceptionContext
from zero_ad_eyes.application.settings import EventSettings
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.events import DetailValue, Event, EventKind, EventOrigin
from zero_ad_eyes.domain.hud import HudState, Population, SelectionState
from zero_ad_eyes.domain.taxonomy import Phase, ResourceType
from zero_ad_eyes.domain.world_model import WorldModel


class HudEventSource:
    """``EventSource`` for the HUD surface."""

    def __init__(self, *, min_resource_delta: int, min_selection_health_delta: float) -> None:
        self._min_resource_delta = min_resource_delta
        self._min_health_delta = min_selection_health_delta
        self._stockpiles: dict[ResourceType, int] = {}
        self._population: Population | None = None
        self._phase = Phase.UNKNOWN
        self._selection: SelectionState | None = None
        self._seen_hud = False

    @classmethod
    def from_settings(cls, settings: EventSettings) -> HudEventSource:
        return cls(
            min_resource_delta=settings.hud_min_resource_delta,
            min_selection_health_delta=settings.hud_min_selection_health_delta,
        )

    def events(self, world_model: WorldModel, context: PerceptionContext) -> tuple[Event, ...]:
        hud = world_model.hud
        if hud is None:
            return ()
        frame_id = world_model.meta.frame_id
        confidence = Confidence(value=hud.confidence.value, provenance=Provenance.CLASSICAL)

        def event(kind: EventKind, **details: DetailValue) -> Event:
            return Event(
                kind=kind,
                frame_id=frame_id,
                confidence=confidence,
                origin=EventOrigin.HUD,
                details=details,
            )

        found: list[Event] = []
        if self._seen_hud:
            found += self._resource_events(hud, event)
            found += self._population_events(hud, event)
            found += self._phase_events(hud, event)
            found += self._selection_events(hud, event)
        self._remember(hud)
        return tuple(found)

    def _resource_events(self, hud: HudState, event: _Factory) -> list[Event]:
        found = []
        for resource in ResourceType:
            before, after = self._stockpiles.get(resource), hud.stockpiles.get(resource)
            if before is None or after is None:
                continue
            delta = after - before
            if abs(delta) < self._min_resource_delta:
                continue
            kind = EventKind.RESOURCE_GAINED if delta > 0 else EventKind.RESOURCE_SPENT
            found.append(
                event(kind, resource=resource.value, before=before, after=after, amount=abs(delta))
            )
        return found

    def _population_events(self, hud: HudState, event: _Factory) -> list[Event]:
        before, after = self._population, hud.population
        if before is None or after is None:
            return []
        found = []
        if after.current != before.current:
            found.append(
                event(
                    EventKind.POPULATION_CHANGED,
                    before=before.current,
                    after=after.current,
                    delta=after.current - before.current,
                    cap=after.cap,
                )
            )
        if after.cap > before.cap:
            found.append(event(EventKind.POPULATION_CAP_RAISED, before=before.cap, after=after.cap))
        elif after.cap < before.cap:
            found.append(
                event(EventKind.POPULATION_CAP_LOWERED, before=before.cap, after=after.cap)
            )
        if after.current >= after.cap > 0 and before.current < before.cap:
            found.append(
                event(EventKind.POPULATION_CAP_REACHED, current=after.current, cap=after.cap)
            )
        return found

    def _phase_events(self, hud: HudState, event: _Factory) -> list[Event]:
        if Phase.UNKNOWN in (self._phase, hud.phase) or hud.phase == self._phase:
            return []
        return [event(EventKind.PHASE_CHANGED, before=self._phase.value, after=hud.phase.value)]

    def _selection_events(self, hud: HudState, event: _Factory) -> list[Event]:
        before, after = self._selection, hud.selection
        before_type = before.entity_type if before else None
        after_type = after.entity_type if after else None
        if before_type != after_type:
            # A different selection: its health/queue/garrison are not changes.
            return [event(EventKind.SELECTION_CHANGED, before=before_type, after=after_type)]
        if before is None or after is None:
            return []
        found = []
        if (
            before.health is not None
            and after.health is not None
            and abs(after.health - before.health) >= self._min_health_delta
        ):
            found.append(
                event(
                    EventKind.SELECTION_HEALTH_CHANGED,
                    entity_type=after_type,
                    before=before.health,
                    after=after.health,
                )
            )
        if before.production_queue != after.production_queue:
            found.append(
                event(
                    EventKind.PRODUCTION_QUEUE_CHANGED,
                    entity_type=after_type,
                    before=",".join(before.production_queue),
                    after=",".join(after.production_queue),
                    length_before=len(before.production_queue),
                    length_after=len(after.production_queue),
                )
            )
        if (
            before.garrison is not None
            and after.garrison is not None
            and before.garrison != after.garrison
        ):
            found.append(
                event(
                    EventKind.GARRISON_CHANGED,
                    entity_type=after_type,
                    before=before.garrison,
                    after=after.garrison,
                )
            )
        return found

    def _remember(self, hud: HudState) -> None:
        self._stockpiles.update(hud.stockpiles)
        if hud.population is not None:
            self._population = hud.population
        if hud.phase is not Phase.UNKNOWN:
            self._phase = hud.phase
        self._selection = hud.selection
        self._seen_hud = True


class _Factory(Protocol):
    """Type of the per-frame ``event(kind, **details)`` helper (for annotations only)."""

    def __call__(self, kind: EventKind, **details: DetailValue) -> Event: ...
