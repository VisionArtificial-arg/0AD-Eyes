"""System event source (EVT-6): the eyes reporting on their own health.

``SystemEventSource`` emits ``origin=system`` events so a consumer knows when to trust
the rest of the stream less:

- **calibration:** ``layout_drift`` when the periodic self-check finds the HUD layout
  moved, and ``recalibrated`` whenever the layout is (re-)detected;
- **reads:** ``hud_read_failed`` / ``hud_read_restored`` (no HUD values read),
  ``hud_low_confidence`` (the read's confidence fell below ``events.system_low_confidence``),
  and ``minimap_read_failed`` / ``minimap_read_restored``;
- **capture:** ``frames_dropped`` when frame ids jump by more than the usual step (so a
  strided replay, which always steps by N, is not a drop);
- **detector:** ``detections_stopped`` after ``events.system_silent_frames`` frames with
  no observed entity, once there had been some; ``detections_resumed`` when they return.

Failures are reported when the state *changes*, not on every frame, so one broken OCR
read gives one event, not a stream.
"""

from __future__ import annotations

from typing import Protocol

from zero_ad_eyes.application.frames import PerceptionContext
from zero_ad_eyes.application.settings import EventSettings
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.events import DetailValue, Event, EventKind, EventOrigin
from zero_ad_eyes.domain.hud import HudState
from zero_ad_eyes.domain.world_model import WorldModel


class SystemEventSource:
    """``EventSource`` for the perception layer's own state."""

    def __init__(self, *, low_confidence: float, silent_frames: int) -> None:
        self._low_confidence = low_confidence
        self._silent_frames = silent_frames
        self._hud_ok: bool | None = None
        self._hud_low = False
        self._minimap_ok: bool | None = None
        self._last_frame_id: int | None = None
        self._usual_step: int | None = None
        self._ever_detected = False
        self._silent_for = 0
        self._silent = False

    @classmethod
    def from_settings(cls, settings: EventSettings) -> SystemEventSource:
        return cls(
            low_confidence=settings.system_low_confidence,
            silent_frames=settings.system_silent_frames,
        )

    def events(self, world_model: WorldModel, context: PerceptionContext) -> tuple[Event, ...]:
        frame_id = world_model.meta.frame_id

        def event(kind: EventKind, **details: DetailValue) -> Event:
            return Event(
                kind=kind,
                frame_id=frame_id,
                confidence=Confidence.certain(Provenance.CLASSICAL),
                origin=EventOrigin.SYSTEM,
                details=details,
            )

        found: list[Event] = []
        if context.layout_drift:
            found.append(event(EventKind.LAYOUT_DRIFT))
        if context.recalibrated:
            found.append(
                event(
                    EventKind.RECALIBRATED,
                    width=world_model.meta.width,
                    height=world_model.meta.height,
                )
            )
        found += self._hud_events(world_model.hud, event)
        found += self._minimap_events(world_model, event)
        found += self._frame_events(frame_id, event)
        found += self._detector_events(world_model, event)
        return tuple(found)

    def _hud_events(self, hud: HudState | None, event: _Factory) -> list[Event]:
        ok = hud is not None and (bool(hud.stockpiles) or hud.population is not None)
        previous, self._hud_ok = self._hud_ok, ok
        found = []
        if ok and previous is False:
            found.append(event(EventKind.HUD_READ_RESTORED))
        elif not ok and previous is not False:
            found.append(event(EventKind.HUD_READ_FAILED, hud_present=hud is not None))
        low = ok and hud is not None and hud.confidence.value < self._low_confidence
        if low and not self._hud_low:
            assert hud is not None
            found.append(event(EventKind.HUD_LOW_CONFIDENCE, confidence=hud.confidence.value))
        self._hud_low = low
        return found

    def _minimap_events(self, world_model: WorldModel, event: _Factory) -> list[Event]:
        ok = world_model.minimap is not None
        previous, self._minimap_ok = self._minimap_ok, ok
        if ok and previous is False:
            return [event(EventKind.MINIMAP_READ_RESTORED)]
        if not ok and previous is not False:
            return [event(EventKind.MINIMAP_READ_FAILED)]
        return []

    def _frame_events(self, frame_id: int, event: _Factory) -> list[Event]:
        last, self._last_frame_id = self._last_frame_id, frame_id
        if last is None or frame_id <= last:
            return []
        step = frame_id - last
        if self._usual_step is None or step < self._usual_step:
            self._usual_step = step
            return []
        if step > self._usual_step:
            missed = step // self._usual_step - 1
            if missed > 0:
                return [
                    event(
                        EventKind.FRAMES_DROPPED,
                        missed=missed,
                        step=step,
                        usual_step=self._usual_step,
                    )
                ]
        return []

    def _detector_events(self, world_model: WorldModel, event: _Factory) -> list[Event]:
        observed = sum(1 for e in world_model.entities if e.staleness == 0)
        if observed:
            self._ever_detected = True
            self._silent_for = 0
            if self._silent:
                self._silent = False
                return [event(EventKind.DETECTIONS_RESUMED, observed=observed)]
            return []
        if not self._ever_detected:
            return []
        self._silent_for += 1
        if not self._silent and self._silent_for >= self._silent_frames:
            self._silent = True
            return [event(EventKind.DETECTIONS_STOPPED, silent_frames=self._silent_for)]
        return []


class _Factory(Protocol):
    def __call__(self, kind: EventKind, **details: DetailValue) -> Event: ...
