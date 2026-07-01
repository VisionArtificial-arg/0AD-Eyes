"""H3 — publish cadence: per-frame vs on-change (REQUIREMENTS.md §4.7 / EPIC H3).

Cadence is *when* the decision layer is handed a world model, orthogonal to *where*
it goes. So it is modelled as a decorator sink: a ``WorldModelSink`` that wraps
another ``WorldModelSink`` and forwards (or withholds) each publish. Any real sink —
in-memory, JSONL, callback — composes with either cadence unchanged.

- :class:`PerFrameSink` — forward every frame (the default, highest-freshness,
  highest-bandwidth cadence). Explicit so the choice is visible at the wiring site.
- :class:`OnChangeSink` — forward only when the *decision-relevant payload* changed
  since the last forwarded model, collapsing steady scenes to zero traffic. Frame
  metadata (id/timestamp) is ignored in the comparison — it changes every frame by
  definition — while HUD, minimap, entities (with their staleness) and the schema
  version are compared by value (frozen pydantic → structural equality).

Latency budget note (H3): a sink's ``publish`` runs inside the per-frame loop, so it
must stay well under the NF1 ~66 ms frame budget. On-change trades a cheap value
comparison for skipped downstream work when nothing changed.
"""

from __future__ import annotations

from zero_ad_eyes.application.ports import WorldModelSink
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.hud import HudState
from zero_ad_eyes.domain.minimap import MinimapModel
from zero_ad_eyes.domain.world_model import WorldModel

_Payload = tuple[str, HudState | None, MinimapModel | None, tuple[Entity, ...]]


def _payload(world_model: WorldModel) -> _Payload:
    """The decision-relevant content, excluding volatile frame metadata."""

    return (
        world_model.schema_version,
        world_model.hud,
        world_model.minimap,
        world_model.entities,
    )


class PerFrameSink:
    """Forwards every published model to the wrapped sink (default cadence)."""

    def __init__(self, inner: WorldModelSink) -> None:
        self._inner = inner

    def publish(self, world_model: WorldModel) -> None:
        self._inner.publish(world_model)


class OnChangeSink:
    """Forwards a model only when its payload differs from the last forwarded one."""

    def __init__(self, inner: WorldModelSink) -> None:
        self._inner = inner
        self._last: _Payload | None = None
        self._suppressed = 0

    @property
    def suppressed(self) -> int:
        """How many publishes were withheld as unchanged (observability, NF6)."""

        return self._suppressed

    def publish(self, world_model: WorldModel) -> None:
        payload = _payload(world_model)
        if payload == self._last:
            self._suppressed += 1
            return
        self._last = payload
        self._inner.publish(world_model)
