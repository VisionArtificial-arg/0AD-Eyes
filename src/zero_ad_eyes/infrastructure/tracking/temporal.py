"""Temporal analysis to stabilise perception (REQUIREMENTS.md G7, CV-19).

A per-frame detector flickers: the same unit may be read as ``UNIT`` then briefly
``RESOURCE_NODE``, its confidence jittering frame to frame. ``TemporalStabilizer``
smooths that noise *after* tracking, keyed by the stable ``entity_id`` the tracker
already assigns:

- categorical fields (``kind``, ``entity_type``) are resolved by **majority vote**
  over a sliding window, ties broken toward the most-recent observation;
- ``confidence`` is a moving average over the window (provenance kept from the
  latest observation);
- geometry (bbox / world position / health / staleness) is passed through from the
  latest observation — stabilisation targets classification flicker, not lag.

It is a stateful post-processor, deliberately separate from the tracker so the raw
tracker output stays available and the two concerns do not entangle (SRP).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Sequence
from typing import TypeVar

from zero_ad_eyes.domain.confidence import Confidence
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.taxonomy import EntityKind

_T = TypeVar("_T", bound=Hashable)


def majority(values: Sequence[_T]) -> _T:
    """Most-frequent value; ties resolved toward the latest occurrence. Order-stable."""

    counts: dict[_T, int] = {}
    last_index: dict[_T, int] = {}
    for i, value in enumerate(values):
        counts[value] = counts.get(value, 0) + 1
        last_index[value] = i
    best = max(counts.items(), key=lambda kv: (kv[1], last_index[kv[0]]))
    return best[0]


class _History:
    """The bounded recent observations of one entity id."""

    def __init__(self, window: int) -> None:
        self.kinds: deque[EntityKind] = deque(maxlen=window)
        self.types: deque[str] = deque(maxlen=window)
        self.confidences: deque[float] = deque(maxlen=window)
        self.absent: int = 0

    def observe(self, entity: Entity) -> None:
        self.kinds.append(entity.kind)
        if entity.entity_type is not None:
            self.types.append(entity.entity_type)
        self.confidences.append(entity.confidence.value)
        self.absent = 0


class TemporalStabilizer:
    """Sliding-window smoother over the tracker's per-frame entity stream (G7)."""

    def __init__(self, *, window: int = 5) -> None:
        self._window = window
        self._histories: dict[int, _History] = {}

    def stabilize(self, entities: Sequence[Entity]) -> tuple[Entity, ...]:
        """Return the entities with categorical fields voted and confidence averaged."""

        present = {e.entity_id for e in entities}
        stabilized = tuple(self._stabilize_one(e) for e in entities)
        self._forget_absent(present)
        return stabilized

    def _stabilize_one(self, entity: Entity) -> Entity:
        history = self._histories.setdefault(entity.entity_id, _History(self._window))
        history.observe(entity)

        smoothed_value = sum(history.confidences) / len(history.confidences)
        entity_type = majority(list(history.types)) if history.types else entity.entity_type
        return entity.model_copy(
            update={
                "kind": majority(list(history.kinds)),
                "entity_type": entity_type,
                "confidence": Confidence(
                    value=smoothed_value, provenance=entity.confidence.provenance
                ),
            }
        )

    def _forget_absent(self, present: set[int]) -> None:
        for entity_id in list(self._histories):
            if entity_id in present:
                continue
            history = self._histories[entity_id]
            history.absent += 1
            if history.absent > self._window:
                del self._histories[entity_id]
