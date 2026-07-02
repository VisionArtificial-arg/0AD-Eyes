"""Classical entity enrichment (EPIC E → G) behind the ``EntityEnricher`` port.

Reads each tracked entity's crop from the frame to fill the attributes the detector
and tracker leave absent:

- **ownership** (E3) via player-colour segmentation (:mod:`.ownership`);
- **health** (E4) via health-bar fill fraction (:mod:`.health`);
- **selected** (E5) via the selection-ring cue (:mod:`.state`).

Fill-if-absent, per the port contract: a field already set by a more authoritative
source (e.g. a learned model's ownership) is never overwritten — only ``UNKNOWN`` /
``None`` fields are filled, and ``selected`` is OR-ed with the pixel cue. Entities
with no screen box are returned untouched (there is nothing to read).

Everything here is ``Provenance.CLASSICAL`` and deliberately coarse; it fills gaps,
it does not adjudicate against confident learned values.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import (
    HealthReadSettings,
    PerceptionSettings,
    StateCueSettings,
)
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.taxonomy import Ownership

from .health import read_health
from .ownership import assign_ownership
from .palette import PlayerPalette
from .state import read_state_cues


class ClassicalEntityEnricher:
    """Fill ownership/health/selection onto tracked entities from pixels."""

    def __init__(
        self,
        *,
        ownership_palette: PlayerPalette,
        ownership_min_fraction: float,
        health: HealthReadSettings,
        state: StateCueSettings,
    ) -> None:
        self._ownership_palette = ownership_palette
        self._ownership_min_fraction = ownership_min_fraction
        self._health = health
        self._state = state

    @classmethod
    def from_settings(cls, settings: PerceptionSettings) -> ClassicalEntityEnricher:
        """Construct from pure config (Approach B): build the cv2 palette at the boundary."""

        return cls(
            ownership_palette=PlayerPalette.from_settings(settings.ownership_palette),
            ownership_min_fraction=settings.ownership_min_fraction,
            health=settings.health,
            state=settings.state,
        )

    def enrich(self, entities: Sequence[Entity], frame: Frame) -> tuple[Entity, ...]:
        return tuple(self._enrich_one(entity, frame) for entity in entities)

    def _enrich_one(self, entity: Entity, frame: Frame) -> Entity:
        bbox = entity.screen_bbox
        if bbox is None:
            return entity

        update: dict[str, Any] = {}

        if entity.ownership is Ownership.UNKNOWN:
            owner, _coverage = assign_ownership(
                frame, bbox, self._ownership_palette, self._ownership_min_fraction
            )
            if owner is not Ownership.UNKNOWN:
                update["ownership"] = owner

        if entity.health is None:
            health = read_health(
                frame,
                bbox,
                max_offset=self._health.max_offset,
                s_min=self._health.s_min,
                v_min=self._health.v_min,
                min_run=self._health.min_run,
            )
            if health is not None:
                update["health"] = health

        if not entity.selected and read_state_cues(frame, bbox, self._state).selected:
            update["selected"] = True

        return entity.model_copy(update=update) if update else entity
