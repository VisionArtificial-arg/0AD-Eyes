"""Selection-panel reading (EPIC C — C5, ``[S]`` best-effort).

The bottom-centre selection panel describes the *currently selected* entity: its
type, health, and production queue. This is a should-have, best-effort signal.

Boundary note: the shared ``HudState`` domain value object (owned elsewhere) has
no field for the selection, and this module may not edit it. So the selection is
exposed as a package-local value object, :class:`SelectionState`, returned by
``ClassicalHudReader.read_selection`` — a method *alongside* the ``HudReader``
port's ``read`` (which stays HudState-typed), not through it. Consumers that want
selection data call the concrete reader; the port contract is unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.confidence import Confidence


class HealthReading(BaseModel):
    """A selected entity's hit points and derived fraction."""

    model_config = ConfigDict(frozen=True)

    current: int = Field(ge=0)
    maximum: int = Field(gt=0)

    @property
    def fraction(self) -> float:
        return self.current / self.maximum


class SelectionState(BaseModel):
    """Best-effort reading of the selection panel (C5, ``[S]``)."""

    model_config = ConfigDict(frozen=True)

    entity_type: str | None = None
    health: HealthReading | None = None
    production_queue: tuple[str, ...] = ()
    confidence: Confidence = Confidence.unknown()
