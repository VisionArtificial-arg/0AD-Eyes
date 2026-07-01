"""HUD state value objects (REQUIREMENTS.md §4.2).

Own-POV only (A6): these describe the *playing* player's HUD. Enemy resources and
population are unobservable by construction and are never represented here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .confidence import Confidence
from .taxonomy import Phase, ResourceType


class Population(BaseModel):
    model_config = ConfigDict(frozen=True)

    current: int = Field(ge=0)
    cap: int = Field(ge=0)


class SelectionState(BaseModel):
    """The currently-selected unit/building panel (v0.2, §4.2 / EPIC E).

    Own-POV: this is whatever the *self* player has selected. All fields optional
    because the panel is only populated when something is selected and each datum
    (health bar, production queue, garrison count) is read independently.
    """

    model_config = ConfigDict(frozen=True)

    entity_type: str | None = None
    health: float | None = Field(default=None, ge=0.0, le=1.0)
    production_queue: tuple[str, ...] = ()
    garrison: int | None = Field(default=None, ge=0)
    confidence: Confidence = Confidence.unknown()


class HudState(BaseModel):
    """Parsed top-bar / self-identification state."""

    model_config = ConfigDict(frozen=True)

    stockpiles: dict[ResourceType, int] = Field(default_factory=dict)
    population: Population | None = None
    phase: Phase = Phase.UNKNOWN
    self_player_color: tuple[int, int, int] | None = None  # RGB, self-identification
    self_civ: str | None = None
    selection: SelectionState | None = None
    confidence: Confidence = Confidence.unknown()
