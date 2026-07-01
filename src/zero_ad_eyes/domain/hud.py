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


class HudState(BaseModel):
    """Parsed top-bar / self-identification state."""

    model_config = ConfigDict(frozen=True)

    stockpiles: dict[ResourceType, int] = Field(default_factory=dict)
    population: Population | None = None
    phase: Phase = Phase.UNKNOWN
    self_player_color: tuple[int, int, int] | None = None  # RGB, self-identification
    self_civ: str | None = None
    confidence: Confidence = Confidence.unknown()
