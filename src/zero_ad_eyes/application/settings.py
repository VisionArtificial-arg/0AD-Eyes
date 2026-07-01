"""Typed configuration policy (REQUIREMENTS.md X3 / NF7).

These are the *value objects* of the configuration system — thresholds, filesystem
paths, and rendering palettes — with sensible defaults and no magic numbers. They
are pure ``pydantic`` models: no JSON, no environment, no OpenCV, no disk. Loading
them from a file and overriding from the environment is *infrastructure* concern
and lives in ``infrastructure.config`` (the loader depends inward on these types).

They sit in the application ring so an interface component (e.g. the overlay) can
consume policy without reaching sideways into infrastructure. Colours are stored as
RGB triples (matching ``HudState.self_player_color``); the overlay converts them to
OpenCV's BGR order at draw time.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.minimap import FogState
from zero_ad_eyes.domain.taxonomy import Ownership

RGB = tuple[int, int, int]


def _default_owner_colors() -> dict[Ownership, RGB]:
    return {
        Ownership.SELF: (60, 180, 75),  # green
        Ownership.ALLY: (0, 130, 200),  # blue
        Ownership.ENEMY: (230, 25, 75),  # red
        Ownership.GAIA: (170, 170, 170),  # grey
        Ownership.UNKNOWN: (255, 255, 255),  # white
    }


def _default_fog_colors() -> dict[FogState, RGB]:
    return {
        FogState.UNEXPLORED: (20, 20, 20),  # near-black
        FogState.EXPLORED: (90, 90, 90),  # shroud grey
        FogState.VISIBLE: (60, 120, 60),  # lit green tint
    }


class OwnerPalette(BaseModel):
    """RGB colour per :class:`Ownership`, used to tint entities and blips (E3)."""

    model_config = ConfigDict(frozen=True)

    colors: dict[Ownership, RGB] = Field(default_factory=_default_owner_colors)

    def for_ownership(self, ownership: Ownership) -> RGB:
        return self.colors.get(ownership, self.colors[Ownership.UNKNOWN])


class FogPalette(BaseModel):
    """RGB colour per :class:`FogState`, used to tint the fog panel (§4.5)."""

    model_config = ConfigDict(frozen=True)

    colors: dict[FogState, RGB] = Field(default_factory=_default_fog_colors)

    def for_state(self, state: FogState) -> RGB:
        return self.colors.get(state, self.colors[FogState.UNEXPLORED])


class OverlaySettings(BaseModel):
    """Everything the debug overlay (X1) needs to draw, with no magic numbers."""

    model_config = ConfigDict(frozen=True)

    owner_palette: OwnerPalette = Field(default_factory=OwnerPalette)
    fog_palette: FogPalette = Field(default_factory=FogPalette)

    health_good: RGB = (60, 180, 75)
    health_warn: RGB = (255, 200, 0)
    health_bad: RGB = (230, 25, 75)
    health_good_min: float = Field(default=0.5, ge=0.0, le=1.0)
    health_warn_min: float = Field(default=0.25, ge=0.0, le=1.0)

    hud_text_color: RGB = (255, 255, 255)
    hud_panel_color: RGB = (0, 0, 0)
    font_scale: float = Field(default=0.4, gt=0.0)
    box_thickness: int = Field(default=1, ge=1)
    minimap_fraction: float = Field(default=0.25, gt=0.0, le=1.0)

    def health_color(self, fraction: float) -> RGB:
        if fraction >= self.health_good_min:
            return self.health_good
        if fraction >= self.health_warn_min:
            return self.health_warn
        return self.health_bad


class Thresholds(BaseModel):
    """Perception thresholds and the NF3 accuracy targets, in one place (NF7)."""

    model_config = ConfigDict(frozen=True)

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_health: float = Field(default=0.0, ge=0.0, le=1.0)
    hud_read_max_error: float = Field(default=0.01, ge=0.0, le=1.0)  # NF3 <1%
    detection_map_target: float = Field(default=0.80, ge=0.0, le=1.0)  # NF3
    ownership_accuracy_target: float = Field(default=0.98, ge=0.0, le=1.0)  # NF3


class Paths(BaseModel):
    """Filesystem locations for recordings and calibration profiles (X2/X3)."""

    model_config = ConfigDict(frozen=True)

    recordings_dir: Path = Path("recordings")
    calibration_dir: Path = Path("calibration")


class Config(BaseModel):
    """The single, typed configuration root (X3)."""

    model_config = ConfigDict(frozen=True)

    thresholds: Thresholds = Field(default_factory=Thresholds)
    paths: Paths = Field(default_factory=Paths)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)
