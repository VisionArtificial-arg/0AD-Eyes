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
    """Perception thresholds and the NF3 accuracy targets, in one place (NF7).

    This is the single home for the NF3 targets (the ML8 ``EvalConfig`` derives from
    it via ``EvalConfig.from_thresholds``), so the file-driven config and the eval
    harness cannot drift apart.
    """

    model_config = ConfigDict(frozen=True)

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_health: float = Field(default=0.0, ge=0.0, le=1.0)
    hud_read_max_error: float = Field(default=0.01, ge=0.0, le=1.0)  # NF3 <1%
    detection_map_target: float = Field(default=0.80, ge=0.0, le=1.0)  # NF3
    ownership_accuracy_target: float = Field(default=0.98, ge=0.0, le=1.0)  # NF3
    tracking_mota_target: float = Field(default=0.70, ge=0.0, le=1.0)  # NF3
    eval_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # ML8 box-match IoU


# --------------------------------------------------------------------------- #
# Perception tuning — pure DATA (Approach B): the ownership palette lives here  #
# as codec-free value objects; infrastructure builds its cv2-backed            #
# ``PlayerPalette`` from this at the boundary (see perception/palette.py).      #
# --------------------------------------------------------------------------- #


class OwnershipHsvBand(BaseModel):
    """An inclusive HSV window (OpenCV ranges: H 0-179, S/V 0-255) — pure data."""

    model_config = ConfigDict(frozen=True)

    h_lo: int = Field(ge=0, le=179)
    h_hi: int = Field(ge=0, le=179)
    s_lo: int = Field(default=70, ge=0, le=255)
    s_hi: int = Field(default=255, ge=0, le=255)
    v_lo: int = Field(default=50, ge=0, le=255)
    v_hi: int = Field(default=255, ge=0, le=255)


class OwnershipColor(BaseModel):
    """A player colour mapped to an ownership relation and its HSV bands."""

    model_config = ConfigDict(frozen=True)

    name: str
    ownership: Ownership
    bands: tuple[OwnershipHsvBand, ...]


def _default_ownership_colors() -> tuple[OwnershipColor, ...]:
    """The default relative palette (SELF=blue, ALLY=green, ENEMY=red, GAIA=yellow).

    Values mirror the former infra ``DEFAULT_PALETTE`` exactly, so externalizing it
    changes no behaviour; the config is now the single source and infra derives it.
    """

    return (
        OwnershipColor(
            name="blue", ownership=Ownership.SELF, bands=(OwnershipHsvBand(h_lo=100, h_hi=130),)
        ),
        OwnershipColor(
            name="green", ownership=Ownership.ALLY, bands=(OwnershipHsvBand(h_lo=45, h_hi=85),)
        ),
        OwnershipColor(
            name="red",
            ownership=Ownership.ENEMY,
            bands=(OwnershipHsvBand(h_lo=0, h_hi=10), OwnershipHsvBand(h_lo=170, h_hi=179)),
        ),
        OwnershipColor(
            name="yellow", ownership=Ownership.GAIA, bands=(OwnershipHsvBand(h_lo=22, h_hi=34),)
        ),
    )


class OwnershipPalette(BaseModel):
    """The set of player colours in play, ordered by matching priority (E3)."""

    model_config = ConfigDict(frozen=True)

    colors: tuple[OwnershipColor, ...] = Field(default_factory=_default_ownership_colors)


class PerceptionSettings(BaseModel):
    """Classical perception tuning (E3 ownership), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    ownership_palette: OwnershipPalette = Field(default_factory=OwnershipPalette)
    ownership_min_fraction: float = Field(default=0.02, ge=0.0, le=1.0)


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
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
