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


class HsvWindow(BaseModel):
    """A generic inclusive HSV window (OpenCV ranges: H 0-179, S/V 0-255).

    Same shape as :class:`OwnershipHsvBand`; kept distinct for now, to be unified
    into a single HSV-window type in the P4 dedup pass.
    """

    model_config = ConfigDict(frozen=True)

    h_lo: int = Field(ge=0, le=179)
    h_hi: int = Field(ge=0, le=179)
    s_lo: int = Field(default=70, ge=0, le=255)
    s_hi: int = Field(default=255, ge=0, le=255)
    v_lo: int = Field(default=50, ge=0, le=255)
    v_hi: int = Field(default=255, ge=0, le=255)


class ResourceCueSetting(BaseModel):
    """A coarse colour signature for one class of natural resource node (E6a)."""

    model_config = ConfigDict(frozen=True)

    entity_type: str
    bands: tuple[HsvWindow, ...]
    min_area: int = Field(default=20, ge=0)


def _default_resource_cues() -> tuple[ResourceCueSetting, ...]:
    """Coarse, overlap-tolerant signatures (recall over precision).

    Mirrors the former infra ``DEFAULT_RESOURCE_CUES`` exactly.
    """

    return (
        ResourceCueSetting(
            entity_type="tree", bands=(HsvWindow(h_lo=35, h_hi=85, s_lo=40, v_lo=30),)
        ),
        ResourceCueSetting(
            entity_type="mine",
            bands=(HsvWindow(h_lo=0, h_hi=179, s_lo=0, s_hi=50, v_lo=50, v_hi=190),),
        ),
        ResourceCueSetting(
            entity_type="bush",
            bands=(
                HsvWindow(h_lo=0, h_hi=8, s_lo=90, v_lo=60),
                HsvWindow(h_lo=168, h_hi=179, s_lo=90, v_lo=60),
            ),
        ),
        ResourceCueSetting(
            entity_type="fauna",
            bands=(HsvWindow(h_lo=9, h_hi=25, s_lo=60, s_hi=200, v_lo=40, v_hi=190),),
        ),
    )


class HealthReadSettings(BaseModel):
    """Health-bar reading knobs (E4)."""

    model_config = ConfigDict(frozen=True)

    max_offset: int = Field(default=20, ge=0)  # search band height above the entity
    s_min: int = Field(default=60, ge=0, le=255)  # HSV saturation floor
    v_min: int = Field(default=60, ge=0, le=255)  # HSV value floor
    min_run: float = Field(default=0.15, ge=0.0, le=1.0)  # min bright-run width fraction


class SelectionCueSettings(BaseModel):
    """Selection-ring cue knobs (E5)."""

    model_config = ConfigDict(frozen=True)

    thickness: int = Field(default=3, ge=1)
    brightness: int = Field(default=200, ge=0, le=255)
    min_fraction: float = Field(default=0.4, ge=0.0, le=1.0)


class ConstructionCueSettings(BaseModel):
    """Construction-scaffold cue knobs (E5)."""

    model_config = ConfigDict(frozen=True)

    edge_density_min: float = Field(default=0.12, ge=0.0, le=1.0)
    canny_lo: float = 60.0
    canny_hi: float = 180.0


class GarrisonCueSettings(BaseModel):
    """Garrison-badge cue knobs (E5)."""

    model_config = ConfigDict(frozen=True)

    top_fraction: float = Field(default=0.35, ge=0.0, le=1.0)
    brightness: int = Field(default=200, ge=0, le=255)
    max_saturation: int = Field(default=70, ge=0, le=255)
    min_badge_area: int = Field(default=6, ge=0)
    max_badge_width_fraction: float = Field(default=0.5, ge=0.0, le=1.0)


class StateCueSettings(BaseModel):
    """The three best-effort entity-state cues (E5)."""

    model_config = ConfigDict(frozen=True)

    selection: SelectionCueSettings = Field(default_factory=SelectionCueSettings)
    construction: ConstructionCueSettings = Field(default_factory=ConstructionCueSettings)
    garrison: GarrisonCueSettings = Field(default_factory=GarrisonCueSettings)


class PerceptionSettings(BaseModel):
    """Classical perception tuning (E3/E4/E5 + E6a resources), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    ownership_palette: OwnershipPalette = Field(default_factory=OwnershipPalette)
    ownership_min_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    detect_resources: bool = True
    resource_cues: tuple[ResourceCueSetting, ...] = Field(default_factory=_default_resource_cues)
    health: HealthReadSettings = Field(default_factory=HealthReadSettings)
    state: StateCueSettings = Field(default_factory=StateCueSettings)


# --------------------------------------------------------------------------- #
# Minimap tuning — pure DATA (Approach B): the minimap reader's collaborators   #
# (segmenter, blips D2, territory D3, fog D4, viewport D5, projector D6) read    #
# their knobs from here; infra builds the cv2 components at the boundary via     #
# ClassicalMinimapReader.from_settings (see minimap/reader.py).                  #
# --------------------------------------------------------------------------- #


class BgrColorSetting(BaseModel):
    """A colour in OpenCV BGR byte order (pure data)."""

    model_config = ConfigDict(frozen=True)

    b: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    r: int = Field(ge=0, le=255)


class MinimapPaletteEntry(BaseModel):
    """One calibrated minimap player colour and the ownership it denotes (D2)."""

    model_config = ConfigDict(frozen=True)

    label: str
    color: BgrColorSetting
    ownership: Ownership


def _default_minimap_palette() -> tuple[MinimapPaletteEntry, ...]:
    """Illustrative blue-self / green-ally / red-enemy / white-gaia scheme.

    Mirrors the former infra ``MinimapPalette.default()`` exactly (BGR order).
    """

    return (
        MinimapPaletteEntry(
            label="self", color=BgrColorSetting(b=235, g=90, r=40), ownership=Ownership.SELF
        ),
        MinimapPaletteEntry(
            label="ally", color=BgrColorSetting(b=60, g=200, r=60), ownership=Ownership.ALLY
        ),
        MinimapPaletteEntry(
            label="enemy", color=BgrColorSetting(b=40, g=40, r=220), ownership=Ownership.ENEMY
        ),
        MinimapPaletteEntry(
            label="gaia", color=BgrColorSetting(b=235, g=235, r=235), ownership=Ownership.GAIA
        ),
    )


class WorldExtentSettings(BaseModel):
    """The world rectangle the minimap covers, in engine units (D6)."""

    model_config = ConfigDict(frozen=True)

    origin_x: float = 0.0
    origin_y: float = 0.0
    width: float = Field(default=1024.0, gt=0.0)
    height: float = Field(default=1024.0, gt=0.0)
    flip_y: bool = True


class FogSettings(BaseModel):
    """Fog-of-war cell classification knobs (D4)."""

    model_config = ConfigDict(frozen=True)

    rows: int = Field(default=16, gt=0)
    cols: int = Field(default=16, gt=0)
    unexplored_max: float = 25.0  # mean brightness below ⇒ never seen
    visible_min: float = 140.0  # mean brightness at/above ⇒ currently visible


class BlipSettings(BaseModel):
    """Blip detection knobs (D2)."""

    model_config = ConfigDict(frozen=True)

    tolerance: float = 70.0  # nearest-palette-colour distance tolerance
    min_area: int = Field(default=1, ge=0)
    max_area: int = Field(default=60, ge=0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class TerritorySettings(BaseModel):
    """Territory region extraction knobs (D3)."""

    model_config = ConfigDict(frozen=True)

    tolerance: float = 90.0
    min_area: int = Field(default=64, ge=0)


class ViewportSettings(BaseModel):
    """Camera-viewport rectangle extraction knobs (D5)."""

    model_config = ConfigDict(frozen=True)

    white_min: int = Field(default=200, ge=0, le=255)
    min_area: int = Field(default=64, ge=0)
    min_side: int = Field(default=8, ge=0)


class MinimapSettings(BaseModel):
    """Classical minimap-reader tuning (EPIC D), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    palette: tuple[MinimapPaletteEntry, ...] = Field(default_factory=_default_minimap_palette)
    world_extent: WorldExtentSettings = Field(default_factory=WorldExtentSettings)
    fog: FogSettings = Field(default_factory=FogSettings)
    blips: BlipSettings = Field(default_factory=BlipSettings)
    territory: TerritorySettings = Field(default_factory=TerritorySettings)
    viewport: ViewportSettings = Field(default_factory=ViewportSettings)
    disc_shape: bool = False  # False ⇒ SQUARE active area (D1 default); True ⇒ DISC
    region_confidence: float = Field(default=0.9, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# HUD tuning — pure DATA (Approach B): the top-bar / selection-panel sub-region  #
# fractions (EPIC C) and the OCR mode. Field names mirror the infra layout types #
# so ClassicalHudReader.from_settings maps them by model_validate at the boundary.#
# --------------------------------------------------------------------------- #


class FractionalRegionSetting(BaseModel):
    """A box as fractions [0, 1] of a parent HUD region (mirrors FractionalRegion)."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


def _hud_slot(x: float, width: float) -> FractionalRegionSetting:
    return FractionalRegionSetting(x=x, y=0.0, width=width, height=1.0)


class TopBarLayoutSettings(BaseModel):
    """Where each top-bar element sits, as fractions of the top_bar box (C1–C4)."""

    model_config = ConfigDict(frozen=True)

    food: FractionalRegionSetting = _hud_slot(0.03, 0.10)
    wood: FractionalRegionSetting = _hud_slot(0.15, 0.10)
    stone: FractionalRegionSetting = _hud_slot(0.27, 0.10)
    metal: FractionalRegionSetting = _hud_slot(0.39, 0.10)
    population: FractionalRegionSetting = _hud_slot(0.51, 0.12)
    phase: FractionalRegionSetting = _hud_slot(0.80, 0.18)
    swatch: FractionalRegionSetting = FractionalRegionSetting(x=0.0, y=0.2, width=0.025, height=0.6)
    civ: FractionalRegionSetting = _hud_slot(0.66, 0.13)


class SelectionPanelLayoutSettings(BaseModel):
    """Sub-regions of the bottom-centre selection panel (C5)."""

    model_config = ConfigDict(frozen=True)

    entity_type: FractionalRegionSetting = FractionalRegionSetting(
        x=0.05, y=0.05, width=0.9, height=0.25
    )
    health: FractionalRegionSetting = FractionalRegionSetting(
        x=0.05, y=0.35, width=0.5, height=0.25
    )
    queue: FractionalRegionSetting = FractionalRegionSetting(x=0.05, y=0.7, width=0.9, height=0.28)


class HudSettings(BaseModel):
    """Classical HUD-reader tuning (EPIC C), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    top_bar: TopBarLayoutSettings = Field(default_factory=TopBarLayoutSettings)
    selection: SelectionPanelLayoutSettings = Field(default_factory=SelectionPanelLayoutSettings)
    ocr_config: str = "--psm 7"  # Tesseract page-segmentation mode (single line)


class TrackingSettings(BaseModel):
    """Classical tracking + event-detection tuning (EPIC G), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    # IoU tracker (G1–G3)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    min_hits: int = Field(default=1, ge=0)  # hits before a track is CONFIRMED
    max_staleness: int = Field(default=15, ge=0)  # memory budget before DEAD (G3)
    decay: float = Field(default=0.85, ge=0.0, le=1.0)  # per-miss confidence decay
    # Event detection (G8)
    combat_drop: float = Field(default=0.05, ge=0.0, le=1.0)  # health drop ⇒ combat event
    depletion_health: float = Field(default=0.02, ge=0.0, le=1.0)  # health ≤ this ⇒ depleted


# --------------------------------------------------------------------------- #
# Preprocessing tuning — pure DATA (Approach B): parameters of the pre-tuned    #
# HUD / scene chains (EPIC P). The variant factories build the cv2 steps from    #
# these; defaults reproduce the former baked values exactly.                     #
# --------------------------------------------------------------------------- #


class HudPipelineSettings(BaseModel):
    """Parameters of the HUD-tuned preprocessing chain (P1/P4/P5)."""

    model_config = ConfigDict(frozen=True)

    gaussian_ksize: int = Field(default=3, gt=0)
    clahe_clip_limit: float = Field(default=2.0, gt=0.0)
    clahe_tile: tuple[int, int] = (8, 8)


class ScenePipelineSettings(BaseModel):
    """Parameters of the scene-tuned preprocessing chain (P1/P4/P5)."""

    model_config = ConfigDict(frozen=True)

    bilateral_diameter: int = Field(default=5, gt=0)
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    clahe_clip_limit: float = Field(default=3.0, gt=0.0)
    clahe_tile: tuple[int, int] = (8, 8)


class PreprocessingSettings(BaseModel):
    """Classical preprocessing tuning (EPIC P), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    hud: HudPipelineSettings = Field(default_factory=HudPipelineSettings)
    scene: ScenePipelineSettings = Field(default_factory=ScenePipelineSettings)


# --------------------------------------------------------------------------- #
# Calibration tuning — pure DATA (Approach B): HUD region ratios + UI-scale     #
# bounds + self-check thresholds (EPIC B). ``ratios`` field names mirror         #
# HudLayoutRatios so the boundary mapping is a model_validate.                   #
# --------------------------------------------------------------------------- #


class HudLayoutRatiosSettings(BaseModel):
    """Resolution-relative HUD region fractions (B2); mirrors HudLayoutRatios."""

    model_config = ConfigDict(frozen=True)

    top_bar_height: float = Field(default=0.035, gt=0.0, le=1.0)
    minimap_side: float = Field(default=0.20, gt=0.0, le=1.0)
    selection_width: float = Field(default=0.34, gt=0.0, le=1.0)
    selection_height: float = Field(default=0.16, gt=0.0, le=1.0)


class CalibrationSettings(BaseModel):
    """Classical HUD-calibration tuning (EPIC B), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    ratios: HudLayoutRatiosSettings = Field(default_factory=HudLayoutRatiosSettings)
    theme: str = "default"
    use_anchors: bool = True
    default_ui_scale: float = Field(default=1.0, gt=0.0)
    ui_scale_min: float = Field(default=0.5, gt=0.0)  # UI-scale clamp lower bound (B1)
    ui_scale_max: float = Field(default=3.0, gt=0.0)  # UI-scale clamp upper bound (B1)
    selfcheck_match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # B4
    selfcheck_use_anchors: bool = True


class PerfSettings(BaseModel):
    """Performance gate targets (NF1/NF2), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    latency_target_ms: float = Field(default=66.0, gt=0.0)  # NF1 budget
    throughput_target_fps: float = Field(default=15.0, gt=0.0)  # NF2 floor


class PipelineSettings(BaseModel):
    """Runtime orchestration tuning for the perception pipeline."""

    model_config = ConfigDict(frozen=True)

    recalibrate_interval: int = Field(default=30, ge=1)  # frames between B4 self-checks


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
    minimap: MinimapSettings = Field(default_factory=MinimapSettings)
    hud: HudSettings = Field(default_factory=HudSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    perf: PerfSettings = Field(default_factory=PerfSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
