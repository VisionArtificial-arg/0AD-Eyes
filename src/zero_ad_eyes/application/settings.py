"""Typed configuration policy (REQUIREMENTS.md X3 / NF7).

These are the *value objects* of the configuration system — thresholds, filesystem
paths, palettes, per-subsystem tuning. They are pure ``pydantic`` models: no JSON, no
environment, no OpenCV, no disk.

**No defaults here (by design).** Every field is required and carries only its
validation constraints. The single source of default *values* is the
``default_config`` generator in the ``interface`` layer (a UI concern), surfaced to
users via the ``config`` CLI commands. A ``Config`` is therefore always built from a
complete, explicit set of values — the generated defaults, a user's file layered on
them, or an env override on top — never from silent in-code fallbacks.

They sit in the application ring so an interface component (e.g. the overlay) can
consume policy without reaching sideways into infrastructure. Colours are stored as
RGB triples; the overlay converts them to OpenCV's BGR order at draw time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.minimap import FogState
from zero_ad_eyes.domain.taxonomy import Ownership

RGB = tuple[int, int, int]


class OwnerPalette(BaseModel):
    """RGB colour per :class:`Ownership`, used to tint entities and blips (E3)."""

    model_config = ConfigDict(frozen=True)

    colors: dict[Ownership, RGB]

    def for_ownership(self, ownership: Ownership) -> RGB:
        return self.colors.get(ownership, self.colors[Ownership.UNKNOWN])


class FogPalette(BaseModel):
    """RGB colour per :class:`FogState`, used to tint the fog panel (§4.5)."""

    model_config = ConfigDict(frozen=True)

    colors: dict[FogState, RGB]

    def for_state(self, state: FogState) -> RGB:
        return self.colors.get(state, self.colors[FogState.UNEXPLORED])


class OverlaySettings(BaseModel):
    """Everything the debug overlay (X1) needs to draw."""

    model_config = ConfigDict(frozen=True)

    owner_palette: OwnerPalette
    fog_palette: FogPalette

    health_good: RGB
    health_warn: RGB
    health_bad: RGB
    health_good_min: float = Field(ge=0.0, le=1.0)
    health_warn_min: float = Field(ge=0.0, le=1.0)

    hud_text_color: RGB
    hud_panel_color: RGB
    font_scale: float = Field(gt=0.0)
    box_thickness: int = Field(ge=1)
    minimap_fraction: float = Field(gt=0.0, le=1.0)

    def health_color(self, fraction: float) -> RGB:
        if fraction >= self.health_good_min:
            return self.health_good
        if fraction >= self.health_warn_min:
            return self.health_warn
        return self.health_bad


class Thresholds(BaseModel):
    """Perception thresholds and the NF3 accuracy targets, in one place (NF7).

    The single home for the NF3 targets (the ML8 ``EvalConfig`` derives from it via
    ``EvalConfig.from_thresholds``), so the file-driven config and the eval harness
    cannot drift apart.
    """

    model_config = ConfigDict(frozen=True)

    min_confidence: float = Field(ge=0.0, le=1.0)
    min_health: float = Field(ge=0.0, le=1.0)
    hud_read_max_error: float = Field(ge=0.0, le=1.0)  # NF3 <1%
    detection_map_target: float = Field(ge=0.0, le=1.0)  # NF3
    ownership_accuracy_target: float = Field(ge=0.0, le=1.0)  # NF3
    tracking_mota_target: float = Field(ge=0.0, le=1.0)  # NF3
    eval_iou_threshold: float = Field(ge=0.0, le=1.0)  # ML8 box-match IoU


# --------------------------------------------------------------------------- #
# Perception tuning — pure DATA (Approach B): infrastructure builds its         #
# cv2-backed types from these at the boundary (see perception/*).               #
# --------------------------------------------------------------------------- #


class HsvWindow(BaseModel):
    """A generic inclusive HSV window (OpenCV ranges: H 0-179, S/V 0-255) — pure data.

    Shared by the ownership palette (E3) and the resource cues (E6a); infrastructure
    rehydrates it into its cv2-capable ``HsvBand`` at the boundary.
    """

    model_config = ConfigDict(frozen=True)

    h_lo: int = Field(ge=0, le=179)
    h_hi: int = Field(ge=0, le=179)
    s_lo: int = Field(ge=0, le=255)
    s_hi: int = Field(ge=0, le=255)
    v_lo: int = Field(ge=0, le=255)
    v_hi: int = Field(ge=0, le=255)


class OwnershipColor(BaseModel):
    """A player colour mapped to an ownership relation and its HSV bands."""

    model_config = ConfigDict(frozen=True)

    name: str
    ownership: Ownership
    bands: tuple[HsvWindow, ...]


class OwnershipPalette(BaseModel):
    """The set of player colours in play, ordered by matching priority (E3)."""

    model_config = ConfigDict(frozen=True)

    colors: tuple[OwnershipColor, ...]


class ResourceCueSetting(BaseModel):
    """A coarse colour signature for one class of natural resource node (E6a)."""

    model_config = ConfigDict(frozen=True)

    entity_type: str
    bands: tuple[HsvWindow, ...]
    min_area: int = Field(ge=0)


class HealthReadSettings(BaseModel):
    """Health-bar reading knobs (E4)."""

    model_config = ConfigDict(frozen=True)

    max_offset: int = Field(ge=0)  # search band height above the entity
    s_min: int = Field(ge=0, le=255)  # HSV saturation floor
    v_min: int = Field(ge=0, le=255)  # HSV value floor
    min_run: float = Field(ge=0.0, le=1.0)  # min bright-run width fraction


class SelectionCueSettings(BaseModel):
    """Selection-ring cue knobs (E5)."""

    model_config = ConfigDict(frozen=True)

    thickness: int = Field(ge=1)
    brightness: int = Field(ge=0, le=255)
    min_fraction: float = Field(ge=0.0, le=1.0)


class ConstructionCueSettings(BaseModel):
    """Construction-scaffold cue knobs (E5)."""

    model_config = ConfigDict(frozen=True)

    edge_density_min: float = Field(ge=0.0, le=1.0)
    canny_lo: float
    canny_hi: float


class GarrisonCueSettings(BaseModel):
    """Garrison-badge cue knobs (E5)."""

    model_config = ConfigDict(frozen=True)

    top_fraction: float = Field(ge=0.0, le=1.0)
    brightness: int = Field(ge=0, le=255)
    max_saturation: int = Field(ge=0, le=255)
    min_badge_area: int = Field(ge=0)
    max_badge_width_fraction: float = Field(ge=0.0, le=1.0)


class StateCueSettings(BaseModel):
    """The three best-effort entity-state cues (E5)."""

    model_config = ConfigDict(frozen=True)

    selection: SelectionCueSettings
    construction: ConstructionCueSettings
    garrison: GarrisonCueSettings


class PerceptionSettings(BaseModel):
    """Classical perception tuning (E3/E4/E5 + E6a resources), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    ownership_palette: OwnershipPalette
    ownership_min_fraction: float = Field(ge=0.0, le=1.0)
    detect_resources: bool
    resource_cues: tuple[ResourceCueSetting, ...]
    health: HealthReadSettings
    state: StateCueSettings


# --------------------------------------------------------------------------- #
# Minimap tuning — pure DATA (Approach B).                                       #
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


class WorldExtentSettings(BaseModel):
    """The world rectangle the minimap covers, in engine units (D6)."""

    model_config = ConfigDict(frozen=True)

    origin_x: float
    origin_y: float
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)
    flip_y: bool


class FogSettings(BaseModel):
    """Fog-of-war cell classification knobs (D4)."""

    model_config = ConfigDict(frozen=True)

    rows: int = Field(gt=0)
    cols: int = Field(gt=0)
    unexplored_max: float  # mean brightness below ⇒ never seen
    visible_min: float  # mean brightness at/above ⇒ currently visible


class BlipSettings(BaseModel):
    """Blip detection knobs (D2)."""

    model_config = ConfigDict(frozen=True)

    tolerance: float  # nearest-palette-colour distance tolerance
    min_area: int = Field(ge=0)
    max_area: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class TerritorySettings(BaseModel):
    """Territory region extraction knobs (D3)."""

    model_config = ConfigDict(frozen=True)

    tolerance: float
    min_area: int = Field(ge=0)


class ViewportSettings(BaseModel):
    """Camera-viewport quad extraction knobs (D5)."""

    model_config = ConfigDict(frozen=True)

    white_min: int = Field(ge=0, le=255)
    min_area: int = Field(ge=0)
    min_side: int = Field(ge=0)
    approx_epsilon_fraction: float = Field(gt=0.0, le=1.0)  # approxPolyDP tol (frac of perimeter)


class MinimapSettings(BaseModel):
    """Classical minimap-reader tuning (EPIC D), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    palette: tuple[MinimapPaletteEntry, ...]
    world_extent: WorldExtentSettings
    fog: FogSettings
    blips: BlipSettings
    territory: TerritorySettings
    viewport: ViewportSettings
    disc_shape: bool  # False ⇒ SQUARE active area (D1); True ⇒ DISC
    region_confidence: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# HUD tuning — pure DATA (Approach B). Field names mirror the infra layout types #
# so ClassicalHudReader.from_settings maps them by model_validate.               #
# --------------------------------------------------------------------------- #


class FractionalRegionSetting(BaseModel):
    """A box as fractions [0, 1] of a parent HUD region (mirrors FractionalRegion)."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class TopBarLayoutSettings(BaseModel):
    """Where each top-bar element sits, as fractions of the top_bar box (C1–C4)."""

    model_config = ConfigDict(frozen=True)

    food: FractionalRegionSetting
    wood: FractionalRegionSetting
    stone: FractionalRegionSetting
    metal: FractionalRegionSetting
    population: FractionalRegionSetting
    phase: FractionalRegionSetting
    swatch: FractionalRegionSetting
    civ: FractionalRegionSetting


class SelectionPanelLayoutSettings(BaseModel):
    """Sub-regions of the bottom-centre selection panel (C5)."""

    model_config = ConfigDict(frozen=True)

    entity_type: FractionalRegionSetting
    health: FractionalRegionSetting
    queue: FractionalRegionSetting


class HudSettings(BaseModel):
    """Classical HUD-reader tuning (EPIC C), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    top_bar: TopBarLayoutSettings
    selection: SelectionPanelLayoutSettings
    ocr_config: str  # Tesseract page-segmentation mode


class TrackingSettings(BaseModel):
    """Classical tracking + event-detection tuning (EPIC G), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    # IoU tracker (G1–G3)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    min_hits: int = Field(ge=0)  # hits before a track is CONFIRMED
    max_staleness: int = Field(ge=0)  # memory budget before DEAD (G3)
    decay: float = Field(ge=0.0, le=1.0)  # per-miss confidence decay
    # Event detection (G8)
    combat_drop: float = Field(ge=0.0, le=1.0)  # health drop ⇒ combat event
    depletion_health: float = Field(ge=0.0, le=1.0)  # health ≤ this ⇒ depleted


# --------------------------------------------------------------------------- #
# Preprocessing tuning — pure DATA (Approach B): parameters of the HUD / scene   #
# chains (EPIC P); the variant factories build the cv2 steps from these.         #
# --------------------------------------------------------------------------- #


class HudPipelineSettings(BaseModel):
    """Parameters of the HUD-tuned preprocessing chain (P1/P4/P5)."""

    model_config = ConfigDict(frozen=True)

    gaussian_ksize: int = Field(gt=0)
    clahe_clip_limit: float = Field(gt=0.0)
    clahe_tile: tuple[int, int]


class ScenePipelineSettings(BaseModel):
    """Parameters of the scene-tuned preprocessing chain (P1/P4/P5)."""

    model_config = ConfigDict(frozen=True)

    bilateral_diameter: int = Field(gt=0)
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    clahe_clip_limit: float = Field(gt=0.0)
    clahe_tile: tuple[int, int]


class PreprocessingSettings(BaseModel):
    """Classical preprocessing tuning (EPIC P), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    hud: HudPipelineSettings
    scene: ScenePipelineSettings


# --------------------------------------------------------------------------- #
# Calibration tuning — pure DATA (Approach B). ``ratios`` field names mirror     #
# HudLayoutRatios so the boundary mapping is a model_validate.                   #
# --------------------------------------------------------------------------- #


class HudLayoutRatiosSettings(BaseModel):
    """Resolution-relative HUD region fractions (B2); mirrors HudLayoutRatios."""

    model_config = ConfigDict(frozen=True)

    top_bar_height: float = Field(gt=0.0, le=1.0)
    minimap_side: float = Field(gt=0.0, le=1.0)
    selection_width: float = Field(gt=0.0, le=1.0)
    selection_height: float = Field(gt=0.0, le=1.0)


class CalibrationSettings(BaseModel):
    """Classical HUD-calibration tuning (EPIC B), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    ratios: HudLayoutRatiosSettings
    theme: str
    use_anchors: bool
    default_ui_scale: float = Field(gt=0.0)
    ui_scale_min: float = Field(gt=0.0)  # UI-scale clamp lower bound (B1)
    ui_scale_max: float = Field(gt=0.0)  # UI-scale clamp upper bound (B1)
    selfcheck_match_threshold: float = Field(ge=0.0, le=1.0)  # B4
    selfcheck_use_anchors: bool
    persist_profiles: (
        bool  # B3 — reuse/persist calibration to paths.calibration_dir across sessions
    )


class PerfSettings(BaseModel):
    """Performance gate targets (NF1/NF2), config-driven (NF7)."""

    model_config = ConfigDict(frozen=True)

    latency_target_ms: float = Field(gt=0.0)  # NF1 budget
    throughput_target_fps: float = Field(gt=0.0)  # NF2 floor


class PipelineSettings(BaseModel):
    """Runtime orchestration tuning for the perception pipeline."""

    model_config = ConfigDict(frozen=True)

    recalibrate_interval: int = Field(ge=1)  # frames between B4 self-checks


class AcquisitionSettings(BaseModel):
    """Acquisition tuning (EPIC A) — offline replay + live capture."""

    model_config = ConfigDict(frozen=True)

    offline_fps: float = Field(gt=0.0)  # image-folder replay pacing
    image_extensions: tuple[str, ...]
    live_monitor: int = Field(ge=0)  # mss monitor index (0 = the all-monitors virtual)
    live_fps: float = Field(gt=0.0)  # live-capture target pacing
    record_fourcc: str = Field(min_length=1)  # cv2 FOURCC for --record (e.g. "FFV1" lossless)
    record_container: str = Field(min_length=1)  # recording file suffix (e.g. ".mkv")
    # X11/mss, grim, portal+PipeWire, or Windows named-window
    capture_backend: Literal["mss", "wayland", "portal", "window"]
    # substring of the target window's title, used when capture_backend="window" (Windows:
    # locate the window each grab and scrape its client rect with mss; needs pywin32).
    window_title: str = Field(min_length=1)
    # command that writes one encoded image to stdout, used when capture_backend="wayland"
    # (e.g. grim on wlroots/Hyprland: ("grim", "-")); ignored under the mss backend.
    wayland_capture_command: tuple[str, ...] = Field(min_length=1)
    # portal+PipeWire backend (capture_backend="portal"): captures a specific window/
    # screen via xdg-desktop-portal, driven by a helper under the system Python.
    portal_source_type: Literal["window", "monitor"]
    portal_cursor: Literal["hidden", "embedded"]
    portal_helper_python: str = Field(min_length=1)  # interpreter with PyGObject+GStreamer
    portal_restore_token_file: str  # path to persist the portal restore token ("" = none)


class GeometrySettings(BaseModel):
    """World-projection / fusion tuning (EPIC F/G4/G5), config-driven (NF7).

    The home for the geometry/fusion knobs. Its values reach the geometry helpers
    (``CameraProjector``, ``reconcile``) and the tracker fusion (``fuse_entities``)
    as required parameters — there are no in-code defaults; a composition root that
    uses those helpers passes ``cfg.geometry.*``. No offline pipeline stage consumes
    them yet, but the values live here (and in the generator), not in the code.
    """

    model_config = ConfigDict(frozen=True)

    camera_error_tolerance: float = Field(gt=0.0)  # F4 1/e error scale
    fusion_agreement_scale: float = Field(gt=0.0)  # G5 1/e distance discount
    fusion_match_radius: float = Field(gt=0.0)  # G4 hint-association radius (screen/world units)


class Paths(BaseModel):
    """Filesystem locations for recordings and calibration profiles (X2/X3)."""

    model_config = ConfigDict(frozen=True)

    recordings_dir: Path
    calibration_dir: Path


class SegmentationModelSettings(BaseModel):
    """Learned U-Net segmentation adapter tuning (MP4), config-driven (NF7).

    ``input_height``/``input_width`` and the [0,1] scaling must match the training
    preprocessing (frame → RGB → resize (W×H) → float32/255, no ImageNet mean/std).
    ``score_threshold`` and ``min_region_area`` are the adapter's seg→entity denoise
    knobs (the model itself has no confidence threshold — argmax always wins)."""

    model_config = ConfigDict(frozen=True)

    weights_path: str
    input_height: int = Field(gt=0)
    input_width: int = Field(gt=0)
    score_threshold: float = Field(ge=0.0, le=1.0)
    min_region_area: int = Field(ge=0)


class Config(BaseModel):
    """The single, typed configuration root (X3). Built only from explicit values —
    see ``interface.default_config.default_config`` for the canonical default set."""

    model_config = ConfigDict(frozen=True)

    thresholds: Thresholds
    paths: Paths
    overlay: OverlaySettings
    perception: PerceptionSettings
    minimap: MinimapSettings
    hud: HudSettings
    tracking: TrackingSettings
    preprocessing: PreprocessingSettings
    calibration: CalibrationSettings
    perf: PerfSettings
    pipeline: PipelineSettings
    acquisition: AcquisitionSettings
    geometry: GeometrySettings
    segmentation: SegmentationModelSettings
