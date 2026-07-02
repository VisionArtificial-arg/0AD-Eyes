"""The default-config generator — the SINGLE source of default values (UI concern).

The typed models in ``application.settings`` carry no defaults; every value the system
would otherwise bury in code lives here, in one explicit ``default_config()``. This is
a UI/interface concern: the ``config`` CLI commands surface it (write / show / validate)
so a user discovers and edits the defaults as data, never by reading source.

Anything that needs "the defaults" — a no-``--config`` run, the base layer under a user's
partial file, the committed ``docs/config.example.json`` — comes from here.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import (
    AcquisitionSettings,
    BgrColorSetting,
    BlipSettings,
    CalibrationSettings,
    Config,
    ConstructionCueSettings,
    FogPalette,
    FogSettings,
    FractionalRegionSetting,
    GarrisonCueSettings,
    GeometrySettings,
    HealthReadSettings,
    HsvWindow,
    HudLayoutRatiosSettings,
    HudPipelineSettings,
    HudSettings,
    MinimapPaletteEntry,
    MinimapSettings,
    OverlaySettings,
    OwnerPalette,
    OwnershipColor,
    OwnershipPalette,
    Paths,
    PerceptionSettings,
    PerfSettings,
    PipelineSettings,
    PreprocessingSettings,
    ResourceCueSetting,
    ScenePipelineSettings,
    SelectionCueSettings,
    SelectionPanelLayoutSettings,
    StateCueSettings,
    TerritorySettings,
    Thresholds,
    TopBarLayoutSettings,
    TrackingSettings,
    ViewportSettings,
    WorldExtentSettings,
)
from zero_ad_eyes.domain.minimap import FogState
from zero_ad_eyes.domain.taxonomy import Ownership


def _slot(x: float, width: float) -> FractionalRegionSetting:
    """A full-height HUD text slot at fractional x with fractional width."""

    return FractionalRegionSetting(x=x, y=0.0, width=width, height=1.0)


def _default_thresholds() -> Thresholds:
    return Thresholds(
        min_confidence=0.5,
        min_health=0.0,
        hud_read_max_error=0.01,
        detection_map_target=0.80,
        ownership_accuracy_target=0.98,
        tracking_mota_target=0.70,
        eval_iou_threshold=0.5,
    )


def _default_overlay() -> OverlaySettings:
    return OverlaySettings(
        owner_palette=OwnerPalette(
            colors={
                Ownership.SELF: (60, 180, 75),
                Ownership.ALLY: (0, 130, 200),
                Ownership.ENEMY: (230, 25, 75),
                Ownership.GAIA: (170, 170, 170),
                Ownership.UNKNOWN: (255, 255, 255),
            }
        ),
        fog_palette=FogPalette(
            colors={
                FogState.UNEXPLORED: (20, 20, 20),
                FogState.EXPLORED: (90, 90, 90),
                FogState.VISIBLE: (60, 120, 60),
            }
        ),
        health_good=(60, 180, 75),
        health_warn=(255, 200, 0),
        health_bad=(230, 25, 75),
        health_good_min=0.5,
        health_warn_min=0.25,
        hud_text_color=(255, 255, 255),
        hud_panel_color=(0, 0, 0),
        font_scale=0.4,
        box_thickness=1,
        minimap_fraction=0.25,
    )


def _default_perception() -> PerceptionSettings:
    return PerceptionSettings(
        ownership_palette=OwnershipPalette(
            colors=(
                OwnershipColor(
                    name="blue",
                    ownership=Ownership.SELF,
                    bands=(HsvWindow(h_lo=100, h_hi=130, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
                OwnershipColor(
                    name="green",
                    ownership=Ownership.ALLY,
                    bands=(HsvWindow(h_lo=45, h_hi=85, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
                OwnershipColor(
                    name="red",
                    ownership=Ownership.ENEMY,
                    bands=(
                        HsvWindow(h_lo=0, h_hi=10, s_lo=70, s_hi=255, v_lo=50, v_hi=255),
                        HsvWindow(h_lo=170, h_hi=179, s_lo=70, s_hi=255, v_lo=50, v_hi=255),
                    ),
                ),
                OwnershipColor(
                    name="yellow",
                    ownership=Ownership.GAIA,
                    bands=(HsvWindow(h_lo=22, h_hi=34, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
            )
        ),
        ownership_min_fraction=0.02,
        detect_resources=True,
        resource_cues=(
            ResourceCueSetting(
                entity_type="tree",
                bands=(HsvWindow(h_lo=35, h_hi=85, s_lo=40, s_hi=255, v_lo=30, v_hi=255),),
                min_area=20,
            ),
            ResourceCueSetting(
                entity_type="mine",
                bands=(HsvWindow(h_lo=0, h_hi=179, s_lo=0, s_hi=50, v_lo=50, v_hi=190),),
                min_area=20,
            ),
            ResourceCueSetting(
                entity_type="bush",
                bands=(
                    HsvWindow(h_lo=0, h_hi=8, s_lo=90, s_hi=255, v_lo=60, v_hi=255),
                    HsvWindow(h_lo=168, h_hi=179, s_lo=90, s_hi=255, v_lo=60, v_hi=255),
                ),
                min_area=20,
            ),
            ResourceCueSetting(
                entity_type="fauna",
                bands=(HsvWindow(h_lo=9, h_hi=25, s_lo=60, s_hi=200, v_lo=40, v_hi=190),),
                min_area=20,
            ),
        ),
        health=HealthReadSettings(max_offset=20, s_min=60, v_min=60, min_run=0.15),
        state=StateCueSettings(
            selection=SelectionCueSettings(thickness=3, brightness=200, min_fraction=0.4),
            construction=ConstructionCueSettings(
                edge_density_min=0.12, canny_lo=60.0, canny_hi=180.0
            ),
            garrison=GarrisonCueSettings(
                top_fraction=0.35,
                brightness=200,
                max_saturation=70,
                min_badge_area=6,
                max_badge_width_fraction=0.5,
            ),
        ),
    )


def _default_minimap() -> MinimapSettings:
    return MinimapSettings(
        palette=(
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
        ),
        world_extent=WorldExtentSettings(
            origin_x=0.0, origin_y=0.0, width=1024.0, height=1024.0, flip_y=True
        ),
        fog=FogSettings(rows=16, cols=16, unexplored_max=25.0, visible_min=140.0),
        blips=BlipSettings(tolerance=70.0, min_area=1, max_area=60, confidence=0.8),
        territory=TerritorySettings(tolerance=90.0, min_area=64),
        viewport=ViewportSettings(
            white_min=200, min_area=64, min_side=8, approx_epsilon_fraction=0.02
        ),
        disc_shape=False,
        region_confidence=0.9,
    )


def _default_hud() -> HudSettings:
    return HudSettings(
        top_bar=TopBarLayoutSettings(
            food=_slot(0.03, 0.10),
            wood=_slot(0.15, 0.10),
            stone=_slot(0.27, 0.10),
            metal=_slot(0.39, 0.10),
            population=_slot(0.51, 0.12),
            phase=_slot(0.80, 0.18),
            swatch=FractionalRegionSetting(x=0.0, y=0.2, width=0.025, height=0.6),
            civ=_slot(0.66, 0.13),
        ),
        selection=SelectionPanelLayoutSettings(
            entity_type=FractionalRegionSetting(x=0.05, y=0.05, width=0.9, height=0.25),
            health=FractionalRegionSetting(x=0.05, y=0.35, width=0.5, height=0.25),
            queue=FractionalRegionSetting(x=0.05, y=0.7, width=0.9, height=0.28),
        ),
        ocr_config="--psm 7",
    )


def default_config() -> Config:
    """The canonical default configuration — the single source of default values."""

    return Config(
        thresholds=_default_thresholds(),
        paths=Paths(recordings_dir=Path("recordings"), calibration_dir=Path("calibration")),
        overlay=_default_overlay(),
        perception=_default_perception(),
        minimap=_default_minimap(),
        hud=_default_hud(),
        tracking=TrackingSettings(
            iou_threshold=0.3,
            min_hits=1,
            max_staleness=15,
            decay=0.85,
            combat_drop=0.05,
            depletion_health=0.02,
        ),
        preprocessing=PreprocessingSettings(
            hud=HudPipelineSettings(gaussian_ksize=3, clahe_clip_limit=2.0, clahe_tile=(8, 8)),
            scene=ScenePipelineSettings(
                bilateral_diameter=5,
                bilateral_sigma_color=50.0,
                bilateral_sigma_space=50.0,
                clahe_clip_limit=3.0,
                clahe_tile=(8, 8),
            ),
        ),
        calibration=CalibrationSettings(
            ratios=HudLayoutRatiosSettings(
                top_bar_height=0.035,
                minimap_side=0.20,
                selection_width=0.34,
                selection_height=0.16,
            ),
            theme="default",
            use_anchors=True,
            default_ui_scale=1.0,
            ui_scale_min=0.5,
            ui_scale_max=3.0,
            selfcheck_match_threshold=0.5,
            selfcheck_use_anchors=True,
            persist_profiles=False,
        ),
        perf=PerfSettings(latency_target_ms=66.0, throughput_target_fps=15.0),
        pipeline=PipelineSettings(recalibrate_interval=30),
        acquisition=AcquisitionSettings(
            offline_fps=30.0,
            image_extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
            live_monitor=1,
            live_fps=30.0,
            record_fourcc="FFV1",  # lossless: --record footage feeds #2 real-frame metrics
            record_container=".mkv",  # FFV1's native container; readable by VideoFileSource
        ),
        geometry=GeometrySettings(
            camera_error_tolerance=1.0,
            fusion_agreement_scale=1.0,
            fusion_match_radius=20.0,
        ),
    )
