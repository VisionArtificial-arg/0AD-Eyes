"""Config-externalization guards for the minimap subsystem (Approach B, P2).

Golden: building the reader from default MinimapSettings reproduces the historical
hard-coded knobs. Threading: a value set in the config file reaches the collaborator.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import (
    BgrColorSetting,
    BlipSettings,
    FogSettings,
    MinimapPaletteEntry,
    MinimapSettings,
    TerritorySettings,
    ViewportSettings,
    WorldExtentSettings,
)
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.minimap.palette import BgrColor, MinimapPalette, PaletteEntry
from zero_ad_eyes.infrastructure.minimap.reader import ClassicalMinimapReader
from zero_ad_eyes.infrastructure.minimap.segmentation import MinimapShape
from zero_ad_eyes.interface.default_config import default_config


def _default_palette() -> MinimapPalette:
    return MinimapPalette(
        entries=(
            PaletteEntry("self", BgrColor(235, 90, 40), Ownership.SELF),
            PaletteEntry("ally", BgrColor(60, 200, 60), Ownership.ALLY),
            PaletteEntry("enemy", BgrColor(40, 40, 220), Ownership.ENEMY),
            PaletteEntry("gaia", BgrColor(235, 235, 235), Ownership.GAIA),
        )
    )


def _settings(*, disc_shape: bool = False) -> MinimapSettings:
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
                label="gaia",
                color=BgrColorSetting(b=235, g=235, r=235),
                ownership=Ownership.GAIA,
            ),
        ),
        world_extent=WorldExtentSettings(
            origin_x=0.0, origin_y=0.0, width=1024.0, height=1024.0, flip_y=True
        ),
        fog=FogSettings(rows=16, cols=16, unexplored_max=25.0, visible_min=140.0),
        blips=BlipSettings(tolerance=70.0, min_area=1, max_area=60, confidence=0.8),
        territory=TerritorySettings(tolerance=90.0, min_area=64),
        viewport=ViewportSettings(white_min=200, min_area=64, min_side=8),
        disc_shape=disc_shape,
        region_confidence=0.9,
    )


def test_minimap_palette_from_default_settings_matches_default() -> None:
    assert MinimapPalette.from_settings(_settings().palette) == _default_palette()


def test_reader_from_default_settings_reproduces_hardcoded_knobs() -> None:
    reader = ClassicalMinimapReader.from_settings(_settings())

    fog = reader._fog_classifier
    assert (fog.rows, fog.cols, fog.unexplored_max, fog.visible_min) == (16, 16, 25.0, 140.0)

    blips = reader._blip_detector
    assert (blips.tolerance, blips.min_area, blips.max_area, blips.confidence) == (70.0, 1, 60, 0.8)

    viewport = reader._viewport_detector
    assert (viewport.white_min, viewport.min_area, viewport.min_side) == (200, 64, 8)

    territory = reader._territory_extractor
    assert (territory.tolerance, territory.min_area) == (90.0, 64)

    extent = reader._world_extent
    assert (extent.origin_x, extent.origin_y, extent.width, extent.height, extent.flip_y) == (
        0.0,
        0.0,
        1024.0,
        1024.0,
        True,
    )

    assert reader._region_confidence == 0.9
    assert reader._blip_detector.palette == _default_palette()


def test_disc_shape_flag_selects_disc_segmenter() -> None:
    reader = ClassicalMinimapReader.from_settings(_settings(disc_shape=True))
    assert reader._segmenter._shape is MinimapShape.DISC


def test_config_file_threads_fog_threshold_to_reader(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"minimap": {"fog": {"visible_min": 200.0}}}', encoding="utf-8")

    config = load_config(default_config(), path, env={})
    reader = ClassicalMinimapReader.from_settings(config.minimap)

    assert reader._fog_classifier.visible_min == 200.0
    assert reader._fog_classifier.unexplored_max == 25.0  # untouched default
