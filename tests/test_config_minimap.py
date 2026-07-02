"""Config-externalization guards for the minimap subsystem (Approach B, P2).

Golden: building the reader from default MinimapSettings reproduces the historical
hard-coded knobs. Threading: a value set in the config file reaches the collaborator.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import MinimapSettings
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.minimap.palette import MinimapPalette
from zero_ad_eyes.infrastructure.minimap.reader import ClassicalMinimapReader
from zero_ad_eyes.infrastructure.minimap.segmentation import MinimapShape


def test_minimap_palette_from_default_settings_matches_default() -> None:
    assert MinimapPalette.from_settings(MinimapSettings().palette) == MinimapPalette.default()


def test_reader_from_default_settings_reproduces_hardcoded_knobs() -> None:
    reader = ClassicalMinimapReader.from_settings(MinimapSettings())

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
    assert reader._blip_detector.palette == MinimapPalette.default()


def test_disc_shape_flag_selects_disc_segmenter() -> None:
    reader = ClassicalMinimapReader.from_settings(MinimapSettings(disc_shape=True))
    assert reader._segmenter._shape is MinimapShape.DISC


def test_config_file_threads_fog_threshold_to_reader(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"minimap": {"fog": {"visible_min": 200.0}}}', encoding="utf-8")

    config = load_config(path, env={})
    reader = ClassicalMinimapReader.from_settings(config.minimap)

    assert reader._fog_classifier.visible_min == 200.0
    assert reader._fog_classifier.unexplored_max == 25.0  # untouched default
