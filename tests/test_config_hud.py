"""Config-externalization guards for the HUD subsystem (Approach B, P1 tail).

Golden: building the reader from default HudSettings reproduces the historical
layouts and OCR mode. Threading: a value from the config file reaches the adapter.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import (
    FractionalRegionSetting,
    HudSettings,
    SelectionPanelLayoutSettings,
    TopBarLayoutSettings,
)
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.hud.layout import (
    FractionalRegion,
    SelectionPanelLayout,
    TopBarLayout,
)
from zero_ad_eyes.infrastructure.hud.ocr import TesseractOcrEngine
from zero_ad_eyes.infrastructure.hud.reader import ClassicalHudReader
from zero_ad_eyes.interface.default_config import default_config


def _ocr_config(reader: ClassicalHudReader) -> str:
    ocr = reader._ocr
    assert isinstance(ocr, TesseractOcrEngine)
    return ocr._config


# Former in-code defaults, now hardcoded as explicit test literals.
def _top_bar_layout() -> TopBarLayout:
    return TopBarLayout(
        food=FractionalRegion(x=0.03, y=0.0, width=0.10, height=1.0),
        wood=FractionalRegion(x=0.15, y=0.0, width=0.10, height=1.0),
        stone=FractionalRegion(x=0.27, y=0.0, width=0.10, height=1.0),
        metal=FractionalRegion(x=0.39, y=0.0, width=0.10, height=1.0),
        population=FractionalRegion(x=0.51, y=0.0, width=0.12, height=1.0),
        phase=FractionalRegion(x=0.80, y=0.0, width=0.18, height=1.0),
        swatch=FractionalRegion(x=0.0, y=0.2, width=0.025, height=0.6),
        civ=FractionalRegion(x=0.66, y=0.0, width=0.13, height=1.0),
    )


def _selection_layout() -> SelectionPanelLayout:
    return SelectionPanelLayout(
        entity_type=FractionalRegion(x=0.05, y=0.05, width=0.9, height=0.25),
        health=FractionalRegion(x=0.05, y=0.35, width=0.5, height=0.25),
        queue=FractionalRegion(x=0.05, y=0.7, width=0.9, height=0.28),
    )


def _hud_settings() -> HudSettings:
    return HudSettings(
        top_bar=TopBarLayoutSettings(
            food=FractionalRegionSetting(x=0.03, y=0.0, width=0.10, height=1.0),
            wood=FractionalRegionSetting(x=0.15, y=0.0, width=0.10, height=1.0),
            stone=FractionalRegionSetting(x=0.27, y=0.0, width=0.10, height=1.0),
            metal=FractionalRegionSetting(x=0.39, y=0.0, width=0.10, height=1.0),
            population=FractionalRegionSetting(x=0.51, y=0.0, width=0.12, height=1.0),
            phase=FractionalRegionSetting(x=0.80, y=0.0, width=0.18, height=1.0),
            swatch=FractionalRegionSetting(x=0.0, y=0.2, width=0.025, height=0.6),
            civ=FractionalRegionSetting(x=0.66, y=0.0, width=0.13, height=1.0),
        ),
        selection=SelectionPanelLayoutSettings(
            entity_type=FractionalRegionSetting(x=0.05, y=0.05, width=0.9, height=0.25),
            health=FractionalRegionSetting(x=0.05, y=0.35, width=0.5, height=0.25),
            queue=FractionalRegionSetting(x=0.05, y=0.7, width=0.9, height=0.28),
        ),
        ocr_config="--psm 7",
    )


def test_reader_from_default_settings_reproduces_layouts() -> None:
    reader = ClassicalHudReader.from_settings(_hud_settings())
    assert reader._top_bar == _top_bar_layout()
    assert reader._selection == _selection_layout()


def test_reader_from_default_settings_uses_default_ocr_mode() -> None:
    reader = ClassicalHudReader.from_settings(_hud_settings())
    assert _ocr_config(reader) == "--psm 7"


def test_config_file_threads_topbar_slot_to_reader(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"hud": {"top_bar": {"food": {"x": 0.5, "y": 0.0, "width": 0.2, "height": 1.0}}}}',
        encoding="utf-8",
    )

    config = load_config(default_config(), path, env={})
    reader = ClassicalHudReader.from_settings(config.hud)

    assert reader._top_bar.food.x == 0.5
    assert reader._top_bar.food.width == 0.2
    assert reader._top_bar.wood.x == 0.15  # untouched default


def test_config_file_threads_ocr_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"hud": {"ocr_config": "--psm 6"}}', encoding="utf-8")

    config = load_config(default_config(), path, env={})
    reader = ClassicalHudReader.from_settings(config.hud)

    assert _ocr_config(reader) == "--psm 6"
