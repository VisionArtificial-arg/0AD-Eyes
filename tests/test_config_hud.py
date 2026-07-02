"""Config-externalization guards for the HUD subsystem (Approach B, P1 tail).

Golden: building the reader from default HudSettings reproduces the historical
layouts and OCR mode. Threading: a value from the config file reaches the adapter.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import HudSettings
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.hud.layout import SelectionPanelLayout, TopBarLayout
from zero_ad_eyes.infrastructure.hud.ocr import TesseractOcrEngine
from zero_ad_eyes.infrastructure.hud.reader import ClassicalHudReader


def _ocr_config(reader: ClassicalHudReader) -> str:
    ocr = reader._ocr
    assert isinstance(ocr, TesseractOcrEngine)
    return ocr._config


def test_reader_from_default_settings_reproduces_layouts() -> None:
    reader = ClassicalHudReader.from_settings(HudSettings())
    assert reader._top_bar == TopBarLayout()
    assert reader._selection == SelectionPanelLayout()


def test_reader_from_default_settings_uses_default_ocr_mode() -> None:
    reader = ClassicalHudReader.from_settings(HudSettings())
    assert _ocr_config(reader) == "--psm 7"


def test_config_file_threads_topbar_slot_to_reader(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"hud": {"top_bar": {"food": {"x": 0.5, "y": 0.0, "width": 0.2, "height": 1.0}}}}',
        encoding="utf-8",
    )

    config = load_config(path, env={})
    reader = ClassicalHudReader.from_settings(config.hud)

    assert reader._top_bar.food.x == 0.5
    assert reader._top_bar.food.width == 0.2
    assert reader._top_bar.wood.x == 0.15  # untouched default


def test_config_file_threads_ocr_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"hud": {"ocr_config": "--psm 6"}}', encoding="utf-8")

    config = load_config(path, env={})
    reader = ClassicalHudReader.from_settings(config.hud)

    assert _ocr_config(reader) == "--psm 6"
