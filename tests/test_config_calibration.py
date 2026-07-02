"""Config-externalization guards for the calibration subsystem (Approach B, P3)."""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import CalibrationSettings
from zero_ad_eyes.infrastructure.calibration.layout import HudCalibrator
from zero_ad_eyes.infrastructure.calibration.ratios import HudLayoutRatios
from zero_ad_eyes.infrastructure.calibration.selfcheck import LayoutSelfCheck
from zero_ad_eyes.infrastructure.config import load_config


def test_calibrator_from_default_settings_reproduces_knobs() -> None:
    cal = HudCalibrator.from_settings(CalibrationSettings())
    assert cal._ratios == HudLayoutRatios()
    assert cal.theme == "default"
    assert cal._use_anchors is True
    assert (cal._default_ui_scale, cal._ui_scale_min, cal._ui_scale_max) == (1.0, 0.5, 3.0)


def test_selfcheck_from_default_settings_reproduces_knobs() -> None:
    check = LayoutSelfCheck.from_settings(CalibrationSettings())
    assert check._match_threshold == 0.5
    assert check._use_anchors is True


def test_ratios_defaults_match_historical() -> None:
    r = CalibrationSettings().ratios
    assert (r.top_bar_height, r.minimap_side, r.selection_width, r.selection_height) == (
        0.035,
        0.20,
        0.34,
        0.16,
    )


def test_config_file_threads_calibration_knobs(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"calibration": {"ui_scale_max": 4.0, "ratios": {"minimap_side": 0.25}, '
        '"selfcheck_match_threshold": 0.7}}',
        encoding="utf-8",
    )

    config = load_config(path, env={})
    cal = HudCalibrator.from_settings(config.calibration)
    check = LayoutSelfCheck.from_settings(config.calibration)

    assert cal._ui_scale_max == 4.0
    assert cal._ratios.minimap_side == 0.25
    assert cal._ratios.top_bar_height == 0.035  # untouched default
    assert check._match_threshold == 0.7
