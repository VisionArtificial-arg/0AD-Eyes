"""Config-externalization guards for the calibration subsystem (Approach B, P3)."""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import CalibrationSettings, HudLayoutRatiosSettings
from zero_ad_eyes.infrastructure.calibration.layout import HudCalibrator
from zero_ad_eyes.infrastructure.calibration.ratios import HudLayoutRatios
from zero_ad_eyes.infrastructure.calibration.selfcheck import LayoutSelfCheck
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.interface.default_config import default_config


def _settings() -> CalibrationSettings:
    # Former in-code defaults, now hardcoded as explicit test literals.
    return CalibrationSettings(
        ratios=HudLayoutRatiosSettings(
            top_bar_height=0.035, minimap_side=0.20, selection_width=0.34, selection_height=0.16
        ),
        theme="default",
        use_anchors=True,
        default_ui_scale=1.0,
        ui_scale_min=0.5,
        ui_scale_max=3.0,
        selfcheck_match_threshold=0.5,
        selfcheck_use_anchors=True,
    )


def test_calibrator_from_default_settings_reproduces_knobs() -> None:
    cal = HudCalibrator.from_settings(_settings())
    assert cal._ratios == HudLayoutRatios(
        top_bar_height=0.035, minimap_side=0.20, selection_width=0.34, selection_height=0.16
    )
    assert cal.theme == "default"
    assert cal._use_anchors is True
    assert (cal._default_ui_scale, cal._ui_scale_min, cal._ui_scale_max) == (1.0, 0.5, 3.0)


def test_selfcheck_from_default_settings_reproduces_knobs() -> None:
    check = LayoutSelfCheck.from_settings(_settings())
    assert check._match_threshold == 0.5
    assert check._use_anchors is True


def test_ratios_defaults_match_historical() -> None:
    # Guard the generated calibration ratios against the frozen historical values.
    r = default_config().calibration.ratios
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

    config = load_config(default_config(), path, env={})
    cal = HudCalibrator.from_settings(config.calibration)
    check = LayoutSelfCheck.from_settings(config.calibration)

    assert cal._ui_scale_max == 4.0
    assert cal._ratios.minimap_side == 0.25
    assert cal._ratios.top_bar_height == 0.035  # untouched default
    assert check._match_threshold == 0.7
