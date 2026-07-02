"""Tests for the calibration self-check (EPIC B — B4)."""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.calibration import (
    HudCalibrator,
    HudLayoutRatios,
    LayoutSelfCheck,
)

from ._hud_fixtures import make_hud_frame


def _calibrator() -> HudCalibrator:
    # Former in-code defaults, now hardcoded as explicit test literals.
    return HudCalibrator(
        HudLayoutRatios(
            top_bar_height=0.035, minimap_side=0.20, selection_width=0.34, selection_height=0.16
        ),
        theme="default",
        use_anchors=True,
        default_ui_scale=1.0,
        ui_scale_min=0.5,
        ui_scale_max=3.0,
    )


def _selfcheck() -> LayoutSelfCheck:
    return LayoutSelfCheck(match_threshold=0.5, use_anchors=True)


def test_matching_frame_scores_high() -> None:
    frame = make_hud_frame(top_frac=0.06, bottom_frac=0.18)
    calibration = _calibrator().calibrate(frame)
    check = _selfcheck().verify(frame, calibration)
    assert check.matches
    assert check.confidence == pytest.approx(1.0, abs=0.05)


def test_resolution_mismatch_is_rejected() -> None:
    calibration = _calibrator().calibrate(make_hud_frame(width=640, height=480, top_frac=0.06))
    check = _selfcheck().verify(make_hud_frame(width=1280, height=720, top_frac=0.06), calibration)
    assert not check.matches
    assert check.confidence == 0.0
    assert "resolution mismatch" in check.reason


def test_anchor_drift_lowers_confidence() -> None:
    calibration = _calibrator().calibrate(make_hud_frame(top_frac=0.10))
    # Same resolution, but the top band is now far thinner: the layout drifted.
    check = _selfcheck().verify(make_hud_frame(top_frac=0.02), calibration)
    assert not check.matches
    assert check.confidence < 0.5


def test_unverifiable_when_no_anchors() -> None:
    calibration = Calibration(
        width=640,
        height=480,
        top_bar=ScreenBBox(x=0, y=0, width=640, height=20),
    )
    check = _selfcheck().verify(make_hud_frame(width=640, height=480), calibration)
    assert check.matches
    assert check.confidence == pytest.approx(0.5)
