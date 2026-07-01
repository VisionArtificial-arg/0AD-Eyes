"""Tests for the calibration self-check (EPIC B — B4)."""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.calibration import HudCalibrator, LayoutSelfCheck

from ._hud_fixtures import make_hud_frame


def test_matching_frame_scores_high() -> None:
    frame = make_hud_frame(top_frac=0.06, bottom_frac=0.18)
    calibration = HudCalibrator().calibrate(frame)
    check = LayoutSelfCheck().verify(frame, calibration)
    assert check.matches
    assert check.confidence == pytest.approx(1.0, abs=0.05)


def test_resolution_mismatch_is_rejected() -> None:
    calibration = HudCalibrator().calibrate(make_hud_frame(width=640, height=480, top_frac=0.06))
    check = LayoutSelfCheck().verify(
        make_hud_frame(width=1280, height=720, top_frac=0.06), calibration
    )
    assert not check.matches
    assert check.confidence == 0.0
    assert "resolution mismatch" in check.reason


def test_anchor_drift_lowers_confidence() -> None:
    calibration = HudCalibrator().calibrate(make_hud_frame(top_frac=0.10))
    # Same resolution, but the top band is now far thinner: the layout drifted.
    check = LayoutSelfCheck().verify(make_hud_frame(top_frac=0.02), calibration)
    assert not check.matches
    assert check.confidence < 0.5


def test_unverifiable_when_no_anchors() -> None:
    calibration = Calibration(
        width=640,
        height=480,
        top_bar=ScreenBBox(x=0, y=0, width=640, height=20),
    )
    check = LayoutSelfCheck().verify(make_hud_frame(width=640, height=480), calibration)
    assert check.matches
    assert check.confidence == pytest.approx(0.5)
