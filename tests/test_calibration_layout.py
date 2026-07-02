"""Tests for the HUD calibrator (EPIC B — B1 resolution/UI-scale, +B3 reuse).

Synthetic frames carry opaque top/bottom HUD bands so the pixel anchors are
recoverable; resolution and UI scale are asserted resolution-relative (A4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zero_ad_eyes.infrastructure.calibration import (
    CalibrationProfileStore,
    HudCalibrator,
    HudLayoutRatios,
)

from ._hud_fixtures import make_hud_frame, within_frame

# Former in-code defaults, now hardcoded as explicit test literals (no config lookup).
_RATIOS = HudLayoutRatios(
    top_bar_height=0.035, minimap_side=0.20, selection_width=0.34, selection_height=0.16
)


def _calibrator(
    ratios: HudLayoutRatios | None = None,
    *,
    theme: str = "default",
    use_anchors: bool = True,
    default_ui_scale: float = 1.0,
    ui_scale_min: float = 0.5,
    ui_scale_max: float = 3.0,
    store: CalibrationProfileStore | None = None,
) -> HudCalibrator:
    return HudCalibrator(
        ratios if ratios is not None else _RATIOS,
        theme=theme,
        use_anchors=use_anchors,
        default_ui_scale=default_ui_scale,
        ui_scale_min=ui_scale_min,
        ui_scale_max=ui_scale_max,
        store=store,
    )


def test_resolution_read_from_pixels() -> None:
    calibration = _calibrator(use_anchors=False).calibrate(make_hud_frame(width=800, height=600))
    assert (calibration.width, calibration.height) == (800, 600)


def test_default_ui_scale_without_anchors() -> None:
    # A plain black scene yields no detectable band, so the default scale stands.
    calibration = _calibrator(default_ui_scale=1.0).calibrate(make_hud_frame())
    assert calibration.ui_scale == pytest.approx(1.0)


def test_ui_scale_estimated_from_top_anchor() -> None:
    # A top bar twice the canonical thickness implies ~2x UI scale.
    ratios = _RATIOS
    frame = make_hud_frame(top_frac=ratios.top_bar_height * 2.0)
    calibration = _calibrator(ratios).calibrate(frame)
    assert calibration.ui_scale == pytest.approx(2.0, abs=0.1)


def test_calibrate_populates_all_regions_within_frame() -> None:
    frame = make_hud_frame(width=1024, height=768, top_frac=0.05, bottom_frac=0.18)
    calibration = _calibrator().calibrate(frame)
    assert calibration.top_bar is not None
    assert calibration.minimap is not None
    assert calibration.selection_panel is not None
    for box in (calibration.top_bar, calibration.minimap, calibration.selection_panel):
        assert within_frame(box, calibration.width, calibration.height)


def test_anchor_refines_top_bar_height() -> None:
    height, top_frac = 480, 0.08
    calibration = _calibrator().calibrate(make_hud_frame(height=height, top_frac=top_frac))
    assert calibration.top_bar is not None
    assert calibration.top_bar.height == pytest.approx(top_frac * height, abs=1.0)


def test_reuse_skips_recomputation(tmp_path: Path) -> None:
    store = CalibrationProfileStore(tmp_path)
    # First session: an anchored frame yields ui_scale ~2.0 and is persisted.
    first = _calibrator(store=store).calibrate(
        make_hud_frame(top_frac=_RATIOS.top_bar_height * 2.0)
    )
    assert first.ui_scale == pytest.approx(2.0, abs=0.1)

    # Second session: same resolution/theme but a plain frame. Reuse must return the
    # stored profile, not recompute it to the default scale.
    reused = _calibrator(store=store).calibrate(make_hud_frame())
    assert reused == first


def test_saved_profile_lands_under_configured_dir(tmp_path: Path) -> None:
    store = CalibrationProfileStore(tmp_path)
    _calibrator(store=store, theme="dusk").calibrate(make_hud_frame(width=640, height=480))
    assert (tmp_path / "640x480@dusk.json").is_file()
