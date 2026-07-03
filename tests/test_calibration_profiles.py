"""Tests for calibration profile persistence (EPIC B — B3).

Profiles are built directly (no calibrator) to isolate the store: it owns *where*
and *when*, serialisation is the ``Calibration`` model's own responsibility. All
writes go under ``tmp_path`` so nothing lands outside the repository.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource
from zero_ad_eyes.infrastructure.calibration import CalibrationProfileStore
from zero_ad_eyes.interface.manual_calibration import (
    collect_manual_calibration,
    collect_manual_calibration_from_source,
    save_manual_calibration,
)

from .conftest import make_frame


def _calibration() -> Calibration:
    return Calibration(
        width=1920,
        height=1080,
        ui_scale=1.5,
        top_bar=ScreenBBox(x=0, y=0, width=1920, height=40),
        minimap=ScreenBBox(x=0, y=880, width=200, height=200),
        selection_panel=ScreenBBox(x=760, y=920, width=400, height=160),
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = CalibrationProfileStore(tmp_path)
    profile = _calibration()

    store.save(profile, "night")
    reloaded = store.load(profile.width, profile.height, "night")

    assert reloaded == profile
    assert (tmp_path / "1920x1080@night.json").is_file()


def test_load_missing_returns_none(tmp_path: Path) -> None:
    store = CalibrationProfileStore(tmp_path)
    assert store.load(1920, 1080, "default") is None


def test_key_and_theme_slug() -> None:
    store = CalibrationProfileStore()
    assert store.key(1920, 1080, "default") == "1920x1080@default"
    assert store.key(1920, 1080, "a theme/with:bad*chars") == "1920x1080@a_theme_with_bad_chars"


def test_theme_scopes_the_profile(tmp_path: Path) -> None:
    store = CalibrationProfileStore(tmp_path)
    profile = _calibration()
    store.save(profile, "day")
    assert store.load(profile.width, profile.height, "night") is None
    assert store.load(profile.width, profile.height, "day") == profile


def test_collect_manual_calibration_uses_selector_boxes() -> None:
    frame = make_frame(width=100, height=80)
    boxes = iter(
        (
            (0, 0, 10, 8),
            (10, 0, 10, 8),
            (20, 0, 10, 8),
            (30, 0, 10, 8),
            (40, 0, 10, 8),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (2, 50, 20, 20),
        )
    )

    calibration = collect_manual_calibration(
        frame,
        selector=lambda _label, _image: next(boxes),
    )

    assert calibration.width == 100
    assert calibration.height == 80
    assert calibration.top_bar == ScreenBBox(x=0, y=0, width=50, height=8)
    assert calibration.hud_regions["food"] == ScreenBBox(x=0, y=0, width=10, height=8)
    assert "phase" not in calibration.hud_regions
    assert "swatch" not in calibration.hud_regions
    assert "civ" not in calibration.hud_regions
    assert calibration.minimap == ScreenBBox(x=2, y=50, width=20, height=20)
    assert calibration.selection_panel is None


def test_collect_manual_calibration_from_source_can_freeze_later_frame() -> None:
    first = make_frame(frame_id=0, width=100, height=80)
    second = make_frame(frame_id=1, width=100, height=80)
    first.image[0, 0, 0] = 10
    second.image[0, 0, 0] = 20

    def freezer(label: str, current, next_frame):
        if label == "selection panel (optional)":
            return next_frame()
        return current

    def selector(_label: str, image):
        return (int(image[0, 0, 0]), 0, 2, 2)

    calibration = collect_manual_calibration_from_source(
        InMemoryFrameSource((first, second)),
        selector=selector,
        freezer=freezer,
    )

    assert calibration.hud_regions["food"] == ScreenBBox(x=10, y=0, width=2, height=2)
    assert calibration.selection_panel == ScreenBBox(x=20, y=0, width=2, height=2)


def test_save_manual_calibration_writes_profile(tmp_path: Path) -> None:
    frame = make_frame(width=100, height=80)
    boxes = iter(
        (
            (0, 0, 10, 8),
            (10, 0, 10, 8),
            (20, 0, 10, 8),
            (30, 0, 10, 8),
            (40, 0, 10, 8),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (2, 50, 20, 20),
        )
    )

    path = save_manual_calibration(
        frame,
        directory=tmp_path,
        theme="manual",
        selector=lambda _label, _image: next(boxes),
    )

    assert path == tmp_path / "100x80@manual.json"
    assert CalibrationProfileStore(tmp_path).load(100, 80, "manual") is not None
