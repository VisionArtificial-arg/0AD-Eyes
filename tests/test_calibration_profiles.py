"""Tests for calibration profile persistence (EPIC B — B3).

Profiles are built directly (no calibrator) to isolate the store: it owns *where*
and *when*, serialisation is the ``Calibration`` model's own responsibility. All
writes go under ``tmp_path`` so nothing lands outside the repository.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.calibration import CalibrationProfileStore


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
