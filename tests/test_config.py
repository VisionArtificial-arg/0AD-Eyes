"""Tests for the typed config system (REQUIREMENTS.md X3 / NF7).

The models are defaultless; the single source of default *values* is the generator
``interface.default_config.default_config`` (a UI concern), and the loader layers a
file/env on top of a caller-supplied base. These tests exercise that contract.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.domain.minimap import FogState
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.infrastructure.config import load_config, save_config
from zero_ad_eyes.interface.default_config import default_config


def test_generated_defaults_hold_the_expected_values() -> None:
    config = default_config()

    assert config.thresholds.min_confidence == 0.5
    assert config.overlay.owner_palette.for_ownership(Ownership.ENEMY) == (230, 25, 75)
    assert config.overlay.fog_palette.for_state(FogState.VISIBLE) == (60, 120, 60)


def test_load_config_with_no_sources_is_the_base() -> None:
    base = default_config()
    assert load_config(base, env={}) == base


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    original = default_config()
    path = tmp_path / "config.json"

    save_config(original, path)
    reloaded = load_config(default_config(), path, env={})

    assert reloaded == original


def test_file_overrides_base(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"thresholds": {"min_confidence": 0.9}}', encoding="utf-8")

    config = load_config(default_config(), path, env={})

    assert config.thresholds.min_confidence == 0.9
    assert config.thresholds.hud_read_max_error == 0.01  # untouched base value


def test_env_overrides_file_and_base(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"thresholds": {"min_confidence": 0.9}}', encoding="utf-8")
    env = {"ZAE_THRESHOLDS__MIN_CONFIDENCE": "0.42"}

    config = load_config(default_config(), path, env=env)

    assert config.thresholds.min_confidence == 0.42


def test_health_color_thresholds() -> None:
    overlay = default_config().overlay

    assert overlay.health_color(0.9) == overlay.health_good
    assert overlay.health_color(0.3) == overlay.health_warn
    assert overlay.health_color(0.1) == overlay.health_bad
