"""Tests for the typed config system (REQUIREMENTS.md X3 / NF7)."""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.domain.minimap import FogState
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.infrastructure.config import Config, load_config, save_config


def test_defaults_construct_without_arguments() -> None:
    config = Config()

    assert config.thresholds.min_confidence == 0.5
    assert config.overlay.owner_palette.for_ownership(Ownership.ENEMY) == (230, 25, 75)
    assert config.overlay.fog_palette.for_state(FogState.VISIBLE) == (60, 120, 60)


def test_load_config_with_no_sources_is_defaults() -> None:
    assert load_config(env={}) == Config()


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    original = Config()
    path = tmp_path / "config.json"

    save_config(original, path)
    reloaded = load_config(path, env={})

    assert reloaded == original


def test_file_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"thresholds": {"min_confidence": 0.9}}', encoding="utf-8")

    config = load_config(path, env={})

    assert config.thresholds.min_confidence == 0.9
    assert config.thresholds.hud_read_max_error == 0.01  # untouched default


def test_env_overrides_file_and_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"thresholds": {"min_confidence": 0.9}}', encoding="utf-8")
    env = {"ZAE_THRESHOLDS__MIN_CONFIDENCE": "0.42"}

    config = load_config(path, env=env)

    assert config.thresholds.min_confidence == 0.42


def test_health_color_thresholds() -> None:
    overlay = Config().overlay

    assert overlay.health_color(0.9) == overlay.health_good
    assert overlay.health_color(0.3) == overlay.health_warn
    assert overlay.health_color(0.1) == overlay.health_bad
