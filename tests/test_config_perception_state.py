"""Config guards for E4 health + E5 state-cue knobs (Approach B, P3).

Golden: default settings equal the historical thresholds. Threading: config values
reach the enricher.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import (
    HealthReadSettings,
    PerceptionSettings,
    StateCueSettings,
)
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.perception.enrichment import ClassicalEntityEnricher


def test_health_defaults_match_historical() -> None:
    h = HealthReadSettings()
    assert (h.max_offset, h.s_min, h.v_min, h.min_run) == (20, 60, 60, 0.15)


def test_state_defaults_match_historical() -> None:
    s = StateCueSettings()
    assert (s.selection.thickness, s.selection.brightness, s.selection.min_fraction) == (
        3,
        200,
        0.4,
    )
    assert (s.construction.edge_density_min, s.construction.canny_lo, s.construction.canny_hi) == (
        0.12,
        60.0,
        180.0,
    )
    assert s.garrison.top_fraction == 0.35
    assert s.garrison.max_saturation == 70


def test_enricher_from_default_settings_holds_defaults() -> None:
    enricher = ClassicalEntityEnricher.from_settings(PerceptionSettings())
    assert enricher._health == HealthReadSettings()
    assert enricher._state == StateCueSettings()


def test_config_file_threads_health_and_state(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"perception": {"health": {"max_offset": 40}, '
        '"state": {"selection": {"min_fraction": 0.6}}}}',
        encoding="utf-8",
    )

    config = load_config(path, env={})
    enricher = ClassicalEntityEnricher.from_settings(config.perception)

    assert enricher._health.max_offset == 40
    assert enricher._health.s_min == 60  # untouched default
    assert enricher._state.selection.min_fraction == 0.6
