"""Config guards for E4 health + E5 state-cue knobs (Approach B, P3).

Golden: default settings equal the historical thresholds. Threading: config values
reach the enricher.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import (
    ConstructionCueSettings,
    GarrisonCueSettings,
    HealthReadSettings,
    HsvWindow,
    OwnershipColor,
    OwnershipPalette,
    PerceptionSettings,
    ResourceCueSetting,
    SelectionCueSettings,
    StateCueSettings,
)
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.perception.enrichment import ClassicalEntityEnricher
from zero_ad_eyes.interface.default_config import default_config

# The historical E4/E5 thresholds, spelled out as explicit literals.
HEALTH = HealthReadSettings(max_offset=20, s_min=60, v_min=60, min_run=0.15)
STATE = StateCueSettings(
    selection=SelectionCueSettings(thickness=3, brightness=200, min_fraction=0.4),
    construction=ConstructionCueSettings(edge_density_min=0.12, canny_lo=60.0, canny_hi=180.0),
    garrison=GarrisonCueSettings(
        top_fraction=0.35,
        brightness=200,
        max_saturation=70,
        min_badge_area=6,
        max_badge_width_fraction=0.5,
    ),
)


def _perception_settings() -> PerceptionSettings:
    """A full PerceptionSettings wired with the historical defaults, spelled out."""

    return PerceptionSettings(
        ownership_palette=OwnershipPalette(
            colors=(
                OwnershipColor(
                    name="blue",
                    ownership=Ownership.SELF,
                    bands=(HsvWindow(h_lo=100, h_hi=130, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
            )
        ),
        ownership_min_fraction=0.02,
        detect_resources=True,
        resource_cues=(
            ResourceCueSetting(
                entity_type="tree",
                bands=(HsvWindow(h_lo=35, h_hi=85, s_lo=40, s_hi=255, v_lo=30, v_hi=255),),
                min_area=20,
            ),
        ),
        health=HEALTH,
        state=STATE,
    )


def test_health_defaults_match_historical() -> None:
    # Guard the generated default against the frozen historical E4 thresholds.
    h = default_config().perception.health
    assert (h.max_offset, h.s_min, h.v_min, h.min_run) == (20, 60, 60, 0.15)


def test_state_defaults_match_historical() -> None:
    s = default_config().perception.state
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
    enricher = ClassicalEntityEnricher.from_settings(_perception_settings())
    assert enricher._health == HEALTH
    assert enricher._state == STATE


def test_config_file_threads_health_and_state(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"perception": {"health": {"max_offset": 40}, '
        '"state": {"selection": {"min_fraction": 0.6}}}}',
        encoding="utf-8",
    )

    config = load_config(default_config(), path, env={})
    enricher = ClassicalEntityEnricher.from_settings(config.perception)

    assert enricher._health.max_offset == 40
    assert enricher._health.s_min == 60  # untouched default
    assert enricher._state.selection.min_fraction == 0.6
