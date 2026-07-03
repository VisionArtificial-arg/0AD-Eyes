"""Config guards for classical resource detection (Approach B, P3).

Golden: default settings reproduce the historical DEFAULT_RESOURCE_CUES. Threading:
config values reach the perception model.
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
from zero_ad_eyes.infrastructure.perception.model import ClassicalPerceptionModel
from zero_ad_eyes.infrastructure.perception.palette import HsvBand
from zero_ad_eyes.infrastructure.perception.resources import (
    ResourceCue,
    resource_cues_from_settings,
)
from zero_ad_eyes.interface.default_config import default_config

# The historical default resource cues (pure data), spelled out as explicit literals.
DEFAULT_CUE_SETTINGS: tuple[ResourceCueSetting, ...] = (
    ResourceCueSetting(
        entity_type="tree",
        bands=(HsvWindow(h_lo=35, h_hi=85, s_lo=40, s_hi=255, v_lo=30, v_hi=255),),
        min_area=20,
    ),
    ResourceCueSetting(
        entity_type="mine",
        bands=(HsvWindow(h_lo=0, h_hi=179, s_lo=0, s_hi=50, v_lo=50, v_hi=190),),
        min_area=20,
    ),
    ResourceCueSetting(
        entity_type="bush",
        bands=(
            HsvWindow(h_lo=0, h_hi=8, s_lo=90, s_hi=255, v_lo=60, v_hi=255),
            HsvWindow(h_lo=168, h_hi=179, s_lo=90, s_hi=255, v_lo=60, v_hi=255),
        ),
        min_area=20,
    ),
    ResourceCueSetting(
        entity_type="fauna",
        bands=(HsvWindow(h_lo=9, h_hi=25, s_lo=60, s_hi=200, v_lo=40, v_hi=190),),
        min_area=20,
    ),
)

# The same cues, rehydrated into the cv2-capable infra type (the historical default).
DEFAULT_RESOURCE_CUES: tuple[ResourceCue, ...] = (
    ResourceCue(
        entity_type="tree",
        bands=(HsvBand(h_lo=35, h_hi=85, s_lo=40, s_hi=255, v_lo=30, v_hi=255),),
        min_area=20,
    ),
    ResourceCue(
        entity_type="mine",
        bands=(HsvBand(h_lo=0, h_hi=179, s_lo=0, s_hi=50, v_lo=50, v_hi=190),),
        min_area=20,
    ),
    ResourceCue(
        entity_type="bush",
        bands=(
            HsvBand(h_lo=0, h_hi=8, s_lo=90, s_hi=255, v_lo=60, v_hi=255),
            HsvBand(h_lo=168, h_hi=179, s_lo=90, s_hi=255, v_lo=60, v_hi=255),
        ),
        min_area=20,
    ),
    ResourceCue(
        entity_type="fauna",
        bands=(HsvBand(h_lo=9, h_hi=25, s_lo=60, s_hi=200, v_lo=40, v_hi=190),),
        min_area=20,
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
                OwnershipColor(
                    name="green",
                    ownership=Ownership.ALLY,
                    bands=(HsvWindow(h_lo=45, h_hi=85, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
                OwnershipColor(
                    name="red",
                    ownership=Ownership.ENEMY,
                    bands=(
                        HsvWindow(h_lo=0, h_hi=10, s_lo=70, s_hi=255, v_lo=50, v_hi=255),
                        HsvWindow(h_lo=170, h_hi=179, s_lo=70, s_hi=255, v_lo=50, v_hi=255),
                    ),
                ),
                OwnershipColor(
                    name="yellow",
                    ownership=Ownership.GAIA,
                    bands=(HsvWindow(h_lo=22, h_hi=34, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
            )
        ),
        ownership_min_fraction=0.02,
        detect_resources=False,
        resource_cues=DEFAULT_CUE_SETTINGS,
        health=HealthReadSettings(max_offset=20, s_min=60, v_min=60, min_run=0.15),
        state=StateCueSettings(
            selection=SelectionCueSettings(thickness=3, brightness=200, min_fraction=0.4),
            construction=ConstructionCueSettings(
                edge_density_min=0.12, canny_lo=60.0, canny_hi=180.0
            ),
            garrison=GarrisonCueSettings(
                top_fraction=0.35,
                brightness=200,
                max_saturation=70,
                min_badge_area=6,
                max_badge_width_fraction=0.5,
            ),
        ),
    )


def test_default_cues_derive_from_config_default() -> None:
    # The generator default rehydrates into exactly the frozen golden infra cues.
    generated = resource_cues_from_settings(default_config().perception.resource_cues)
    assert generated == DEFAULT_RESOURCE_CUES


def test_default_cue_values_unchanged() -> None:
    by_type = {cue.entity_type: cue for cue in default_config().perception.resource_cues}
    assert set(by_type) == {"tree", "mine", "bush", "fauna"}
    tree = by_type["tree"].bands[0]
    assert (tree.h_lo, tree.h_hi, tree.s_lo, tree.v_lo) == (35, 85, 40, 30)
    assert by_type["bush"].bands[1].h_lo == 168  # the hue-wrapped second band
    assert by_type["mine"].bands[0].v_hi == 190


def test_model_from_default_settings_matches_defaults() -> None:
    model = ClassicalPerceptionModel.from_settings(_perception_settings())
    assert model._detect_resources is False
    assert model._resource_cues == DEFAULT_RESOURCE_CUES


def test_config_file_threads_detect_resources_toggle(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"perception": {"detect_resources": true}}', encoding="utf-8")

    config = load_config(default_config(), path, env={})
    model = ClassicalPerceptionModel.from_settings(config.perception)

    assert model._detect_resources is True
    assert model._resource_cues == DEFAULT_RESOURCE_CUES  # cues untouched
