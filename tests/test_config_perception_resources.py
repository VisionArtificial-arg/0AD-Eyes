"""Config guards for classical resource detection (Approach B, P3).

Golden: default settings reproduce the historical DEFAULT_RESOURCE_CUES. Threading:
config values reach the perception model.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import PerceptionSettings
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.perception.model import ClassicalPerceptionModel
from zero_ad_eyes.infrastructure.perception.resources import (
    DEFAULT_RESOURCE_CUES,
    resource_cues_from_settings,
)


def test_default_cues_derive_from_config_default() -> None:
    assert resource_cues_from_settings(PerceptionSettings().resource_cues) == DEFAULT_RESOURCE_CUES


def test_default_cue_values_unchanged() -> None:
    by_type = {cue.entity_type: cue for cue in DEFAULT_RESOURCE_CUES}
    assert set(by_type) == {"tree", "mine", "bush", "fauna"}
    tree = by_type["tree"].bands[0]
    assert (tree.h_lo, tree.h_hi, tree.s_lo, tree.v_lo) == (35, 85, 40, 30)
    assert by_type["bush"].bands[1].h_lo == 168  # the hue-wrapped second band
    assert by_type["mine"].bands[0].v_hi == 190


def test_model_from_default_settings_matches_defaults() -> None:
    model = ClassicalPerceptionModel.from_settings(PerceptionSettings())
    assert model._detect_resources is True
    assert model._resource_cues == DEFAULT_RESOURCE_CUES


def test_config_file_threads_detect_resources_toggle(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"perception": {"detect_resources": false}}', encoding="utf-8")

    config = load_config(path, env={})
    model = ClassicalPerceptionModel.from_settings(config.perception)

    assert model._detect_resources is False
    assert model._resource_cues == DEFAULT_RESOURCE_CUES  # cues untouched
