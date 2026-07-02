"""Full-tree config snapshot (P5) — the whole externalized surface, in one guard.

docs/config.example.json is the canonical dump of Config() defaults. If any config
default changes (or a new section is added) without updating the sample, this fails —
which is the point: the sample stays an accurate, copy-pasteable template, and no
default drifts silently.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import Config
from zero_ad_eyes.infrastructure.config import load_config

_SAMPLE = Path(__file__).resolve().parents[1] / "docs" / "config.example.json"


def test_sample_config_matches_defaults() -> None:
    expected = Config().model_dump_json(indent=2) + "\n"
    assert _SAMPLE.read_text(encoding="utf-8") == expected


def test_sample_config_round_trips_to_defaults() -> None:
    assert load_config(_SAMPLE, env={}) == Config()
