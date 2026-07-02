"""Config-externalization guards (Approach B, P0+P1).

Golden checks that moving tuning values into the config changed NO behaviour, plus
proof that a file/env value actually threads through to the adapter.

Post-"no-defaults": the models are defaultless and the single source of default
*values* is ``interface.default_config.default_config``. These tests therefore assert
against the generated defaults (the config system is their subject) — the golden
bands below are frozen literals, so a drift in the generator still fails here.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.data.evaluation import EvalConfig
from zero_ad_eyes.infrastructure.perception.enrichment import ClassicalEntityEnricher
from zero_ad_eyes.infrastructure.perception.palette import PlayerPalette
from zero_ad_eyes.interface.default_config import default_config

# The historical default palette, frozen here as the golden. If the generated default
# ever changes a value, this fails — which is the point.
_EXPECTED_BANDS: dict[Ownership, list[tuple[int, int, int, int, int, int]]] = {
    Ownership.SELF: [(100, 130, 70, 255, 50, 255)],
    Ownership.ALLY: [(45, 85, 70, 255, 50, 255)],
    Ownership.ENEMY: [(0, 10, 70, 255, 50, 255), (170, 179, 70, 255, 50, 255)],
    Ownership.GAIA: [(22, 34, 70, 255, 50, 255)],
}


def _default_palette() -> PlayerPalette:
    """The infra palette rehydrated from the generated default perception config."""

    return PlayerPalette.from_settings(default_config().perception.ownership_palette)


def test_default_palette_values_unchanged_by_externalization() -> None:
    by_owner = {color.ownership: color for color in _default_palette().colors}
    assert set(by_owner) == set(_EXPECTED_BANDS)
    for ownership, expected in _EXPECTED_BANDS.items():
        actual = [
            (b.h_lo, b.h_hi, b.s_lo, b.s_hi, b.v_lo, b.v_hi) for b in by_owner[ownership].bands
        ]
        assert actual == expected


def test_eval_config_from_thresholds_matches_standalone_defaults() -> None:
    # Unifying the NF3 targets onto Thresholds must not shift any default.
    assert EvalConfig.from_thresholds(default_config().thresholds) == EvalConfig()


def test_thresholds_own_all_nf3_targets() -> None:
    t = default_config().thresholds
    assert (t.hud_read_max_error, t.detection_map_target) == (0.01, 0.80)
    assert (t.ownership_accuracy_target, t.tracking_mota_target) == (0.98, 0.70)
    assert t.eval_iou_threshold == 0.5


def test_enricher_from_default_settings_uses_default_palette() -> None:
    enricher = ClassicalEntityEnricher.from_settings(default_config().perception)
    assert enricher._ownership_palette == _default_palette()
    assert enricher._ownership_min_fraction == 0.02


def test_config_file_threads_perception_knob_to_adapter(tmp_path: Path) -> None:
    # A value set in the config file must reach the constructed adapter.
    path = tmp_path / "config.json"
    path.write_text('{"perception": {"ownership_min_fraction": 0.25}}', encoding="utf-8")

    config = load_config(default_config(), path, env={})
    enricher = ClassicalEntityEnricher.from_settings(config.perception)

    assert enricher._ownership_min_fraction == 0.25


def test_env_overrides_nf3_target() -> None:
    config = load_config(default_config(), env={"ZAE_THRESHOLDS__TRACKING_MOTA_TARGET": "0.9"})
    assert EvalConfig.from_thresholds(config.thresholds).tracking_mota_min == 0.9


def test_config_round_trips_with_new_sections(tmp_path: Path) -> None:
    original = default_config()
    path = tmp_path / "config.json"
    path.write_text(original.model_dump_json(indent=2), encoding="utf-8")
    assert load_config(default_config(), path, env={}) == original
