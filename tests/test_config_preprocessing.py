"""Config-externalization guards for the preprocessing variants (Approach B, P3).

Golden: default settings reproduce the historical chain parameters. Threading: a
setting reaches the constructed step. (The offline pipeline uses no preprocessing by
design; these factories are the tuned chains whose params are now config-driven.)
"""

from __future__ import annotations

from zero_ad_eyes.application.settings import HudPipelineSettings, ScenePipelineSettings
from zero_ad_eyes.infrastructure.preprocessing import hud_pipeline, scene_pipeline
from zero_ad_eyes.infrastructure.preprocessing.noise import GaussianBlur


def test_default_settings_reproduce_chain_shapes() -> None:
    assert len(hud_pipeline().steps) == 2
    assert len(scene_pipeline().steps) == 3


def test_hud_default_settings_match_historical_values() -> None:
    s = HudPipelineSettings()
    assert (s.gaussian_ksize, s.clahe_clip_limit, s.clahe_tile) == (3, 2.0, (8, 8))


def test_scene_default_settings_match_historical_values() -> None:
    s = ScenePipelineSettings()
    assert (s.bilateral_diameter, s.bilateral_sigma_color, s.bilateral_sigma_space) == (
        5,
        50.0,
        50.0,
    )
    assert s.clahe_clip_limit == 3.0


def test_settings_thread_into_constructed_step() -> None:
    pipeline = hud_pipeline(settings=HudPipelineSettings(gaussian_ksize=5))
    blur = pipeline.steps[0]
    assert isinstance(blur, GaussianBlur)
    assert blur._ksize == 5
