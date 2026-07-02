"""Per-consumer pipeline variants (P1).

The same steps compose into different chains tuned for different consumers. These
factories encode two sensible starting points; they are deliberately plain
functions returning a fresh :class:`PreprocessingPipeline`, so a caller can inspect
``.steps`` or build its own chain instead.

- :func:`hud_pipeline` — the HUD is crisp, high-contrast UI. Optionally gate to the
  HUD region, then a gentle denoise plus CLAHE to lift faint icons/text; no colour
  conversion, since HUD readers work in BGR.
- :func:`scene_pipeline` — the 3D scene is noisy and lit inconsistently. Edge-
  preserving denoise, min-max normalization to tame render variation, and CLAHE to
  surface small units.
"""

from __future__ import annotations

from zero_ad_eyes.application.settings import HudPipelineSettings, ScenePipelineSettings
from zero_ad_eyes.domain.geometry import ScreenBBox

from .base import PreprocessStep
from .contrast import ClaheContrast
from .noise import BilateralFilter, GaussianBlur
from .normalize import MinMaxNormalize
from .pipeline import PreprocessingPipeline
from .roi import GateMode, RoiGate


def hud_pipeline(
    region: ScreenBBox | None = None, *, settings: HudPipelineSettings | None = None
) -> PreprocessingPipeline:
    """A HUD-tuned preprocessing chain (P1), parameterised by config (NF7).

    When ``region`` is given, the frame is masked to it first so downstream HUD
    parsing ignores the scene. Default ``settings`` reproduce the historical chain.
    """

    cfg = settings or HudPipelineSettings()
    steps: list[PreprocessStep] = []
    if region is not None:
        steps.append(RoiGate(region, mode=GateMode.MASK))
    steps.append(GaussianBlur(ksize=cfg.gaussian_ksize))
    steps.append(ClaheContrast(clip_limit=cfg.clahe_clip_limit, tile_grid_size=cfg.clahe_tile))
    return PreprocessingPipeline(steps)


def scene_pipeline(*, settings: ScenePipelineSettings | None = None) -> PreprocessingPipeline:
    """A scene-tuned preprocessing chain (P1), parameterised by config (NF7)."""

    cfg = settings or ScenePipelineSettings()
    return PreprocessingPipeline(
        [
            BilateralFilter(
                diameter=cfg.bilateral_diameter,
                sigma_color=cfg.bilateral_sigma_color,
                sigma_space=cfg.bilateral_sigma_space,
            ),
            MinMaxNormalize(),
            ClaheContrast(clip_limit=cfg.clahe_clip_limit, tile_grid_size=cfg.clahe_tile),
        ]
    )
