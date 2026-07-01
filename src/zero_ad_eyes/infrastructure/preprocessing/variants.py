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

from zero_ad_eyes.domain.geometry import ScreenBBox

from .base import PreprocessStep
from .contrast import ClaheContrast
from .noise import BilateralFilter, GaussianBlur
from .normalize import MinMaxNormalize
from .pipeline import PreprocessingPipeline
from .roi import GateMode, RoiGate


def hud_pipeline(region: ScreenBBox | None = None) -> PreprocessingPipeline:
    """A HUD-tuned preprocessing chain (P1).

    When ``region`` is given, the frame is masked to it first so downstream HUD
    parsing ignores the scene.
    """

    steps: list[PreprocessStep] = []
    if region is not None:
        steps.append(RoiGate(region, mode=GateMode.MASK))
    steps.append(GaussianBlur(ksize=3))
    steps.append(ClaheContrast(clip_limit=2.0, tile_grid_size=(8, 8)))
    return PreprocessingPipeline(steps)


def scene_pipeline() -> PreprocessingPipeline:
    """A scene-tuned preprocessing chain (P1)."""

    return PreprocessingPipeline(
        [
            BilateralFilter(diameter=5, sigma_color=50.0, sigma_space=50.0),
            MinMaxNormalize(),
            ClaheContrast(clip_limit=3.0, tile_grid_size=(8, 8)),
        ]
    )
