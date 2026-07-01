"""Preprocessing adapters (EPIC P, REQUIREMENTS.md §5).

Conditions a raw ``Frame`` for downstream perception. The public surface is a set
of small, composable steps and a :class:`PreprocessingPipeline` that chains them —
the pipeline satisfies the application ``Preprocessor`` port (it exposes
``process``), while each step is also usable standalone. Per-consumer variants
(:func:`hud_pipeline`, :func:`scene_pipeline`) are pre-tuned chains.

Task coverage: P1 pipeline/variants, P2 colour spaces, P3 normalization, P4 noise
filtering, P5 CLAHE contrast, P6 edge detection, P7 ROI gating.
"""

from __future__ import annotations

from .base import ImageStep, PreprocessStep
from .color import ColorSpace, ColorSpaceConvert
from .contrast import ClaheContrast
from .edges import EdgeDetect, EdgeOperator, canny_edges, sobel_edges
from .noise import BilateralFilter, GaussianBlur, MedianBlur
from .normalize import BrightnessContrast, MinMaxNormalize
from .pipeline import PreprocessingPipeline
from .roi import GateMode, RoiGate
from .variants import hud_pipeline, scene_pipeline

__all__ = [
    "BilateralFilter",
    "BrightnessContrast",
    "ClaheContrast",
    "ColorSpace",
    "ColorSpaceConvert",
    "EdgeDetect",
    "EdgeOperator",
    "GateMode",
    "GaussianBlur",
    "ImageStep",
    "MedianBlur",
    "MinMaxNormalize",
    "PreprocessStep",
    "PreprocessingPipeline",
    "RoiGate",
    "canny_edges",
    "hud_pipeline",
    "scene_pipeline",
    "sobel_edges",
]
