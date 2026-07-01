"""Intensity normalization (P3, CV-04).

Different maps, times of day, and graphics settings shift the same object across a
wide brightness/contrast range. Normalization pins each frame to a stable
intensity envelope so downstream thresholds and the model see a consistent input,
regardless of render variation.

Two complementary tools:
- :class:`MinMaxNormalize` — data-driven full-range stretch (per frame).
- :class:`BrightnessContrast` — a fixed affine ``alpha * x + beta`` when a known,
  reproducible adjustment is wanted instead of a per-frame stretch.
"""

from __future__ import annotations

import numpy as np

from .base import Image, ImageStep


class MinMaxNormalize(ImageStep):
    """Linearly stretch intensities to ``[out_min, out_max]`` (P3).

    Operates on the whole array. A flat frame (min == max) is left unchanged rather
    than dividing by zero, keeping the step total.
    """

    def __init__(self, out_min: float = 0.0, out_max: float = 255.0) -> None:
        if out_min >= out_max:
            raise ValueError("out_min must be strictly less than out_max")
        self._out_min = out_min
        self._out_max = out_max

    def transform(self, image: Image) -> Image:
        lo = float(image.min())
        hi = float(image.max())
        if hi <= lo:
            return image.copy()
        scale = (self._out_max - self._out_min) / (hi - lo)
        stretched = (image.astype(np.float32) - lo) * scale + self._out_min
        return np.clip(stretched, 0, 255).astype(np.uint8)


class BrightnessContrast(ImageStep):
    """Apply a fixed affine ``alpha * x + beta`` with saturation (P3).

    ``alpha`` scales contrast (1.0 = unchanged), ``beta`` shifts brightness in
    intensity units. Results are clipped to the ``uint8`` range.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0) -> None:
        if alpha < 0.0:
            raise ValueError("alpha (contrast gain) must be non-negative")
        self._alpha = alpha
        self._beta = beta

    def transform(self, image: Image) -> Image:
        scaled = image.astype(np.float32) * self._alpha + self._beta
        return np.clip(scaled, 0, 255).astype(np.uint8)
