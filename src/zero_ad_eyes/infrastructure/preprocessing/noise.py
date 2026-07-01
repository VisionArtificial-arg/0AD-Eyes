"""Noise filtering (P4, CV-05).

Screen capture and the renderer introduce compression blocking, dithering, and
aliasing that mislead edge and contour detection. Each filter here removes a
different class of artifact:
- :class:`GaussianBlur` — generic high-frequency noise (edge-softening).
- :class:`MedianBlur` — salt-and-pepper / speckle (edge-preserving for impulses).
- :class:`BilateralFilter` — smooths flat regions while keeping strong edges,
  the best default before edge detection on UI-heavy frames.

All three share one small callable shape so they drop into a pipeline
interchangeably.
"""

from __future__ import annotations

import cv2

from .base import Image, ImageStep


class GaussianBlur(ImageStep):
    """Isotropic Gaussian smoothing (P4)."""

    def __init__(self, ksize: int = 3, sigma: float = 0.0) -> None:
        if ksize < 1 or ksize % 2 == 0:
            raise ValueError("ksize must be a positive odd integer")
        self._ksize = ksize
        self._sigma = sigma

    def transform(self, image: Image) -> Image:
        return cv2.GaussianBlur(image, (self._ksize, self._ksize), self._sigma)


class MedianBlur(ImageStep):
    """Median smoothing — removes impulse (salt-and-pepper) noise (P4)."""

    def __init__(self, ksize: int = 3) -> None:
        if ksize < 3 or ksize % 2 == 0:
            raise ValueError("ksize must be an odd integer >= 3")
        self._ksize = ksize

    def transform(self, image: Image) -> Image:
        return cv2.medianBlur(image, self._ksize)


class BilateralFilter(ImageStep):
    """Edge-preserving smoothing (P4).

    ``diameter`` is the neighbourhood size; ``sigma_color`` controls how much
    colour difference is blurred across, ``sigma_space`` the spatial reach.
    """

    def __init__(
        self,
        diameter: int = 5,
        sigma_color: float = 50.0,
        sigma_space: float = 50.0,
    ) -> None:
        if diameter < 1:
            raise ValueError("diameter must be positive")
        self._diameter = diameter
        self._sigma_color = sigma_color
        self._sigma_space = sigma_space

    def transform(self, image: Image) -> Image:
        return cv2.bilateralFilter(image, self._diameter, self._sigma_color, self._sigma_space)
