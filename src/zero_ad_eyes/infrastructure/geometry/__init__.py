"""Camera & coordinate geometry adapters (EPIC F).

This package turns screen detections into world coordinates and back under the
ground-plane assumption (REQUIREMENTS.md §4.6, CV-22..26):

- ``Homography`` — the planar projective primitive (F1; CV-23/CV-24).
- ``CameraProjector`` — the cohesive screen ⇄ world API (F1), camera-motion
  updates across frames (F2), and projection-error confidence (F4).
- geometric-transform builders for pan / zoom / rotation (F2; CV-25).
- ``reconcile`` — fuses a main-view and a minimap world estimate (F3; CV-26).

It imports only from ``zero_ad_eyes.domain`` and this package; it holds no port of
its own but exposes ``CameraProjector`` as its public surface.
"""

from __future__ import annotations

from .fusion import reconcile
from .homography import DegenerateHomographyError, Homography
from .projector import CameraProjector
from .transforms import chain, rotation, scaling, translation

__all__ = [
    "CameraProjector",
    "DegenerateHomographyError",
    "Homography",
    "chain",
    "reconcile",
    "rotation",
    "scaling",
    "translation",
]
