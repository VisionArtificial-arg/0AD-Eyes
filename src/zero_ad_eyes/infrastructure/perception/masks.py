"""E10 — Mask post-processing (CV-30 / CV-31 / CV-32).

Connected-component analysis, contour detection, and morphological open/close
to clean binary masks. Every function is a pure transform over a uint8 mask in
the OpenCV convention (background 0, foreground 255); no learned model is
involved, so anything derived here is ``Provenance.CLASSICAL`` by construction.

These are the shared building blocks the other classical tasks (E3 ownership,
E6a resources) lean on to turn a raw colour threshold into clean instances.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from zero_ad_eyes.domain.geometry import ScreenBBox, ScreenPoint


@dataclass(frozen=True)
class Component:
    """One connected foreground blob: its bbox, pixel area, and centroid."""

    label: int
    bbox: ScreenBBox
    area: int
    centroid: ScreenPoint


def to_binary(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Coerce any single-channel image to a 0/255 uint8 mask."""

    if mask.ndim != 2:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
    _, binary = cv2.threshold(gray.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    return binary


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def morphological_open(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Erode-then-dilate: remove speckle noise while preserving blob extent."""

    return cv2.morphologyEx(
        to_binary(mask), cv2.MORPH_OPEN, _kernel(ksize), iterations=max(1, iterations)
    )


def morphological_close(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Dilate-then-erode: fill small holes and bridge gaps inside a blob."""

    return cv2.morphologyEx(
        to_binary(mask), cv2.MORPH_CLOSE, _kernel(ksize), iterations=max(1, iterations)
    )


def clean_mask(mask: np.ndarray, open_ksize: int = 3, close_ksize: int = 5) -> np.ndarray:
    """Denoise (open) then consolidate (close) — the usual mask-tidy pipeline."""

    opened = morphological_open(mask, ksize=open_ksize)
    return morphological_close(opened, ksize=close_ksize)


def connected_components(mask: np.ndarray, min_area: int = 1) -> tuple[Component, ...]:
    """Group foreground pixels into labelled blobs (CV-30), background dropped.

    Components with area below ``min_area`` are discarded, giving callers a cheap
    way to suppress residual noise after morphology.
    """

    binary = to_binary(mask)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[Component] = []
    for label in range(1, count):  # 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = float(stats[label, cv2.CC_STAT_LEFT])
        y = float(stats[label, cv2.CC_STAT_TOP])
        w = float(stats[label, cv2.CC_STAT_WIDTH])
        h = float(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[label][0]), float(centroids[label][1])
        components.append(
            Component(
                label=label,
                bbox=ScreenBBox(x=x, y=y, width=w, height=h),
                area=area,
                centroid=ScreenPoint(x=cx, y=cy),
            )
        )
    components.sort(key=lambda c: c.area, reverse=True)
    return tuple(components)


def find_contours(mask: np.ndarray, min_area: float = 0.0) -> tuple[tuple[ScreenPoint, ...], ...]:
    """Outline every foreground region (CV-31) as a polygon of ``ScreenPoint``.

    Only external contours are returned (holes are ignored); polygons whose
    enclosed area is below ``min_area`` are dropped.
    """

    binary = to_binary(mask)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[tuple[ScreenPoint, ...]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = tuple(ScreenPoint(x=float(p[0][0]), y=float(p[0][1])) for p in contour)
        if points:
            polygons.append(points)
    return tuple(polygons)
