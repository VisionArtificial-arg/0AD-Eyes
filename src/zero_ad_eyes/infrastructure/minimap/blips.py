"""D2 — Blip detection + owner-colour classification (REQUIREMENTS.md §4.4, CV-30).

A blip is a small, saturated coloured dot the engine paints on the minimap for a
unit/building. Detection is a connected-component pass over a per-pixel
nearest-palette-colour label map:

1. Label every active pixel with its nearest :class:`PaletteEntry` (or background if
   no entry is within ``tolerance``) — this both *finds* candidate pixels and
   *classifies* their owner in one step, and prevents a blip being double-counted by
   two overlapping colour thresholds.
2. Connected-component analysis per label; keep components whose area is in
   ``[min_area, max_area]`` — the upper bound is what separates a point-like blip
   from a filled territory region (D3).
3. Each surviving component's centroid is projected to world space (D6).

Area thresholds and colour tolerance are configurable (NF7).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.minimap import Blip
from zero_ad_eyes.domain.taxonomy import Ownership

from .palette import MinimapPalette
from .projector import MinimapProjector
from .segmentation import Segmentation


@dataclass(frozen=True)
class BlipDetector:
    """Finds coloured blips and classifies each by owner colour (D2)."""

    palette: MinimapPalette
    tolerance: float = 70.0
    min_area: int = 1
    max_area: int = 60
    confidence: float = 0.8

    @classmethod
    def with_default_palette(cls) -> BlipDetector:
        return cls(palette=MinimapPalette.default())

    def detect(self, segmentation: Segmentation, projector: MinimapProjector) -> tuple[Blip, ...]:
        labels = self._label_pixels(segmentation)
        blips: list[Blip] = []
        for index, entry in enumerate(self.palette.entries):
            entry_mask = (labels == index).astype(np.uint8)
            blips.extend(self._components_to_blips(entry_mask, entry.ownership, projector))
        return tuple(blips)

    def _label_pixels(self, segmentation: Segmentation) -> np.ndarray:
        """Per-pixel nearest-palette index; ``-1`` for background/off-tolerance."""

        region = segmentation.region.astype(np.float32)
        h, w = region.shape[:2]
        flat = region.reshape(-1, 3)
        colors = self.palette.colors()  # (K, 3)
        # (N, K) distances via broadcasting.
        distances = np.linalg.norm(flat[:, None, :] - colors[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        best_distance = distances[np.arange(distances.shape[0]), nearest]
        labels = np.where(best_distance <= self.tolerance, nearest, -1)
        labels = labels.reshape(h, w)
        active = segmentation.mask > 0
        labels[~active] = -1
        return labels

    def _components_to_blips(
        self, entry_mask: np.ndarray, ownership: Ownership, projector: MinimapProjector
    ) -> list[Blip]:
        count, _, stats, centroids = cv2.connectedComponentsWithStats(entry_mask, connectivity=8)
        blips: list[Blip] = []
        for component in range(1, count):  # 0 is background
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue
            cx, cy = float(centroids[component][0]), float(centroids[component][1])
            blips.append(
                Blip(
                    world_pos=projector.to_world(cx, cy),
                    ownership=ownership,
                    confidence=Confidence(value=self.confidence, provenance=Provenance.CLASSICAL),
                )
            )
        return blips
