"""D3 — Territory / border region extraction (REQUIREMENTS.md §4.4, CV-30/CV-32).

Territory shows on the minimap as broad, player-tinted areas (as opposed to the
point-like blips of D2). This stage thresholds each palette colour, keeps the
*large* connected components (area ≥ ``min_area`` — the mirror of the blip pass's
upper bound), and derives a border mask via a morphological gradient of the union of
all territory.

The domain :class:`MinimapModel` has no territory field, so results are returned as
an infrastructure value object (:class:`TerritoryMap`) for the fusion layer (EPIC G)
to consume; this stage never mutates the shared contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.taxonomy import Ownership

from .palette import MinimapPalette
from .projector import MinimapProjector
from .segmentation import Segmentation


@dataclass(frozen=True)
class TerritoryRegion:
    """One contiguous player-owned area on the minimap."""

    ownership: Ownership
    area: int
    bbox: ScreenBBox  # region-local minimap pixels
    world_center: WorldPoint


@dataclass(frozen=True)
class TerritoryMap:
    """All territory regions plus the border mask separating them."""

    regions: tuple[TerritoryRegion, ...]
    borders: np.ndarray  # HxW uint8, 255 = territory boundary pixel


@dataclass(frozen=True)
class TerritoryExtractor:
    """Segments broad player-tinted areas and their borders (D3)."""

    palette: MinimapPalette
    tolerance: float
    min_area: int

    def extract(self, segmentation: Segmentation, projector: MinimapProjector) -> TerritoryMap:
        region = segmentation.region.astype(np.float32)
        active = segmentation.mask > 0
        regions: list[TerritoryRegion] = []
        union = np.zeros(segmentation.mask.shape, dtype=np.uint8)

        for entry in self.palette.entries:
            entry_mask = self._color_mask(region, entry.color.as_array(), active)
            regions.extend(self._components(entry_mask, entry.ownership, projector, union))

        return TerritoryMap(regions=tuple(regions), borders=self._borders(union))

    def _color_mask(self, region: np.ndarray, color: np.ndarray, active: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(region - color, axis=2)
        return ((distance <= self.tolerance) & active).astype(np.uint8)

    def _components(
        self,
        entry_mask: np.ndarray,
        ownership: Ownership,
        projector: MinimapProjector,
        union: np.ndarray,
    ) -> list[TerritoryRegion]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            entry_mask, connectivity=8
        )
        found: list[TerritoryRegion] = []
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            union[labels == component] = 255
            x = int(stats[component, cv2.CC_STAT_LEFT])
            y = int(stats[component, cv2.CC_STAT_TOP])
            w = int(stats[component, cv2.CC_STAT_WIDTH])
            h = int(stats[component, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[component][0]), float(centroids[component][1])
            found.append(
                TerritoryRegion(
                    ownership=ownership,
                    area=area,
                    bbox=ScreenBBox(x=x, y=y, width=w, height=h),
                    world_center=projector.to_world(cx, cy),
                )
            )
        return found

    def _borders(self, union: np.ndarray) -> np.ndarray:
        if not union.any():
            return union
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(union, cv2.MORPH_GRADIENT, kernel)
