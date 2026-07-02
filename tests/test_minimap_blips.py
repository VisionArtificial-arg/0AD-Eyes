"""D2 — BlipDetector detection + owner classification tests (synthetic minimap)."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.infrastructure.minimap import (
    BgrColor,
    BlipDetector,
    MinimapPalette,
    MinimapProjector,
    PaletteEntry,
    Segmentation,
    WorldExtent,
)


def _palette() -> MinimapPalette:
    return MinimapPalette(
        entries=(
            PaletteEntry("self", BgrColor(235, 90, 40), Ownership.SELF),
            PaletteEntry("ally", BgrColor(60, 200, 60), Ownership.ALLY),
            PaletteEntry("enemy", BgrColor(40, 40, 220), Ownership.ENEMY),
            PaletteEntry("gaia", BgrColor(235, 235, 235), Ownership.GAIA),
        )
    )


def _detector() -> BlipDetector:
    return BlipDetector(palette=_palette(), tolerance=70.0, min_area=1, max_area=60, confidence=0.8)


def _full_segmentation(region: np.ndarray) -> Segmentation:
    h, w = region.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    return Segmentation(region=region, mask=mask, origin_x=0, origin_y=0)


def _projector(seg: Segmentation) -> MinimapProjector:
    return MinimapProjector(
        region_width=seg.width, region_height=seg.height, extent=WorldExtent.square(100.0)
    )


def test_detects_one_blip_per_coloured_dot_with_owner() -> None:
    region = np.zeros((64, 64, 3), dtype=np.uint8)
    palette = _palette()
    self_bgr = palette.entries[0].color
    enemy_bgr = palette.entries[2].color
    cv2.circle(region, (16, 16), 3, (self_bgr.b, self_bgr.g, self_bgr.r), thickness=-1)
    cv2.circle(region, (48, 40), 3, (enemy_bgr.b, enemy_bgr.g, enemy_bgr.r), thickness=-1)

    seg = _full_segmentation(region)
    blips = _detector().detect(seg, _projector(seg))

    owners = sorted(blip.ownership for blip in blips)
    assert len(blips) == 2
    assert owners == sorted([Ownership.SELF, Ownership.ENEMY])
    for blip in blips:
        assert blip.confidence.provenance.value == "classical"


def test_blip_world_position_matches_projected_centroid() -> None:
    region = np.zeros((64, 64, 3), dtype=np.uint8)
    enemy = _palette().entries[2].color
    cv2.circle(region, (48, 16), 3, (enemy.b, enemy.g, enemy.r), thickness=-1)

    seg = _full_segmentation(region)
    proj = _projector(seg)
    (blip,) = _detector().detect(seg, proj)

    expected = proj.to_world(48.0, 16.0)
    assert abs(blip.world_pos.x - expected.x) < 1.0
    assert abs(blip.world_pos.y - expected.y) < 1.0


def test_large_filled_region_is_not_a_blip() -> None:
    # A territory-sized block exceeds max_area and must be ignored by the blip pass.
    region = np.zeros((64, 64, 3), dtype=np.uint8)
    ally = _palette().entries[1].color
    region[5:40, 5:40] = (ally.b, ally.g, ally.r)

    seg = _full_segmentation(region)
    blips = _detector().detect(seg, _projector(seg))
    assert blips == ()


def test_mask_excludes_blips_outside_active_area() -> None:
    region = np.zeros((64, 64, 3), dtype=np.uint8)
    enemy = _palette().entries[2].color
    cv2.circle(region, (5, 5), 3, (enemy.b, enemy.g, enemy.r), thickness=-1)

    h, w = region.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:, 30:] = 255  # corner blip falls outside the active mask
    seg = Segmentation(region=region, mask=mask, origin_x=0, origin_y=0)

    blips = _detector().detect(seg, _projector(seg))
    assert blips == ()
