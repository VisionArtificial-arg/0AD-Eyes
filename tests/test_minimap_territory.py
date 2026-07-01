"""D3 — TerritoryExtractor region + border extraction tests (synthetic minimap)."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.infrastructure.minimap import (
    MinimapPalette,
    MinimapProjector,
    Segmentation,
    TerritoryExtractor,
    WorldExtent,
)


def _full_segmentation(region: np.ndarray) -> Segmentation:
    h, w = region.shape[:2]
    return Segmentation(
        region=region, mask=np.full((h, w), 255, dtype=np.uint8), origin_x=0, origin_y=0
    )


def _projector(seg: Segmentation) -> MinimapProjector:
    return MinimapProjector(
        region_width=seg.width, region_height=seg.height, extent=WorldExtent.square(100.0)
    )


def test_extracts_two_player_territories() -> None:
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    palette = MinimapPalette.default()
    self_c = palette.entries[0].color
    enemy_c = palette.entries[2].color
    region[5:35, 5:35] = (self_c.b, self_c.g, self_c.r)
    region[45:75, 45:75] = (enemy_c.b, enemy_c.g, enemy_c.r)

    seg = _full_segmentation(region)
    territory = TerritoryExtractor.with_default_palette().extract(seg, _projector(seg))

    owners = sorted(r.ownership for r in territory.regions)
    assert owners == sorted([Ownership.SELF, Ownership.ENEMY])
    assert all(r.area >= 64 for r in territory.regions)


def test_borders_trace_the_territory_edge() -> None:
    region = np.zeros((80, 80, 3), dtype=np.uint8)
    ally = MinimapPalette.default().entries[1].color
    region[20:60, 20:60] = (ally.b, ally.g, ally.r)

    seg = _full_segmentation(region)
    territory = TerritoryExtractor.with_default_palette().extract(seg, _projector(seg))

    assert territory.borders.shape == (80, 80)
    # The border is a hollow ring: edge pixels are set, the interior is not.
    assert int(territory.borders[20, 40]) == 255
    assert int(territory.borders[40, 40]) == 0


def test_no_territory_yields_empty_map() -> None:
    region = np.zeros((40, 40, 3), dtype=np.uint8)
    seg = _full_segmentation(region)
    territory = TerritoryExtractor.with_default_palette().extract(seg, _projector(seg))
    assert territory.regions == ()
    assert not territory.borders.any()
