"""D6 — MinimapProjector round-trip and convention tests."""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.infrastructure.minimap import MinimapProjector, WorldExtent


def test_corners_map_to_world_extent_with_flipped_y() -> None:
    proj = MinimapProjector(region_width=200, region_height=100, extent=WorldExtent.square(1024.0))

    top_left = proj.to_world(0.0, 0.0)
    bottom_right = proj.to_world(200.0, 100.0)

    # x grows east; y grows north, so the top-left pixel is the *north*-west world corner.
    assert top_left == WorldPoint(x=0.0, y=1024.0)
    assert bottom_right == WorldPoint(x=1024.0, y=0.0)


def test_flip_y_off_keeps_y_growing_downward() -> None:
    proj = MinimapProjector(
        region_width=10, region_height=10, extent=WorldExtent(width=10.0, height=10.0, flip_y=False)
    )
    assert proj.to_world(0.0, 0.0) == WorldPoint(x=0.0, y=0.0)
    assert proj.to_world(10.0, 10.0) == WorldPoint(x=10.0, y=10.0)


def test_to_world_and_to_pixel_are_inverse() -> None:
    proj = MinimapProjector(
        region_width=128, region_height=96, extent=WorldExtent(origin_x=5.0, origin_y=-3.0)
    )
    for px, py in [(0.0, 0.0), (40.0, 12.0), (128.0, 96.0), (63.5, 47.25)]:
        back = proj.to_pixel(proj.to_world(px, py))
        assert back[0] == pytest.approx(px)
        assert back[1] == pytest.approx(py)


def test_configurable_origin_offsets_the_world() -> None:
    proj = MinimapProjector(
        region_width=100,
        region_height=100,
        extent=WorldExtent(origin_x=100.0, origin_y=200.0, width=50.0, height=50.0, flip_y=False),
    )
    assert proj.to_world(0.0, 0.0) == WorldPoint(x=100.0, y=200.0)


def test_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError):
        MinimapProjector(region_width=0, region_height=10)
    with pytest.raises(ValueError):
        WorldExtent(width=0.0, height=10.0)
