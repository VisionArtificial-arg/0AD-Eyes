"""Tests for HUD region geometry (EPIC B — B2).

Region math is exercised directly on ``HudLayoutRatios`` (no frame/detector), so
these assertions pin the resolution-relative layout: ratios and bounds only, never
absolute per-resolution pixel constants (A4).
"""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.calibration import HudLayoutRatios


def _within(box: ScreenBBox, width: int, height: int) -> bool:
    return (
        box.x >= 0.0
        and box.y >= 0.0
        and box.x + box.width <= width + 1e-6
        and box.y + box.height <= height + 1e-6
    )


def test_ratio_fallback_regions_are_resolution_relative() -> None:
    width, height, ui = 1000, 800, 1.0
    ratios = HudLayoutRatios(
        top_bar_height=0.035, minimap_side=0.20, selection_width=0.34, selection_height=0.16
    )

    top = ratios.top_bar(width, height, ui, None)
    minimap = ratios.minimap(width, height, ui, None)
    panel = ratios.selection_panel(width, height, ui, None)

    # Top bar spans the full width along the top edge.
    assert (top.x, top.y) == (0.0, 0.0)
    assert top.width == pytest.approx(width)
    assert top.height == pytest.approx(ratios.top_bar_height * height)

    # Minimap is a square anchored bottom-left.
    side = ratios.minimap_side * height
    assert minimap.x == 0.0
    assert minimap.y == pytest.approx(height - side)
    assert minimap.width == pytest.approx(side)
    assert minimap.height == pytest.approx(side)

    # Selection panel is anchored bottom and horizontally centred.
    assert panel.y + panel.height == pytest.approx(height)
    assert panel.center.x == pytest.approx(width / 2.0)

    for box in (top, minimap, panel):
        assert _within(box, width, height)


def test_regions_scale_with_ui_scale() -> None:
    width, height = 1000, 800
    ratios = HudLayoutRatios(
        top_bar_height=0.035, minimap_side=0.20, selection_width=0.34, selection_height=0.16
    )
    small = ratios.minimap(width, height, 1.0, None)
    large = ratios.minimap(width, height, 2.0, None)
    assert large.height > small.height


def test_anchor_overrides_ratio_for_top_bar() -> None:
    width, height = 1280, 720
    ratios = HudLayoutRatios(
        top_bar_height=0.035, minimap_side=0.20, selection_width=0.34, selection_height=0.16
    )
    anchored = ratios.top_bar(width, height, 1.0, 0.08)
    assert anchored.height == pytest.approx(0.08 * height)


def test_regions_stay_within_frame_at_extreme_ui_scale() -> None:
    width, height = 640, 480
    ratios = HudLayoutRatios(
        top_bar_height=0.035, minimap_side=0.20, selection_width=0.34, selection_height=0.16
    )
    for box in (
        ratios.top_bar(width, height, 5.0, None),
        ratios.minimap(width, height, 5.0, None),
        ratios.selection_panel(width, height, 5.0, None),
    ):
        assert _within(box, width, height)
