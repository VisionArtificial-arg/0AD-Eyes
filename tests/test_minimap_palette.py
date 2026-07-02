"""D2 — MinimapPalette nearest-colour owner classification tests."""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.infrastructure.minimap import BgrColor, MinimapPalette, PaletteEntry


def _default_palette() -> MinimapPalette:
    return MinimapPalette(
        entries=(
            PaletteEntry("self", BgrColor(235, 90, 40), Ownership.SELF),
            PaletteEntry("ally", BgrColor(60, 200, 60), Ownership.ALLY),
            PaletteEntry("enemy", BgrColor(40, 40, 220), Ownership.ENEMY),
            PaletteEntry("gaia", BgrColor(235, 235, 235), Ownership.GAIA),
        )
    )


def test_default_palette_maps_the_four_ownerships() -> None:
    palette = _default_palette()
    owners = {entry.ownership for entry in palette.entries}
    assert {Ownership.SELF, Ownership.ALLY, Ownership.ENEMY, Ownership.GAIA} <= owners


def test_classify_returns_nearest_entry() -> None:
    palette = _default_palette()
    # A near-red pixel should classify as the enemy entry.
    entry = palette.classify((30, 35, 210), tolerance=80.0)
    assert entry is not None
    assert entry.ownership is Ownership.ENEMY


def test_classify_rejects_when_beyond_tolerance() -> None:
    palette = MinimapPalette(entries=(PaletteEntry("self", BgrColor(255, 0, 0), Ownership.SELF),))
    assert palette.classify((0, 255, 0), tolerance=10.0) is None


def test_empty_palette_is_rejected() -> None:
    with pytest.raises(ValueError):
        MinimapPalette(entries=())
