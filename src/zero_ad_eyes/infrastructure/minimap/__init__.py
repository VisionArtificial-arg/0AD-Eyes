"""Minimap interpretation adapters (REQUIREMENTS.md EPIC D).

This package is the classical (deterministic, pixel-only) implementation of the
``MinimapReader`` port. The minimap is 0 A.D.'s high-signal, 2D position source
(risk R1/R2 mitigation): it is interpreted independently of the main viewport so a
projection error in one cannot silently corrupt the other.

Collaborators (each an autonomous object, composed by :class:`ClassicalMinimapReader`):

- :class:`MinimapSegmenter` (D1)      — crop + mask the active minimap region.
- :class:`MinimapPalette` (D2)        — configurable player-colour → ``Ownership`` map.
- :class:`BlipDetector` (D2)          — coloured dots → world-space blips.
- :class:`TerritoryExtractor` (D3)    — player territory / border regions.
- :class:`FogClassifier` (D4)         — per-cell unexplored/explored/visible grid.
- :class:`ViewportDetector` (D5)      — camera footprint rectangle.
- :class:`MinimapProjector` (D6)      — minimap-pixel ⇄ ``WorldPoint`` mapping.

Everything this package emits is tagged ``Provenance.CLASSICAL``.
"""

from __future__ import annotations

from .blips import BlipDetector
from .fog import FogClassifier, FogGrid
from .palette import BgrColor, MinimapPalette, PaletteEntry
from .projector import MinimapProjector, WorldExtent
from .reader import ClassicalMinimapReader
from .segmentation import MinimapSegmenter, MinimapShape, Segmentation
from .territory import TerritoryExtractor, TerritoryMap, TerritoryRegion
from .viewport import ViewportDetector

__all__ = [
    "BgrColor",
    "BlipDetector",
    "ClassicalMinimapReader",
    "FogClassifier",
    "FogGrid",
    "MinimapPalette",
    "MinimapProjector",
    "MinimapSegmenter",
    "MinimapShape",
    "PaletteEntry",
    "Segmentation",
    "TerritoryExtractor",
    "TerritoryMap",
    "TerritoryRegion",
    "ViewportDetector",
    "WorldExtent",
]
