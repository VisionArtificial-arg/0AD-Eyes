"""EPIC C — HUD parsing infrastructure (classical, model-free).

Public surface: the :class:`ClassicalHudReader` adapter (satisfies the
``HudReader`` port), the injectable :class:`OcrEngine` seam and its production
``TesseractOcrEngine``, and the configurable region layouts. OCR text parsing
lives in :mod:`parsing` and is importable for direct unit testing.
"""

from __future__ import annotations

from .layout import FractionalRegion, SelectionPanelLayout, TopBarLayout
from .ocr import OcrEngine, TesseractOcrEngine
from .reader import ClassicalHudReader
from .selection import HealthReading, SelectionState

__all__ = [
    "ClassicalHudReader",
    "FractionalRegion",
    "HealthReading",
    "OcrEngine",
    "SelectionPanelLayout",
    "SelectionState",
    "TesseractOcrEngine",
    "TopBarLayout",
]
