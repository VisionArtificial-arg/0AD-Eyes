"""Calibration & UI-layout adapters (EPIC B).

This package implements the ``Calibrator`` port (REQUIREMENTS.md §5 EPIC B) as a
pure-pixel adapter: given a ``Frame`` it produces a ``Calibration`` locating the
0 A.D. HUD regions (top bar, minimap, selection panel).

Per A4/A5 the layout is *discovered per session*, never hard-coded to one
resolution: region boxes are derived from frame width/height/ui_scale (optionally
refined by anchor detection), and profiles are persisted keyed by resolution+theme
for reuse across sessions.
"""

from __future__ import annotations

from .layout import HudCalibrator
from .profiles import CalibrationProfileStore
from .ratios import HudLayoutRatios
from .selfcheck import CalibrationCheck, LayoutSelfCheck

__all__ = [
    "CalibrationCheck",
    "CalibrationProfileStore",
    "HudCalibrator",
    "HudLayoutRatios",
    "LayoutSelfCheck",
]
