"""Configuration package (REQUIREMENTS.md X3 / NF7).

Re-exports the typed :class:`Config` and its loader/saver so callers depend on the
package, not the module layout.
"""

from __future__ import annotations

from .config import (
    RGB,
    Config,
    FogPalette,
    OverlaySettings,
    OwnerPalette,
    Paths,
    Thresholds,
    load_config,
    save_config,
)

__all__ = [
    "RGB",
    "Config",
    "FogPalette",
    "OverlaySettings",
    "OwnerPalette",
    "Paths",
    "Thresholds",
    "load_config",
    "save_config",
]
