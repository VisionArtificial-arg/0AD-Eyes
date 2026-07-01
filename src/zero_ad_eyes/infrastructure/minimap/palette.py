"""D2 — Configurable player-colour palette → ``Ownership`` map (REQUIREMENTS.md §4.4).

Owner classification on the minimap is a nearest-colour lookup, and the mapping
from a player colour to an :class:`Ownership` is **session state**, not a constant:
0 A.D. lets the player pick colours and the colour-blind palettes reshuffle them
(risk R4). So the palette is a first-class, configurable object calibrated per
session (EPIC B/B3) rather than literals scattered through the detector.

The :meth:`MinimapPalette.default` palette is *illustrative* (a conventional
blue-self / green-ally / red-enemy / white-gaia scheme) to make the reader runnable
out of the box; production wiring replaces it with the calibrated one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zero_ad_eyes.domain.taxonomy import Ownership


@dataclass(frozen=True)
class BgrColor:
    """A colour in OpenCV's BGR byte order."""

    b: int
    g: int
    r: int

    def as_array(self) -> np.ndarray:
        return np.array([self.b, self.g, self.r], dtype=np.float32)


@dataclass(frozen=True)
class PaletteEntry:
    """One calibrated player colour and the ownership it denotes."""

    label: str
    color: BgrColor
    ownership: Ownership


@dataclass(frozen=True)
class MinimapPalette:
    """An ordered set of colour→ownership entries with nearest-colour lookup."""

    entries: tuple[PaletteEntry, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("a palette needs at least one entry")

    @classmethod
    def default(cls) -> MinimapPalette:
        return cls(
            entries=(
                PaletteEntry("self", BgrColor(235, 90, 40), Ownership.SELF),  # blue
                PaletteEntry("ally", BgrColor(60, 200, 60), Ownership.ALLY),  # green
                PaletteEntry("enemy", BgrColor(40, 40, 220), Ownership.ENEMY),  # red
                PaletteEntry("gaia", BgrColor(235, 235, 235), Ownership.GAIA),  # white/neutral
            )
        )

    def colors(self) -> np.ndarray:
        """Entry colours stacked as a ``(K, 3)`` float array (BGR)."""

        return np.stack([entry.color.as_array() for entry in self.entries])

    def classify(self, bgr: tuple[int, int, int], tolerance: float) -> PaletteEntry | None:
        """Return the nearest entry within ``tolerance`` (Euclidean, BGR), else ``None``."""

        sample = np.array(bgr, dtype=np.float32)
        distances = np.linalg.norm(self.colors() - sample, axis=1)
        best = int(np.argmin(distances))
        if float(distances[best]) > tolerance:
            return None
        return self.entries[best]
