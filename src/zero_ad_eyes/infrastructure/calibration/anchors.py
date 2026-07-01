"""Pure-pixel HUD anchor detection (EPIC B — B1/B2 support, B4 support).

The 0 A.D. HUD draws opaque, low-variance bands over the 3D scene: a resource/phase
bar spanning the very top edge, and a control strip along the bottom edge (which
hosts the bottom-left minimap and the bottom-centre selection panel). Those bands
are the cheapest deterministic *anchors* we can recover from pixels alone.

These helpers detect the thickness of the top and bottom HUD bands as a *fraction*
of frame height. Fractions (never absolute pixels) keep detection resolution-
independent (A4) and let the calibrator both refine region boxes (B2) and estimate
UI scale (B1). Detection is intentionally conservative: when the signal is weak it
returns ``None`` so callers fall back to layout ratios rather than trusting noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# A HUD band is a contiguous run of rows, starting at a frame edge, whose rows are
# mutually similar (the opaque bar) and collectively distinct from the scene beyond
# the run. These thresholds are expressed in 8-bit intensity units.
_SIMILARITY_TOL = 12.0  # max mean-intensity delta between rows *inside* the band
_CONTRAST_MIN = 8.0  # min mean-intensity delta between the band and the row past it
_MAX_BAND_FRACTION = 0.20  # a HUD band never plausibly covers >20% of frame height


def _row_means(image: Any) -> np.ndarray:
    """Per-row mean intensity (grayscale-collapsed), as a float vector of length H."""

    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr.mean(axis=1).astype(np.float64)


def _leading_band_rows(row_means: np.ndarray) -> int:
    """Length (in rows) of the uniform band anchored at index 0 of ``row_means``.

    Returns 0 when no plausible band is present (no contrast, or the band would be
    implausibly thick), signalling the caller to fall back to ratios.
    """

    height = row_means.shape[0]
    if height < 4:
        return 0

    reference = float(row_means[0])
    limit = int(height * _MAX_BAND_FRACTION)
    run = 1
    while run < height and abs(float(row_means[run]) - reference) <= _SIMILARITY_TOL:
        run += 1

    if run <= 0 or run > limit:
        return 0
    if run >= height:
        return 0

    contrast = abs(float(row_means[run]) - reference)
    if contrast < _CONTRAST_MIN:
        return 0
    return run


def top_band_fraction(image: Any) -> float | None:
    """Fraction of frame height occupied by the top HUD band, or ``None``.

    ``None`` means "no confident anchor" — the caller should use ratio fallback.
    """

    means = _row_means(image)
    run = _leading_band_rows(means)
    if run == 0:
        return None
    return run / means.shape[0]


def bottom_band_fraction(image: Any) -> float | None:
    """Fraction of frame height occupied by the bottom HUD band, or ``None``.

    Implemented by detecting the leading band on the vertically-flipped rows.
    """

    means = _row_means(image)[::-1]
    run = _leading_band_rows(means)
    if run == 0:
        return None
    return run / means.shape[0]
