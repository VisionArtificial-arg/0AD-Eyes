"""Elementary geometric-transform builders for camera motion (F2; CV-25).

Between frames the RTS camera pans, zooms, and rotates. Each of those is an affine
screen-space transform expressible as a ``Homography`` (a homography with an affine
bottom row ``[0, 0, 1]``). Composing such a *screen motion* onto an existing
``CameraProjector`` re-parameterises the screen→world map without a fresh recovery
(REQUIREMENTS.md EPIC F / F2). These builders normalise scale, rotation, and
orientation (CV-25) into that common form.

All angles are in radians; all transforms answer immutable ``Homography`` values.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .homography import Homography


def translation(dx: float, dy: float) -> Homography:
    """A pure pan by ``(dx, dy)`` screen pixels."""

    return Homography.from_matrix([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def scaling(
    sx: float, sy: float | None = None, *, center: tuple[float, float] = (0.0, 0.0)
) -> Homography:
    """A zoom by ``sx`` (and ``sy``, defaulting to ``sx``) about ``center``."""

    if sy is None:
        sy = sx
    cx, cy = center
    core = Homography.from_matrix([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    return _about(core, cx, cy)


def rotation(theta: float, *, center: tuple[float, float] = (0.0, 0.0)) -> Homography:
    """A rotation by ``theta`` radians about ``center``."""

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cx, cy = center
    core = Homography.from_matrix([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    return _about(core, cx, cy)


def chain(transforms: Iterable[Homography]) -> Homography:
    """Compose transforms left-to-right: the first is applied to points first.

    ``chain([a, b, c]).apply(p) == c.apply(b.apply(a.apply(p)))``.
    """

    result = Homography.identity()
    for transform in transforms:
        result = transform.compose(result)
    return result


def _about(core: Homography, cx: float, cy: float) -> Homography:
    """Conjugate ``core`` so it acts about ``(cx, cy)`` instead of the origin."""

    if cx == 0.0 and cy == 0.0:
        return core
    to_origin = translation(-cx, -cy)
    back = translation(cx, cy)
    # apply: shift to origin, run core, shift back.
    matrix = back.matrix @ core.matrix @ to_origin.matrix
    return Homography(np.asarray(matrix, dtype=np.float64))
