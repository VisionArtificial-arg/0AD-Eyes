"""Planar homography helper (F1; CV-23 perspective transform, CV-24 homography).

A ``Homography`` wraps a 3x3 projective matrix that maps points on one plane to
another under the ground-plane assumption (REQUIREMENTS.md §4.6, EPIC F). It is the
numeric primitive the ``CameraProjector`` (F1) is built on and the same primitive
that aligns the minimap grid to world space (F3, CV-26).

Design notes:
- Recovery from point correspondences uses OpenCV's ``cv2.findHomography`` (a
  least-squares / RANSAC planar fit).
- Point application is done with an explicit ``float64`` homogeneous multiply
  rather than ``cv2.perspectiveTransform`` (which is ``float32`` internally) so
  round-trips stay numerically tight and deterministic (NF5).
- The type is an immutable value object: transforms return *new* homographies,
  never mutate in place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
PointPairs = Sequence[tuple[float, float]]


class DegenerateHomographyError(ValueError):
    """Raised when a homography cannot be recovered or inverted."""


@dataclass(frozen=True, eq=False)
class Homography:
    """A 3x3 projective plane-to-plane map, normalised so ``matrix[2, 2] == 1``."""

    matrix: Matrix

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError(f"homography matrix must be 3x3, got {m.shape}")
        corner = m[2, 2]
        if corner != 0.0:
            m = m / corner
        # dataclass is frozen; bypass to store the normalised copy.
        object.__setattr__(self, "matrix", m)

    # -- construction ----------------------------------------------------

    @classmethod
    def identity(cls) -> Homography:
        """The map that leaves every point unchanged."""

        return cls(np.eye(3, dtype=np.float64))

    @classmethod
    def from_matrix(cls, matrix: NDArray[np.floating] | Sequence[Sequence[float]]) -> Homography:
        """Wrap an existing 3x3 matrix (validated and normalised)."""

        return cls(np.asarray(matrix, dtype=np.float64))

    @classmethod
    def from_correspondences(cls, src: PointPairs, dst: PointPairs) -> Homography:
        """Recover the homography mapping ``src`` points to ``dst`` points.

        At least four non-collinear correspondences are required (the minimum for
        a projective plane fit). Raises ``DegenerateHomographyError`` when the
        configuration is degenerate.
        """

        if len(src) != len(dst):
            raise ValueError("src and dst must have equal length")
        if len(src) < 4:
            raise ValueError("at least 4 correspondences are required")
        src_arr = np.asarray(src, dtype=np.float64)
        dst_arr = np.asarray(dst, dtype=np.float64)
        matrix, _mask = cv2.findHomography(src_arr, dst_arr, method=0)
        if matrix is None:
            raise DegenerateHomographyError("cv2.findHomography returned no solution")
        return cls(np.asarray(matrix, dtype=np.float64))

    # -- application -----------------------------------------------------

    def apply(self, points: NDArray[np.floating]) -> Matrix:
        """Map an ``(N, 2)`` array of points; returns an ``(N, 2)`` array."""

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        projected = homogeneous @ self.matrix.T
        w = projected[:, 2:3]
        # Guard against points mapped onto the line at infinity.
        w = np.where(w == 0.0, np.finfo(np.float64).eps, w)
        return projected[:, :2] / w

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        """Map a single point, returning a plain float pair."""

        out = self.apply(np.array([[x, y]], dtype=np.float64))[0]
        return float(out[0]), float(out[1])

    # -- algebra ---------------------------------------------------------

    def inverse(self) -> Homography:
        """The reverse map (dst plane back to src plane)."""

        try:
            inv = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as exc:  # singular matrix
            raise DegenerateHomographyError("homography is not invertible") from exc
        return Homography(inv)

    def compose(self, before: Homography) -> Homography:
        """Return the map that applies ``before`` first, then ``self``.

        ``self.compose(before).apply(p) == self.apply(before.apply(p))``.
        """

        return Homography(self.matrix @ before.matrix)

    # -- fit quality -----------------------------------------------------

    def reprojection_error(self, src: PointPairs, dst: PointPairs) -> float:
        """Root-mean-square distance between ``apply(src)`` and ``dst``.

        Expressed in the units of the ``dst`` plane. Zero means a perfect fit.
        """

        if len(src) != len(dst):
            raise ValueError("src and dst must have equal length")
        if not src:
            return 0.0
        predicted = self.apply(np.asarray(src, dtype=np.float64))
        target = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
        residuals = predicted - target
        return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
