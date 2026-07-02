"""Screen ⇄ world projection under the ground-plane assumption (F1).

``CameraProjector`` is the cohesive API this feature exposes. It projects screen
detections to world coordinates and back through a recovered ground-plane
homography (REQUIREMENTS.md §4.6, EPIC F, CV-22), returning domain
``WorldPoint`` / ``ScreenPoint`` value objects.

The projector is immutable: it never mutates its homography. Camera motion across
frames (F2) is handled by messages that answer a *new* projector, and fit quality
is quantified and exposed as a ``Confidence`` (F4).
"""

from __future__ import annotations

import math

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.geometry import ScreenPoint, WorldPoint

from .homography import Homography


class CameraProjector:
    """Maps ``ScreenPoint`` ⇄ ``WorldPoint`` via a ground-plane homography.

    ``screen_to_world`` is the recovered map from screen pixels to world
    coordinates. ``reprojection_error`` is the RMS fit residual (world units)
    from recovery; ``error_tolerance`` is the world-unit scale at which that
    residual is treated as a 1/e drop in projection confidence (F4).
    """

    def __init__(
        self,
        screen_to_world: Homography,
        *,
        error_tolerance: float,
        reprojection_error: float = 0.0,
        provenance: Provenance = Provenance.CLASSICAL,
    ) -> None:
        if error_tolerance <= 0.0:
            raise ValueError("error_tolerance must be positive")
        self._screen_to_world = screen_to_world
        self._world_to_screen = screen_to_world.inverse()
        self._reprojection_error = float(reprojection_error)
        self._error_tolerance = float(error_tolerance)
        self._provenance = provenance

    # -- construction ----------------------------------------------------

    @classmethod
    def from_correspondences(
        cls,
        screen_points: list[ScreenPoint],
        world_points: list[WorldPoint],
        *,
        error_tolerance: float,
        provenance: Provenance = Provenance.CLASSICAL,
    ) -> CameraProjector:
        """Recover the projector from screen↔world correspondences (F1).

        The recovery residual is measured and retained so the resulting
        projector reports projection confidence against ``error_tolerance`` (F4).
        """

        src = [(p.x, p.y) for p in screen_points]
        dst = [(p.x, p.y) for p in world_points]
        homography = Homography.from_correspondences(src, dst)
        error = homography.reprojection_error(src, dst)
        return cls(
            homography,
            reprojection_error=error,
            error_tolerance=error_tolerance,
            provenance=provenance,
        )

    # -- projection ------------------------------------------------------

    def to_world(self, point: ScreenPoint) -> WorldPoint:
        """Project a screen detection to world coordinates."""

        x, y = self._screen_to_world.apply_point(point.x, point.y)
        return WorldPoint(x=x, y=y)

    def to_screen(self, point: WorldPoint) -> ScreenPoint:
        """Project a world position back to screen pixels."""

        x, y = self._world_to_screen.apply_point(point.x, point.y)
        return ScreenPoint(x=x, y=y)

    # -- camera motion across frames (F2) --------------------------------

    def update(
        self,
        screen_to_world: Homography,
        *,
        reprojection_error: float = 0.0,
    ) -> CameraProjector:
        """Answer a new projector for a freshly re-recovered screen→world map (F2).

        The projector's ``error_tolerance`` and ``provenance`` are preserved.
        """

        return CameraProjector(
            screen_to_world,
            reprojection_error=reprojection_error,
            error_tolerance=self._error_tolerance,
            provenance=self._provenance,
        )

    def apply_screen_motion(self, motion: Homography) -> CameraProjector:
        """Re-parameterise for a camera move without re-recovering (F2).

        ``motion`` maps *previous-frame* screen coordinates to *current-frame*
        screen coordinates (a pan / zoom / rotation from :mod:`.transforms`). The
        new screen→world map first inverts that motion, then applies the old map:
        ``new = old ∘ motion⁻¹``. The recovery residual (and tolerance/provenance)
        is carried forward, since an affine screen motion does not change the
        ground-plane fit quality.
        """

        moved = self._screen_to_world.compose(motion.inverse())
        return CameraProjector(
            moved,
            reprojection_error=self._reprojection_error,
            error_tolerance=self._error_tolerance,
            provenance=self._provenance,
        )

    # -- projection error as confidence (F4) -----------------------------

    def confidence(self) -> Confidence:
        """Projection confidence derived from the recovery residual (F4).

        Maps the RMS residual ``e`` (world units) to ``exp(-e / tolerance)`` in
        ``[0, 1]``: a perfect fit is ``1.0`` and confidence decays as the residual
        grows past ``error_tolerance``. An explicit heuristic, not a calibrated
        probability.
        """

        value = math.exp(-self._reprojection_error / self._error_tolerance)
        return Confidence(value=value, provenance=self._provenance)

    def to_world_with_confidence(self, point: ScreenPoint) -> tuple[WorldPoint, Confidence]:
        """Project to world space and attach the projection confidence (F1 + F4)."""

        return self.to_world(point), self.confidence()

    @property
    def homography(self) -> Homography:
        """The recovered screen→world map (read-only)."""

        return self._screen_to_world

    @property
    def reprojection_error(self) -> float:
        """RMS recovery residual in world units (0.0 for a perfect / given fit)."""

        return self._reprojection_error

    @property
    def error_tolerance(self) -> float:
        """World-unit residual scale mapped to a 1/e confidence drop (F4)."""

        return self._error_tolerance
