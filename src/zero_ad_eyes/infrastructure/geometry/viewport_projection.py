"""Viewport-driven screen→world projection (F1 integration).

``ViewportCameraProjector`` implements the ``ScreenToWorldProjector`` port. It
recovers a ground-plane homography from a *pure-pixel* correspondence source — the
minimap camera-viewport quad (D5) — and stamps a world position onto each tracked
entity so the G4/G5 fusion has main-view and minimap estimates in one frame.

The correspondence is: the four screen corners (the full rendered frame, in
TL/TR/BR/BL order) map to the four world corners of the camera's ground footprint,
read off the minimap as a :class:`ViewportQuad` in the same order.

**Assumption (unvalidated — real frames are blocked on the engine exporter).** The
corners are paired by canonical position, which is correct only while the camera yaw
is near 0 A.D.'s default (main view aligned with minimap-north). A large *manual
camera rotation* would pair physically-wrong corners and yield a bad map; the camera
*tilt* (the perspective this recovers) is handled correctly. With exactly four exact
correspondences the recovery residual is ~0, so the F4 confidence is ~1.0 and is
deliberately **not** stamped onto entities (it would overstate certainty); it guards
only degenerate geometry, which is caught as an exception below.
"""

from __future__ import annotations

from collections.abc import Sequence

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import GeometrySettings
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenPoint
from zero_ad_eyes.domain.minimap import MinimapModel, ViewportQuad

from .homography import DegenerateHomographyError
from .projector import CameraProjector


class ViewportCameraProjector:
    """A ``ScreenToWorldProjector`` recovering the map from the minimap viewport (F1)."""

    def __init__(self, *, error_tolerance: float) -> None:
        self._error_tolerance = error_tolerance

    @classmethod
    def from_settings(cls, geometry: GeometrySettings) -> ViewportCameraProjector:
        """Build from the ``geometry`` config (Approach B boundary mapping)."""

        return cls(error_tolerance=geometry.camera_error_tolerance)

    def project(
        self, entities: Sequence[Entity], minimap: MinimapModel, frame: Frame
    ) -> tuple[Entity, ...]:
        if minimap.viewport is None:
            return tuple(entities)
        projector = self._recover(minimap.viewport, frame)
        if projector is None:
            return tuple(entities)
        return tuple(self._project_entity(entity, projector) for entity in entities)

    def _recover(self, viewport: ViewportQuad, frame: Frame) -> CameraProjector | None:
        width, height = self._frame_size(frame)
        if width <= 0 or height <= 0:
            return None
        screen_corners = [
            ScreenPoint(x=0.0, y=0.0),
            ScreenPoint(x=float(width), y=0.0),
            ScreenPoint(x=float(width), y=float(height)),
            ScreenPoint(x=0.0, y=float(height)),
        ]
        world_corners = list(viewport.corners())  # TL, TR, BR, BL — same order
        try:
            return CameraProjector.from_correspondences(
                screen_corners, world_corners, error_tolerance=self._error_tolerance
            )
        except (DegenerateHomographyError, ValueError):
            return None  # degenerate quad → no usable map; leave entities in screen space

    def _project_entity(self, entity: Entity, projector: CameraProjector) -> Entity:
        if entity.screen_bbox is None or entity.world_pos is not None:
            return entity  # nothing to project, or a world position already exists
        world = projector.to_world(entity.screen_bbox.center)
        return entity.model_copy(update={"world_pos": world})

    def _frame_size(self, frame: Frame) -> tuple[int, int]:
        """Prefer the actual pixel-buffer shape; fall back to declared metadata."""

        shape = getattr(frame.image, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
        return frame.meta.width, frame.meta.height
