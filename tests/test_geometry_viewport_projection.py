"""F1 integration — ViewportCameraProjector projects entities via the minimap quad."""

from __future__ import annotations

from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.minimap import MinimapModel, ViewportQuad
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.geometry import ViewportCameraProjector

from .conftest import make_frame

_PROJECTOR = ViewportCameraProjector(error_tolerance=10.0)


def _quad(*corners: tuple[float, float]) -> ViewportQuad:
    tl, tr, br, bl = (WorldPoint(x=x, y=y) for x, y in corners)
    return ViewportQuad(top_left=tl, top_right=tr, bottom_right=br, bottom_left=bl)


def _entity(entity_id: int, box: ScreenBBox | None, world: WorldPoint | None = None) -> Entity:
    return Entity(entity_id=entity_id, kind=EntityKind.UNIT, screen_bbox=box, world_pos=world)


def test_projects_screen_centre_to_world_via_axis_aligned_quad() -> None:
    # 64x48 frame; the viewport quad spans world (0,0)-(64,48) axis-aligned, so the
    # screen->world map is the identity and an entity's centre maps to itself.
    frame = make_frame(width=64, height=48)
    minimap = MinimapModel(viewport=_quad((0.0, 0.0), (64.0, 0.0), (64.0, 48.0), (0.0, 48.0)))
    entity = _entity(0, ScreenBBox(x=30.0, y=20.0, width=4.0, height=4.0))  # centre (32, 22)

    projected = _PROJECTOR.project((entity,), minimap, frame)

    assert projected[0].world_pos is not None
    assert projected[0].world_pos.x == 32.0
    assert projected[0].world_pos.y == 22.0


def test_scaled_quad_maps_proportionally() -> None:
    # World quad half the frame extent → screen (w,h) maps to world (w/2, h/2).
    frame = make_frame(width=64, height=48)
    minimap = MinimapModel(viewport=_quad((0.0, 0.0), (32.0, 0.0), (32.0, 24.0), (0.0, 24.0)))
    entity = _entity(0, ScreenBBox(x=62.0, y=46.0, width=2.0, height=2.0))  # centre (63, 47)

    projected = _PROJECTOR.project((entity,), minimap, frame)

    assert projected[0].world_pos is not None
    assert abs(projected[0].world_pos.x - 31.5) < 1e-6
    assert abs(projected[0].world_pos.y - 23.5) < 1e-6


def test_no_viewport_leaves_entities_unchanged() -> None:
    frame = make_frame()
    entity = _entity(0, ScreenBBox(x=1.0, y=1.0, width=2.0, height=2.0))
    assert _PROJECTOR.project((entity,), MinimapModel(viewport=None), frame) == (entity,)


def test_existing_world_pos_is_preserved() -> None:
    frame = make_frame(width=64, height=48)
    minimap = MinimapModel(viewport=_quad((0.0, 0.0), (64.0, 0.0), (64.0, 48.0), (0.0, 48.0)))
    fixed = WorldPoint(x=999.0, y=999.0)
    entity = _entity(0, ScreenBBox(x=30.0, y=20.0, width=4.0, height=4.0), world=fixed)

    projected = _PROJECTOR.project((entity,), minimap, frame)

    assert projected[0].world_pos == fixed  # projector must not overwrite it


def test_entity_without_screen_box_is_left_alone() -> None:
    frame = make_frame(width=64, height=48)
    minimap = MinimapModel(viewport=_quad((0.0, 0.0), (64.0, 0.0), (64.0, 48.0), (0.0, 48.0)))
    entity = _entity(0, None)

    projected = _PROJECTOR.project((entity,), minimap, frame)

    assert projected[0].world_pos is None


def test_degenerate_quad_yields_no_projection() -> None:
    # All four corners collinear → no homography can be recovered; entities pass through.
    frame = make_frame(width=64, height=48)
    minimap = MinimapModel(viewport=_quad((0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)))
    entity = _entity(0, ScreenBBox(x=30.0, y=20.0, width=4.0, height=4.0))

    projected = _PROJECTOR.project((entity,), minimap, frame)

    assert projected[0].world_pos is None


def test_from_settings_reads_camera_error_tolerance() -> None:
    from zero_ad_eyes.interface.default_config import default_config

    geometry = default_config().geometry
    projector = ViewportCameraProjector.from_settings(geometry)
    assert projector._error_tolerance == geometry.camera_error_tolerance
