"""Domain value-object tests: construction, bounds, immutability, serialisation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.minimap import ViewportQuad
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.domain.world_model import SCHEMA_VERSION, FrameMeta, WorldModel


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Confidence(value=1.5)


def test_bbox_center_and_area() -> None:
    box = ScreenBBox(x=10, y=20, width=30, height=40)
    assert box.center.x == 25
    assert box.center.y == 40
    assert box.area == 1200


def test_value_objects_are_frozen() -> None:
    box = ScreenBBox(x=0, y=0, width=1, height=1)
    with pytest.raises(ValidationError):
        box.x = 5  # type: ignore[misc]


def test_detection_carries_provenance() -> None:
    det = Detection(
        kind=EntityKind.UNIT,
        bbox=ScreenBBox(x=0, y=0, width=4, height=4),
        confidence=Confidence(value=0.8, provenance=Provenance.CLASSICAL),
    )
    assert det.confidence.provenance is Provenance.CLASSICAL
    assert len(Detections(frame_id=0, items=(det,))) == 1


def test_world_model_roundtrips_json() -> None:
    wm = WorldModel(
        meta=FrameMeta(frame_id=1, timestamp=1.0, source="test", width=8, height=8),
    )
    restored = WorldModel.model_validate_json(wm.model_dump_json())
    assert restored == wm
    assert restored.schema_version == SCHEMA_VERSION


def _quad(*corners: tuple[float, float]) -> ViewportQuad:
    tl, tr, br, bl = (WorldPoint(x=x, y=y) for x, y in corners)
    return ViewportQuad(top_left=tl, top_right=tr, bottom_right=br, bottom_left=bl)


def test_viewport_quad_corners_order() -> None:
    quad = _quad((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    assert quad.corners() == (quad.top_left, quad.top_right, quad.bottom_right, quad.bottom_left)


def test_viewport_quad_contains_inside_and_outside() -> None:
    quad = _quad((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    assert quad.contains(WorldPoint(x=5.0, y=5.0)) is True
    assert quad.contains(WorldPoint(x=0.0, y=0.0)) is True  # corner (edge inclusive)
    assert quad.contains(WorldPoint(x=5.0, y=0.0)) is True  # edge
    assert quad.contains(WorldPoint(x=-1.0, y=5.0)) is False
    assert quad.contains(WorldPoint(x=5.0, y=20.0)) is False


def test_viewport_quad_contains_is_winding_agnostic_and_handles_trapezoid() -> None:
    # Counter-clockwise winding + a perspective trapezoid (narrow far edge).
    ccw = _quad((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))
    assert ccw.contains(WorldPoint(x=5.0, y=5.0)) is True

    trapezoid = _quad((3.0, 0.0), (7.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    assert trapezoid.contains(WorldPoint(x=5.0, y=1.0)) is True  # inside the narrow top
    assert trapezoid.contains(WorldPoint(x=1.0, y=1.0)) is False  # outside the slanted side
