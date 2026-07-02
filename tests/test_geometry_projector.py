"""Tests for the CameraProjector screen ⇄ world API (F1). Deterministic."""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.geometry import ScreenPoint, WorldPoint
from zero_ad_eyes.infrastructure.geometry.homography import Homography
from zero_ad_eyes.infrastructure.geometry.projector import CameraProjector

# Ground-plane map with a perspective term, standing in for a tilted RTS camera.
SCREEN_TO_WORLD = Homography.from_matrix(
    [
        [0.5, 0.02, 10.0],
        [0.0, 0.4, 5.0],
        [0.0, 0.001, 1.0],
    ]
)

SCREEN_CORNERS = [
    ScreenPoint(x=0.0, y=0.0),
    ScreenPoint(x=640.0, y=0.0),
    ScreenPoint(x=640.0, y=480.0),
    ScreenPoint(x=0.0, y=480.0),
    ScreenPoint(x=320.0, y=240.0),
]


def _projector() -> CameraProjector:
    return CameraProjector(SCREEN_TO_WORLD, error_tolerance=1.0)


def test_to_world_returns_world_point() -> None:
    world = _projector().to_world(ScreenPoint(x=320.0, y=240.0))
    assert isinstance(world, WorldPoint)


def test_screen_world_round_trip() -> None:
    projector = _projector()
    for screen in SCREEN_CORNERS:
        world = projector.to_world(screen)
        back = projector.to_screen(world)
        assert back.x == pytest.approx(screen.x, abs=1e-6)
        assert back.y == pytest.approx(screen.y, abs=1e-6)


def test_from_correspondences_recovers_projection() -> None:
    world_pts = [projected for projected in (_projector().to_world(s) for s in SCREEN_CORNERS)]
    recovered = CameraProjector.from_correspondences(SCREEN_CORNERS, world_pts, error_tolerance=1.0)
    for screen, expected in zip(SCREEN_CORNERS, world_pts, strict=True):
        got = recovered.to_world(screen)
        assert got.x == pytest.approx(expected.x, abs=1e-4)
        assert got.y == pytest.approx(expected.y, abs=1e-4)


def test_from_correspondences_reports_small_residual() -> None:
    world_pts = [_projector().to_world(s) for s in SCREEN_CORNERS]
    recovered = CameraProjector.from_correspondences(SCREEN_CORNERS, world_pts, error_tolerance=1.0)
    assert recovered.reprojection_error == pytest.approx(0.0, abs=1e-4)


def test_default_reprojection_error_is_zero() -> None:
    assert _projector().reprojection_error == 0.0
