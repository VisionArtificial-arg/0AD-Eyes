"""Tests for geometric transforms and camera-motion updates (F2). Deterministic."""

from __future__ import annotations

import math

import pytest

from zero_ad_eyes.domain.geometry import ScreenPoint, WorldPoint
from zero_ad_eyes.infrastructure.geometry.homography import Homography
from zero_ad_eyes.infrastructure.geometry.projector import CameraProjector
from zero_ad_eyes.infrastructure.geometry.transforms import (
    chain,
    rotation,
    scaling,
    translation,
)

SCREEN_TO_WORLD = Homography.from_matrix([[0.5, 0.0, 10.0], [0.0, 0.5, 5.0], [0.0, 0.0, 1.0]])


def test_translation_pans() -> None:
    assert translation(4.0, -3.0).apply_point(10.0, 10.0) == pytest.approx((14.0, 7.0))


def test_scaling_about_center_fixes_center() -> None:
    zoom = scaling(2.0, center=(100.0, 50.0))
    assert zoom.apply_point(100.0, 50.0) == pytest.approx((100.0, 50.0))
    assert zoom.apply_point(110.0, 50.0) == pytest.approx((120.0, 50.0))


def test_rotation_about_center_fixes_center() -> None:
    rot = rotation(math.pi / 2, center=(0.0, 0.0))
    assert rot.apply_point(1.0, 0.0) == pytest.approx((0.0, 1.0), abs=1e-12)


def test_chain_applies_left_to_right() -> None:
    combined = chain([translation(5.0, 0.0), scaling(2.0)])
    # translate (0,0)->(5,0), then scale by 2 -> (10,0)
    assert combined.apply_point(0.0, 0.0) == pytest.approx((10.0, 0.0))


def test_apply_screen_motion_tracks_a_pan() -> None:
    projector = CameraProjector(SCREEN_TO_WORLD)
    world = WorldPoint(x=42.0, y=17.0)
    old_screen = projector.to_screen(world)

    # The camera pans so the same world point moves +8px right, -6px down on screen.
    motion = translation(8.0, -6.0)
    moved = projector.apply_screen_motion(motion)

    new_screen = ScreenPoint(x=old_screen.x + 8.0, y=old_screen.y - 6.0)
    recovered = moved.to_world(new_screen)
    assert recovered.x == pytest.approx(world.x, abs=1e-9)
    assert recovered.y == pytest.approx(world.y, abs=1e-9)


def test_apply_screen_motion_carries_error_forward() -> None:
    projector = CameraProjector(SCREEN_TO_WORLD, reprojection_error=0.7)
    moved = projector.apply_screen_motion(scaling(1.5))
    assert moved.reprojection_error == pytest.approx(0.7)


def test_update_replaces_the_map() -> None:
    projector = CameraProjector(SCREEN_TO_WORLD)
    replacement = Homography.from_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    updated = projector.update(replacement, reprojection_error=0.2)
    assert updated.reprojection_error == pytest.approx(0.2)
    assert updated.to_world(ScreenPoint(x=3.0, y=4.0)) == WorldPoint(x=3.0, y=4.0)


def test_update_is_immutable() -> None:
    projector = CameraProjector(SCREEN_TO_WORLD)
    projector.apply_screen_motion(translation(100.0, 100.0))
    # Original projector is unchanged by the motion.
    assert projector.to_world(ScreenPoint(x=0.0, y=0.0)) == WorldPoint(x=10.0, y=5.0)
