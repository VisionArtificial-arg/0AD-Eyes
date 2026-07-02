"""Tests for projection-error confidence (F4). Deterministic, no display."""

from __future__ import annotations

import math

import pytest

from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenPoint, WorldPoint
from zero_ad_eyes.infrastructure.geometry.homography import Homography
from zero_ad_eyes.infrastructure.geometry.projector import CameraProjector

IDENTITY = Homography.identity()


def test_perfect_fit_is_full_confidence() -> None:
    projector = CameraProjector(IDENTITY, reprojection_error=0.0, error_tolerance=1.0)
    conf = projector.confidence()
    assert conf.value == pytest.approx(1.0)
    assert conf.provenance is Provenance.CLASSICAL


def test_confidence_decays_with_residual() -> None:
    small = CameraProjector(IDENTITY, reprojection_error=0.5, error_tolerance=2.0)
    large = CameraProjector(IDENTITY, reprojection_error=4.0, error_tolerance=2.0)
    assert small.confidence().value > large.confidence().value


def test_confidence_matches_exponential_model() -> None:
    projector = CameraProjector(IDENTITY, reprojection_error=3.0, error_tolerance=1.5)
    assert projector.confidence().value == pytest.approx(math.exp(-3.0 / 1.5))


def test_residual_equal_to_tolerance_is_one_over_e() -> None:
    projector = CameraProjector(IDENTITY, reprojection_error=2.0, error_tolerance=2.0)
    assert projector.confidence().value == pytest.approx(math.exp(-1.0))


def test_to_world_with_confidence_pairs_point_and_confidence() -> None:
    projector = CameraProjector(IDENTITY, reprojection_error=0.0, error_tolerance=1.0)
    world, conf = projector.to_world_with_confidence(ScreenPoint(x=7.0, y=9.0))
    assert world == WorldPoint(x=7.0, y=9.0)
    assert conf.value == pytest.approx(1.0)


def test_provenance_is_configurable() -> None:
    projector = CameraProjector(
        IDENTITY, reprojection_error=0.0, error_tolerance=1.0, provenance=Provenance.FUSED
    )
    assert projector.confidence().provenance is Provenance.FUSED


def test_non_positive_tolerance_rejected() -> None:
    with pytest.raises(ValueError):
        CameraProjector(IDENTITY, error_tolerance=0.0)


def test_recovered_projector_reports_high_confidence() -> None:
    screen = [
        ScreenPoint(x=0.0, y=0.0),
        ScreenPoint(x=100.0, y=0.0),
        ScreenPoint(x=100.0, y=80.0),
        ScreenPoint(x=0.0, y=80.0),
    ]
    world = [WorldPoint(x=s.x * 0.5, y=s.y * 0.5) for s in screen]
    projector = CameraProjector.from_correspondences(screen, world, error_tolerance=1.0)
    assert projector.reprojection_error == pytest.approx(0.0, abs=1e-6)
    assert projector.confidence().value == pytest.approx(1.0, abs=1e-6)


def test_tolerance_and_provenance_survive_screen_motion() -> None:
    from zero_ad_eyes.infrastructure.geometry.transforms import translation

    projector = CameraProjector(
        IDENTITY,
        reprojection_error=1.0,
        error_tolerance=4.0,
        provenance=Provenance.FUSED,
    )
    moved = projector.apply_screen_motion(translation(5.0, 5.0))
    assert moved.error_tolerance == pytest.approx(4.0)
    assert moved.confidence().provenance is Provenance.FUSED
    assert moved.confidence().value == pytest.approx(math.exp(-1.0 / 4.0))
