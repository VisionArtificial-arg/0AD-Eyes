"""Tests for the homography primitive (F1). Deterministic, no display."""

from __future__ import annotations

import math

import numpy as np
import pytest

from zero_ad_eyes.infrastructure.geometry.homography import (
    DegenerateHomographyError,
    Homography,
)

# A known, non-trivial projective transform (rotation + scale + shear + a genuine
# perspective term in the bottom row).
KNOWN = Homography.from_matrix(
    [
        [1.2, 0.1, 30.0],
        [-0.05, 0.9, -10.0],
        [0.0005, 0.0002, 1.0],
    ]
)

SRC = [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0), (50.0, 40.0)]


def test_identity_is_a_no_op() -> None:
    h = Homography.identity()
    assert h.apply_point(3.0, 7.0) == (3.0, 7.0)


def test_matrix_is_normalised() -> None:
    h = Homography.from_matrix([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    assert h.matrix[2, 2] == pytest.approx(1.0)
    # Scale factor 2/2 == 1: identity after normalisation.
    assert h.apply_point(5.0, 9.0) == pytest.approx((5.0, 9.0))


def test_inverse_round_trips() -> None:
    inv = KNOWN.inverse()
    for x, y in SRC:
        mapped = KNOWN.apply_point(x, y)
        back = inv.apply_point(*mapped)
        assert back == pytest.approx((x, y), abs=1e-9)


def test_compose_matches_sequential_application() -> None:
    a = KNOWN
    b = Homography.from_matrix([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    composed = a.compose(b)  # apply b first, then a
    for x, y in SRC:
        bx, by = b.apply_point(x, y)
        expected = a.apply_point(bx, by)
        assert composed.apply_point(x, y) == pytest.approx(expected, abs=1e-9)


def test_recovers_a_known_homography_from_correspondences() -> None:
    dst = [KNOWN.apply_point(x, y) for x, y in SRC]
    recovered = Homography.from_correspondences(SRC, dst)
    for x, y in SRC:
        assert recovered.apply_point(x, y) == pytest.approx(KNOWN.apply_point(x, y), abs=1e-4)
    assert recovered.reprojection_error(SRC, dst) == pytest.approx(0.0, abs=1e-4)


def test_reprojection_error_reports_a_known_offset() -> None:
    # dst is the true mapping shifted by (3, 4) on every point -> RMS distance 5.
    dst = [(mx + 3.0, my + 4.0) for mx, my in (KNOWN.apply_point(x, y) for x, y in SRC)]
    err = KNOWN.reprojection_error(SRC, dst)
    assert err == pytest.approx(5.0, abs=1e-9)


def test_apply_vectorised_matches_pointwise() -> None:
    pts = np.array(SRC, dtype=np.float64)
    batch = KNOWN.apply(pts)
    for (x, y), row in zip(SRC, batch, strict=True):
        assert (float(row[0]), float(row[1])) == pytest.approx(KNOWN.apply_point(x, y))


def test_too_few_correspondences_rejected() -> None:
    with pytest.raises(ValueError):
        Homography.from_correspondences(SRC[:3], SRC[:3])


def test_singular_matrix_is_not_invertible() -> None:
    singular = Homography.from_matrix([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]])
    with pytest.raises(DegenerateHomographyError):
        singular.inverse()


def test_non_square_matrix_rejected() -> None:
    with pytest.raises(ValueError):
        Homography.from_matrix([[1.0, 0.0], [0.0, 1.0]])


def test_error_on_empty_is_zero_and_finite() -> None:
    assert KNOWN.reprojection_error([], []) == 0.0
    assert math.isfinite(KNOWN.apply_point(1.0, 1.0)[0])
