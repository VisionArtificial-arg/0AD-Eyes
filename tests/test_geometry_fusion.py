"""Tests for main-view / minimap world-position reconciliation (F3)."""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.infrastructure.geometry.fusion import reconcile


def _conf(value: float, prov: Provenance = Provenance.CLASSICAL) -> Confidence:
    return Confidence(value=value, provenance=prov)


def test_fused_provenance_is_fused() -> None:
    _, conf = reconcile(WorldPoint(x=0.0, y=0.0), _conf(0.8), WorldPoint(x=0.0, y=0.0), _conf(0.8))
    assert conf.provenance is Provenance.FUSED


def test_confidence_weighted_position() -> None:
    point, _ = reconcile(
        WorldPoint(x=0.0, y=0.0),
        _conf(0.75),
        WorldPoint(x=10.0, y=0.0),
        _conf(0.25),
    )
    # Weighted mean: (0.75*0 + 0.25*10) / 1.0 == 2.5
    assert point.x == pytest.approx(2.5)
    assert point.y == pytest.approx(0.0)


def test_equal_confidence_is_midpoint() -> None:
    point, _ = reconcile(WorldPoint(x=2.0, y=4.0), _conf(0.6), WorldPoint(x=6.0, y=8.0), _conf(0.6))
    assert point.x == pytest.approx(4.0)
    assert point.y == pytest.approx(6.0)


def test_agreeing_sources_reinforce_confidence() -> None:
    _, conf = reconcile(WorldPoint(x=5.0, y=5.0), _conf(0.6), WorldPoint(x=5.0, y=5.0), _conf(0.6))
    # Same location -> agreement 1.0; noisy-OR of 0.6, 0.6 = 0.84 > either input.
    assert conf.value == pytest.approx(0.84)


def test_disagreement_discounts_confidence() -> None:
    _, close = reconcile(WorldPoint(x=0.0, y=0.0), _conf(0.6), WorldPoint(x=0.0, y=0.0), _conf(0.6))
    _, near = reconcile(WorldPoint(x=0.0, y=0.0), _conf(0.6), WorldPoint(x=0.5, y=0.0), _conf(0.6))
    _, far = reconcile(WorldPoint(x=0.0, y=0.0), _conf(0.6), WorldPoint(x=20.0, y=0.0), _conf(0.6))
    assert close.value > near.value > far.value
    assert far.value < 0.1


def test_zero_confidence_falls_back_to_midpoint_unknown() -> None:
    point, conf = reconcile(
        WorldPoint(x=0.0, y=0.0), _conf(0.0), WorldPoint(x=4.0, y=2.0), _conf(0.0)
    )
    assert point.x == pytest.approx(2.0)
    assert point.y == pytest.approx(1.0)
    assert conf.value == 0.0
    assert conf.provenance is Provenance.FUSED


def test_single_trusted_source_dominates_position() -> None:
    point, _ = reconcile(
        WorldPoint(x=3.0, y=3.0), _conf(0.9), WorldPoint(x=100.0, y=100.0), _conf(0.0)
    )
    assert point.x == pytest.approx(3.0)
    assert point.y == pytest.approx(3.0)


def test_agreement_scale_widens_tolerance() -> None:
    tight = reconcile(
        WorldPoint(x=0.0, y=0.0),
        _conf(0.7),
        WorldPoint(x=5.0, y=0.0),
        _conf(0.7),
        agreement_scale=1.0,
    )[1]
    loose = reconcile(
        WorldPoint(x=0.0, y=0.0),
        _conf(0.7),
        WorldPoint(x=5.0, y=0.0),
        _conf(0.7),
        agreement_scale=10.0,
    )[1]
    assert loose.value > tight.value


def test_non_positive_agreement_scale_rejected() -> None:
    with pytest.raises(ValueError):
        reconcile(
            WorldPoint(x=0.0, y=0.0),
            _conf(0.5),
            WorldPoint(x=0.0, y=0.0),
            _conf(0.5),
            agreement_scale=0.0,
        )
