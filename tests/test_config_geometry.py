"""Config guards for the geometry / fusion knobs (Approach B, NF7).

The geometry section has no offline pipeline stage yet, but its values now live in the
config (and the generator) and flow into the geometry/fusion helpers as required
parameters — no in-code defaults. These guard the generated defaults and prove a
config value threads into the helpers.
"""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.geometry.fusion import reconcile
from zero_ad_eyes.infrastructure.geometry.homography import Homography
from zero_ad_eyes.infrastructure.geometry.projector import CameraProjector
from zero_ad_eyes.interface.default_config import default_config


def _conf(value: float) -> Confidence:
    return Confidence(value=value, provenance=Provenance.CLASSICAL)


def test_defaults_match_historical() -> None:
    g = default_config().geometry
    assert g.camera_error_tolerance == 1.0
    assert g.fusion_agreement_scale == 1.0
    assert g.fusion_match_radius == 20.0


def test_config_threads_error_tolerance_into_projector() -> None:
    config = load_config(default_config(), env={"ZAE_GEOMETRY__CAMERA_ERROR_TOLERANCE": "4.0"})
    projector = CameraProjector(
        Homography.identity(), error_tolerance=config.geometry.camera_error_tolerance
    )
    assert projector.error_tolerance == 4.0


def test_config_threads_agreement_scale_into_reconcile() -> None:
    config = load_config(default_config(), env={"ZAE_GEOMETRY__FUSION_AGREEMENT_SCALE": "10.0"})
    # A wider agreement scale discounts the same disagreement less than the tight one.
    _, loose = reconcile(
        WorldPoint(x=0.0, y=0.0),
        _conf(0.7),
        WorldPoint(x=5.0, y=0.0),
        _conf(0.7),
        agreement_scale=config.geometry.fusion_agreement_scale,
    )
    _, tight = reconcile(
        WorldPoint(x=0.0, y=0.0),
        _conf(0.7),
        WorldPoint(x=5.0, y=0.0),
        _conf(0.7),
        agreement_scale=1.0,
    )
    assert loose.value > tight.value
