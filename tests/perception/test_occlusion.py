"""E7 occlusion tests — pure geometry over stub detections, no pixels needed."""

from __future__ import annotations

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.perception.occlusion import (
    resolve_occlusions,
    visible_fraction,
)


def _det(x: float, y: float, w: float, h: float, conf: float = 1.0) -> Detection:
    return Detection(
        kind=EntityKind.UNIT,
        bbox=ScreenBBox(x=x, y=y, width=w, height=h),
        confidence=Confidence(value=conf, provenance=Provenance.CLASSICAL),
    )


def test_visible_fraction_no_occluders() -> None:
    assert visible_fraction(ScreenBBox(x=0, y=0, width=10, height=10), []) == 1.0


def test_visible_fraction_half_covered() -> None:
    target = ScreenBBox(x=0, y=0, width=10, height=10)
    occ = ScreenBBox(x=5, y=0, width=10, height=10)  # covers right half
    assert abs(visible_fraction(target, [occ]) - 0.5) <= 1e-6


def test_visible_fraction_capped_at_full_cover() -> None:
    target = ScreenBBox(x=0, y=0, width=10, height=10)
    occ = ScreenBBox(x=0, y=0, width=20, height=20)
    assert visible_fraction(target, [occ]) == 0.0


def test_nearer_box_occludes_farther() -> None:
    back = _det(0, 0, 10, 10)  # bottom edge y=10
    front = _det(5, 5, 10, 10)  # bottom edge y=15 → nearer
    infos = resolve_occlusions([back, front])
    back_info = infos[0]
    front_info = infos[1]
    assert back_info.is_occluded
    assert back_info.occluder_indices == (1,)
    assert back_info.visible_fraction < 1.0
    # The front object is occluded by nobody.
    assert not front_info.is_occluded
    assert front_info.visible_fraction == 1.0


def test_occlusion_scales_confidence_and_keeps_provenance() -> None:
    back = _det(0, 0, 10, 10, conf=0.8)
    front = _det(5, 0, 10, 10)  # covers right half, nearer (equal bottom? no)
    # Make front strictly nearer via a taller box reaching lower.
    front = _det(5, 0, 10, 12)
    infos = resolve_occlusions([back, front])
    back_info = infos[0]
    assert abs(back_info.adjusted_confidence.value - 0.8 * back_info.visible_fraction) <= 1e-6
    assert back_info.adjusted_confidence.provenance is Provenance.CLASSICAL


def test_non_overlapping_are_all_visible() -> None:
    a = _det(0, 0, 10, 10)
    b = _det(50, 50, 10, 10)
    infos = resolve_occlusions([a, b])
    assert all(not info.is_occluded for info in infos)
