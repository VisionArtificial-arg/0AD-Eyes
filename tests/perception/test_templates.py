"""E11 template matching + feature descriptor tests — synthetic art, no display."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.perception.templates import (
    Template,
    TemplateBank,
    count_feature_matches,
    describe_features,
    match_template,
)


def _scene() -> np.ndarray:
    scene = np.zeros((80, 120, 3), dtype=np.uint8)
    # A distinctive icon (bright yellow square with a red core) placed twice.
    for ox, oy in ((10, 10), (70, 40)):
        cv2.rectangle(scene, (ox, oy), (ox + 12, oy + 12), (0, 220, 220), -1)
        cv2.rectangle(scene, (ox + 4, oy + 4), (ox + 8, oy + 8), (0, 0, 220), -1)
    return scene


def _icon() -> np.ndarray:
    icon = np.zeros((12, 12, 3), dtype=np.uint8)
    cv2.rectangle(icon, (0, 0), (12, 12), (0, 220, 220), -1)
    cv2.rectangle(icon, (4, 4), (8, 8), (0, 0, 220), -1)
    return icon


def test_match_template_finds_both_instances() -> None:
    template = Template(name="icon", image=_icon(), threshold=0.7)
    matches = match_template(_scene(), template)
    assert len(matches) == 2
    centers = sorted((m.bbox.x, m.bbox.y) for m in matches)
    assert abs(centers[0][0] - 10) <= 2
    assert abs(centers[1][0] - 70) <= 2


def test_match_template_respects_roi() -> None:
    from zero_ad_eyes.domain.geometry import ScreenBBox

    template = Template(name="icon", image=_icon(), threshold=0.7)
    roi = ScreenBBox(x=60, y=30, width=40, height=40)
    matches = match_template(_scene(), template, roi=roi)
    assert len(matches) == 1
    assert matches[0].bbox.x >= 60  # coordinates are full-frame, inside the ROI


def test_match_template_absent_returns_empty() -> None:
    template = Template(name="icon", image=_icon(), threshold=0.7)
    blank = np.zeros((80, 120, 3), dtype=np.uint8)
    assert match_template(blank, template) == ()


def test_template_bank_emits_classical_detections() -> None:
    template = Template(
        name="tower",
        image=_icon(),
        kind=EntityKind.BUILDING,
        entity_type="tower",
        threshold=0.7,
    )
    bank = TemplateBank(templates=(template,))
    dets = bank.detect(_scene())
    assert len(dets) == 2
    for det in dets:
        assert det.kind is EntityKind.BUILDING
        assert det.entity_type == "tower"
        assert det.confidence.provenance is Provenance.CLASSICAL
        assert 0.0 <= det.confidence.value <= 1.0


def test_feature_descriptors_match_self_more_than_noise() -> None:
    icon = _icon()
    _, desc_icon = describe_features(icon)
    scene = _scene()
    _, desc_scene = describe_features(scene)
    noise = np.random.default_rng(0).integers(0, 255, (12, 12, 3), dtype=np.uint8)
    _, desc_noise = describe_features(noise)
    self_matches = count_feature_matches(desc_icon, desc_scene)
    noise_matches = count_feature_matches(desc_icon, desc_noise)
    assert self_matches >= noise_matches
