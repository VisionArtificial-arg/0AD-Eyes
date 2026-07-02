"""E6a classical resource-node detection tests — coloured blobs + template art."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.perception.palette import HsvBand
from zero_ad_eyes.infrastructure.perception.resources import ResourceCue, detect_resource_nodes
from zero_ad_eyes.infrastructure.perception.templates import Template

# The historical default resource cues, spelled out as explicit literals.
CUES = (
    ResourceCue(
        entity_type="tree",
        bands=(HsvBand(h_lo=35, h_hi=85, s_lo=40, s_hi=255, v_lo=30, v_hi=255),),
        min_area=20,
    ),
    ResourceCue(
        entity_type="mine",
        bands=(HsvBand(h_lo=0, h_hi=179, s_lo=0, s_hi=50, v_lo=50, v_hi=190),),
        min_area=20,
    ),
    ResourceCue(
        entity_type="bush",
        bands=(
            HsvBand(h_lo=0, h_hi=8, s_lo=90, s_hi=255, v_lo=60, v_hi=255),
            HsvBand(h_lo=168, h_hi=179, s_lo=90, s_hi=255, v_lo=60, v_hi=255),
        ),
        min_area=20,
    ),
    ResourceCue(
        entity_type="fauna",
        bands=(HsvBand(h_lo=9, h_hi=25, s_lo=60, s_hi=200, v_lo=40, v_hi=190),),
        min_area=20,
    ),
)


def _wrap(image: np.ndarray) -> Frame:
    h, w = image.shape[:2]
    return Frame(
        image=image,
        meta=FrameMeta(frame_id=1, timestamp=1.0, source="test", width=w, height=h),
    )


def _scene() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def _types(dets: tuple) -> set[str]:
    return {d.entity_type for d in dets}


def test_detects_green_tree() -> None:
    img = _scene()
    cv2.rectangle(img, (10, 10), (40, 40), (30, 180, 30), -1)  # green foliage
    dets = detect_resource_nodes(_wrap(img), CUES)
    assert "tree" in _types(dets)
    tree = next(d for d in dets if d.entity_type == "tree")
    assert tree.kind is EntityKind.RESOURCE_NODE
    assert tree.confidence.provenance is Provenance.CLASSICAL
    assert len(tree.contour) >= 3


def test_detects_grey_mine() -> None:
    img = _scene()
    cv2.rectangle(img, (80, 60), (120, 100), (130, 130, 130), -1)  # grey rock
    assert "mine" in _types(detect_resource_nodes(_wrap(img), CUES))


def test_detects_red_bush() -> None:
    img = _scene()
    cv2.circle(img, (60, 40), 14, (30, 30, 200), -1)  # red berries
    assert "bush" in _types(detect_resource_nodes(_wrap(img), CUES))


def test_empty_scene_no_detections() -> None:
    assert detect_resource_nodes(_wrap(_scene()), CUES) == ()


def test_roi_limits_search() -> None:
    img = _scene()
    cv2.rectangle(img, (10, 10), (40, 40), (30, 180, 30), -1)  # tree, outside ROI
    roi = ScreenBBox(x=80.0, y=60.0, width=70.0, height=50.0)
    assert detect_resource_nodes(_wrap(img), CUES, roi=roi) == ()


def test_template_resource_art() -> None:
    img = _scene()
    icon = np.zeros((14, 14, 3), dtype=np.uint8)
    cv2.rectangle(icon, (0, 0), (14, 14), (10, 120, 200), -1)
    cv2.circle(icon, (7, 7), 3, (255, 255, 255), -1)
    img[50:64, 100:114] = icon
    template = Template(name="stone_pile", image=icon, entity_type="stone_mine", threshold=0.8)
    dets = detect_resource_nodes(_wrap(img), cues=(), templates=(template,))
    assert any(d.entity_type == "stone_mine" for d in dets)
    assert all(d.kind is EntityKind.RESOURCE_NODE for d in dets)
