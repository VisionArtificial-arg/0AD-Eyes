"""ClassicalPerceptionModel tests — port conformance + classical detections."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import PerceptionModel
from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.detections import Detections
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.perception import (
    ClassicalPerceptionModel,
    Template,
    TemplateBank,
)
from zero_ad_eyes.infrastructure.perception.palette import HsvBand
from zero_ad_eyes.infrastructure.perception.resources import ResourceCue

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


def _frame(image: np.ndarray, frame_id: int = 7) -> Frame:
    h, w = image.shape[:2]
    return Frame(
        image=image,
        meta=FrameMeta(frame_id=frame_id, timestamp=1.0, source="test", width=w, height=h),
    )


def test_satisfies_perception_model_port() -> None:
    model = ClassicalPerceptionModel(resource_cues=CUES, detect_resources=True)
    assert isinstance(model, PerceptionModel)


def test_infer_tags_frame_id_and_returns_detections() -> None:
    model = ClassicalPerceptionModel(resource_cues=CUES, detect_resources=False)
    out = model.infer(_frame(np.zeros((40, 40, 3), dtype=np.uint8), frame_id=42))
    assert isinstance(out, Detections)
    assert out.frame_id == 42
    assert len(out) == 0


def test_infer_finds_resources_and_templates_all_classical() -> None:
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (30, 180, 30), -1)  # a green tree

    icon = np.zeros((14, 14, 3), dtype=np.uint8)
    cv2.rectangle(icon, (0, 0), (14, 14), (10, 120, 200), -1)
    cv2.circle(icon, (7, 7), 3, (255, 255, 255), -1)
    img[70:84, 120:134] = icon
    bank = TemplateBank(
        templates=(
            Template(name="depot", image=icon, kind=EntityKind.BUILDING, entity_type="depot"),
        )
    )

    model = ClassicalPerceptionModel(resource_cues=CUES, detect_resources=True, template_bank=bank)
    out = model.infer(_frame(img))
    types = {d.entity_type for d in out.items}
    assert "tree" in types
    assert "depot" in types
    assert all(d.confidence.provenance is Provenance.CLASSICAL for d in out.items)
    assert all(d.confidence.provenance is not Provenance.LEARNED for d in out.items)
