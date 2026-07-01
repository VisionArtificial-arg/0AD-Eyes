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


def _frame(image: np.ndarray, frame_id: int = 7) -> Frame:
    h, w = image.shape[:2]
    return Frame(
        image=image,
        meta=FrameMeta(frame_id=frame_id, timestamp=1.0, source="test", width=w, height=h),
    )


def test_satisfies_perception_model_port() -> None:
    model = ClassicalPerceptionModel()
    assert isinstance(model, PerceptionModel)


def test_infer_tags_frame_id_and_returns_detections() -> None:
    model = ClassicalPerceptionModel(detect_resources=False)
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

    model = ClassicalPerceptionModel(template_bank=bank)
    out = model.infer(_frame(img))
    types = {d.entity_type for d in out.items}
    assert "tree" in types
    assert "depot" in types
    assert all(d.confidence.provenance is Provenance.CLASSICAL for d in out.items)
    assert all(d.confidence.provenance is not Provenance.LEARNED for d in out.items)
