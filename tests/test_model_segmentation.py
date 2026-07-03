"""MP4 — learned segmentation adapter.

Two layers: the mask→``Detections`` bridge is exercised torch-free (pure numpy +
opencv), and an end-to-end load+infer test is guarded by ``importorskip('torch')``
so the gate stays green on the core stack (torch is the optional ``learned`` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.detections import Detections
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.model import (
    SegmentationPerceptionModel,
    default_contract,
)
from zero_ad_eyes.infrastructure.model.segmentation_adapter import (
    _EMIT_CLASSES,
    CLASS_NAMES,
    KIND_BY_CLASS_ID,
    NUM_CLASSES,
)


def test_class_map_is_total_and_emits_only_entities() -> None:
    ids = set(range(NUM_CLASSES))
    assert set(KIND_BY_CLASS_ID) == ids == set(CLASS_NAMES)
    assert _EMIT_CLASSES <= ids
    # terrain (0-8), decorative vegetation (13) and the backdrop are scene, not entities.
    assert _EMIT_CLASSES.isdisjoint({0, 1, 2, 3, 4, 5, 6, 7, 8, 13})
    assert {14, 15, 16} <= _EMIT_CLASSES  # fauna, building, unit are emitted


def test_mask_postprocessing_recovers_one_detection_per_blob() -> None:
    # Torch-free: drive the segmentation->Detections bridge directly.
    adapter = SegmentationPerceptionModel(
        model=None, input_size=(4, 4), score_threshold=0.1, min_region_area=1
    )
    class_map = np.zeros((4, 4), dtype=np.int32)
    class_map[0:2, 0:2] = 16  # a 2x2 'unit' blob in the top-left of the tensor grid
    prob_map = np.full((4, 4), 0.9, dtype=np.float32)

    dets = adapter._detections_from_mask(class_map, prob_map, scale_x=2.0, scale_y=2.0)

    assert len(dets) == 1
    d = dets[0]
    assert d.kind == EntityKind.UNIT
    assert d.entity_type == "unit"
    assert d.confidence.provenance == Provenance.LEARNED
    assert d.confidence.value == pytest.approx(0.9)
    # tensor bbox (0,0,2,2) rescaled by 2 -> source (0,0,4,4)
    assert (d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height) == (0.0, 0.0, 4.0, 4.0)


def test_emitted_detections_satisfy_the_mp2_contract() -> None:
    adapter = SegmentationPerceptionModel(
        model=None, input_size=(4, 4), score_threshold=0.1, min_region_area=1
    )
    class_map = np.full((4, 4), 15, dtype=np.int32)  # all 'building'
    prob_map = np.full((4, 4), 0.8, dtype=np.float32)

    dets = adapter._detections_from_mask(class_map, prob_map, scale_x=1.0, scale_y=1.0)

    # LEARNED provenance is in the contract's allowed set — validates without raising.
    default_contract().validate_detections(Detections(frame_id=0, items=tuple(dets)), 0)


def test_from_weights_infers_learned_detections() -> None:
    pytest.importorskip("torch")  # only runs with the `learned` extra installed

    from zero_ad_eyes.application.frames import Frame
    from zero_ad_eyes.domain.world_model import FrameMeta

    model = SegmentationPerceptionModel.from_weights()
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = Frame(
        image=image,
        meta=FrameMeta(frame_id=7, timestamp=0.0, source="synthetic", width=1920, height=1080),
    )

    dets = model.infer(frame)

    assert dets.frame_id == 7
    for d in dets.items:
        assert d.confidence.provenance == Provenance.LEARNED
        assert 0.0 <= d.confidence.value <= 1.0
