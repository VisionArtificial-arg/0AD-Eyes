"""MP2 — tests for the frozen model I/O contract value object."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.model.contract import (
    CONTRACT_VERSION,
    MODEL_IO_CONTRACT_V1,
    ChannelOrder,
    ContractViolation,
    InputTensorSpec,
    ModelIOContract,
    Normalization,
    TensorLayout,
    committed_contract,
    default_contract,
)


def test_contract_is_versioned_and_frozen() -> None:
    contract = default_contract()
    assert contract.version == CONTRACT_VERSION
    with pytest.raises(ValidationError):
        contract.version = "9.9.9"  # type: ignore[misc]


def test_default_contract_is_provisional_without_a_model() -> None:
    # No model exists, so the model-internal input spec is deliberately uncommitted.
    contract = default_contract()
    assert contract.is_provisional
    assert contract.input is None


def test_committed_contract_supplies_the_input_spec() -> None:
    spec = InputTensorSpec(
        height=640,
        width=640,
        channel_order=ChannelOrder.RGB,
        layout=TensorLayout.NCHW,
        dtype="float32",
        normalization=Normalization(scale=1.0 / 255.0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    )
    contract = committed_contract(spec)
    assert not contract.is_provisional
    assert contract.input == spec


def test_contract_round_trips_through_json() -> None:
    payload = MODEL_IO_CONTRACT_V1.model_dump(mode="json")
    rehydrated = ModelIOContract.model_validate(payload)
    assert rehydrated == MODEL_IO_CONTRACT_V1


def test_output_schema_points_at_detections() -> None:
    assert MODEL_IO_CONTRACT_V1.output_schema.endswith("Detections")


def test_class_map_resolves_every_coarse_kind() -> None:
    resolved = set(MODEL_IO_CONTRACT_V1.classes.kind_by_class_id.values())
    assert resolved == set(EntityKind)
    assert MODEL_IO_CONTRACT_V1.classes.kind_of(2) is EntityKind.RESOURCE_NODE


def test_validate_detections_accepts_conformant_payload() -> None:
    det = Detection(
        kind=EntityKind.RESOURCE_NODE,
        bbox=ScreenBBox(x=0, y=0, width=4, height=4),
        confidence=Confidence(value=0.7, provenance=Provenance.CLASSICAL),
    )
    detections = Detections(frame_id=5, items=(det,))
    MODEL_IO_CONTRACT_V1.validate_detections(detections, frame_id=5)


def test_validate_detections_rejects_frame_id_mismatch() -> None:
    detections = Detections(frame_id=1, items=())
    with pytest.raises(ContractViolation):
        MODEL_IO_CONTRACT_V1.validate_detections(detections, frame_id=2)


def test_validate_detections_rejects_disallowed_provenance() -> None:
    det = Detection(
        kind=EntityKind.UNIT,
        bbox=ScreenBBox(x=0, y=0, width=1, height=1),
        confidence=Confidence(value=0.5, provenance=Provenance.UNKNOWN),
    )
    detections = Detections(frame_id=0, items=(det,))
    with pytest.raises(ContractViolation):
        MODEL_IO_CONTRACT_V1.validate_detections(detections, frame_id=0)
