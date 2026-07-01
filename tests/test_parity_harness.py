"""MP5 — run the adapter-parity harness against the stub adapter.

The same harness is expected to be run against the real adapter (MP4) unchanged.
"""

from __future__ import annotations

import pytest

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.model.contract import ContractViolation
from zero_ad_eyes.infrastructure.model.parity import assert_satisfies_contract
from zero_ad_eyes.infrastructure.model.stub_adapter import StubPerceptionModel

from .conftest import make_frame


def test_empty_stub_satisfies_contract() -> None:
    result = assert_satisfies_contract(StubPerceptionModel(), make_frame(3))
    assert result.frame_id == 3
    assert len(result) == 0


def test_fixture_backed_stub_satisfies_contract() -> None:
    det = Detection(
        kind=EntityKind.RESOURCE_NODE,
        bbox=ScreenBBox(x=0, y=0, width=5, height=5),
        confidence=Confidence(value=0.8, provenance=Provenance.ENGINE_GT),
    )
    fixtures = {2: Detections(frame_id=2, items=(det,))}
    result = assert_satisfies_contract(StubPerceptionModel(fixtures), make_frame(2))
    assert len(result) == 1


def test_harness_accepts_explicit_roi() -> None:
    roi = ScreenBBox(x=0, y=0, width=10, height=10)
    assert_satisfies_contract(StubPerceptionModel(), make_frame(0), roi=roi)


def test_harness_flags_a_contract_breaking_model() -> None:
    """A deliberately non-conformant model must fail the harness (guards R6)."""

    class WrongFrameIdModel:
        def infer(self, frame: Frame, roi: ScreenBBox | None = None) -> Detections:
            return Detections(frame_id=frame.meta.frame_id + 1, items=())

    with pytest.raises(ContractViolation):
        assert_satisfies_contract(WrongFrameIdModel(), make_frame(0))
