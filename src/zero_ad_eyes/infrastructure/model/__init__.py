"""Model seam adapters (REQUIREMENTS.md §5.10).

``StubPerceptionModel`` (MP3) satisfies the ``PerceptionModel`` port without a
trained model, so the whole pipeline runs today. The real adapter (MP4, loading the
delivered ONNX artifact) is added here later and is the ONLY new code at plug-in
time; parity tests (MP5) assert it satisfies the identical contract.
"""

from __future__ import annotations

from .contract import ContractViolation, ModelIOContract, default_contract
from .fixture_loader import FixtureFormatError, load_fixture_detections, load_fixture_model
from .parity import assert_satisfies_contract
from .stub_adapter import StubPerceptionModel

__all__ = [
    "ContractViolation",
    "FixtureFormatError",
    "ModelIOContract",
    "StubPerceptionModel",
    "assert_satisfies_contract",
    "default_contract",
    "load_fixture_detections",
    "load_fixture_model",
]
