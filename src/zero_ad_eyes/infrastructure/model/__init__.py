"""Model seam adapters (REQUIREMENTS.md §5.10).

``StubPerceptionModel`` (MP3) satisfies the ``PerceptionModel`` port without a
trained model, so the whole pipeline runs today. ``SegmentationPerceptionModel``
(MP4) is the real adapter over the delivered U-Net weights (``best.pt``); it needs
the optional ``learned`` extra (torch) and imports it lazily, so this package stays
importable on the core numpy+opencv stack. Parity tests (MP5) assert both satisfy
the identical contract.
"""

from __future__ import annotations

from .contract import ContractViolation, ModelIOContract, default_contract
from .fixture_loader import FixtureFormatError, load_fixture_detections, load_fixture_model
from .parity import assert_satisfies_contract
from .segmentation_adapter import SegmentationPerceptionModel
from .stub_adapter import StubPerceptionModel

__all__ = [
    "ContractViolation",
    "FixtureFormatError",
    "ModelIOContract",
    "SegmentationPerceptionModel",
    "StubPerceptionModel",
    "assert_satisfies_contract",
    "default_contract",
    "load_fixture_detections",
    "load_fixture_model",
]
