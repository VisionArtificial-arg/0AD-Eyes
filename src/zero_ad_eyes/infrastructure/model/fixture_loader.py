"""MP3 — load ``frame_id -> Detections`` fixtures from disk (JSON).

The existing ``StubPerceptionModel`` already replays an in-memory
``Mapping[int, Detections]`` (its *fixture mode*). This module is the missing
on-disk half: it reads hand-labelled / engine-ground-truth (D6) detections from a
JSON file, validates them against the frozen MP2 contract, and builds a ready
``StubPerceptionModel`` — so downstream (G/H/X) can be exercised deterministically
from a committed golden file without a trained model.

The stub is reused verbatim, not rewritten: this module only produces the mapping
it already consumes.

File format (v1)::

    {
      "contract_version": "1.0.0",
      "frames": [
        {"frame_id": 0, "items": [ <Detection>, ... ]},
        {"frame_id": 1, "items": []}
      ]
    }

Each ``frames[i]`` is exactly a serialised domain ``Detections``; each
``<Detection>`` a serialised domain ``Detection``. Pinning ``contract_version``
ties every fixture to the MP2 seam and fails loudly on drift (risk R6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zero_ad_eyes.domain.detections import Detections

from .contract import MODEL_IO_CONTRACT_V1, ContractViolation, ModelIOContract
from .stub_adapter import StubPerceptionModel


class FixtureFormatError(ValueError):
    """Raised when a fixture file is malformed or targets a different contract."""


def load_fixture_detections(
    path: str | Path,
    contract: ModelIOContract = MODEL_IO_CONTRACT_V1,
) -> dict[int, Detections]:
    """Read a fixture file into a validated ``frame_id -> Detections`` mapping.

    Every frame is parsed through the domain ``Detections`` model and then checked
    against ``contract`` (provenance + frame-id invariants), so a fixture that
    would violate the seam is rejected at load time rather than mid-pipeline.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FixtureFormatError("fixture root must be a JSON object")

    version = raw.get("contract_version")
    if version != contract.version:
        raise FixtureFormatError(
            f"fixture contract_version {version!r} != loader contract {contract.version!r}"
        )

    frames = raw.get("frames")
    if not isinstance(frames, list):
        raise FixtureFormatError("fixture 'frames' must be a JSON array")

    mapping: dict[int, Detections] = {}
    for entry in frames:
        detections = _parse_frame(entry, contract)
        if detections.frame_id in mapping:
            raise FixtureFormatError(f"duplicate frame_id {detections.frame_id} in fixture")
        mapping[detections.frame_id] = detections
    return mapping


def load_fixture_model(
    path: str | Path,
    contract: ModelIOContract = MODEL_IO_CONTRACT_V1,
) -> StubPerceptionModel:
    """Build a fixture-backed ``StubPerceptionModel`` from a JSON file (MP3).

    Convenience over ``load_fixture_detections`` for the common case: hand the
    resulting model straight to the pipeline as a drop-in ``PerceptionModel``.
    """

    return StubPerceptionModel(load_fixture_detections(path, contract))


def _parse_frame(entry: Any, contract: ModelIOContract) -> Detections:
    if not isinstance(entry, dict):
        raise FixtureFormatError("each 'frames' item must be a JSON object")
    try:
        detections = Detections.model_validate(entry)
    except Exception as exc:  # pydantic ValidationError → uniform fixture error
        raise FixtureFormatError(f"invalid Detections payload: {exc}") from exc
    try:
        contract.validate_detections(detections, detections.frame_id)
    except ContractViolation as exc:
        raise FixtureFormatError(str(exc)) from exc
    return detections
