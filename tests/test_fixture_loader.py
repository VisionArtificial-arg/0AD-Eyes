"""MP3 — tests for the on-disk JSON fixture loader.

Writes a temp fixture, loads it, and infers through the stub the loader builds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.infrastructure.model.contract import CONTRACT_VERSION
from zero_ad_eyes.infrastructure.model.fixture_loader import (
    FixtureFormatError,
    load_fixture_detections,
    load_fixture_model,
)

from .conftest import make_frame


def _detections(frame_id: int, provenance: Provenance = Provenance.ENGINE_GT) -> Detections:
    det = Detection(
        kind=EntityKind.RESOURCE_NODE,
        bbox=ScreenBBox(x=1, y=2, width=3, height=4),
        confidence=Confidence(value=0.9, provenance=provenance),
        entity_type="oak_tree",
    )
    return Detections(frame_id=frame_id, items=(det,))


def _write_fixture(path: Path, *frames: Detections, version: str = CONTRACT_VERSION) -> Path:
    payload = {
        "contract_version": version,
        "frames": [f.model_dump(mode="json") for f in frames],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_fixture_detections_round_trips(tmp_path: Path) -> None:
    original = _detections(0)
    fixture = _write_fixture(tmp_path / "golden.json", original, _detections(1))

    mapping = load_fixture_detections(fixture)

    assert set(mapping) == {0, 1}
    assert mapping[0] == original


def test_loaded_model_replays_fixture_for_its_frame(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "golden.json", _detections(7))

    model = load_fixture_model(fixture)
    result = model.infer(make_frame(7))

    assert result.frame_id == 7
    assert result.items[0].entity_type == "oak_tree"
    assert result.items[0].confidence.provenance is Provenance.ENGINE_GT


def test_unknown_frame_yields_empty_detections(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "golden.json", _detections(0))

    model = load_fixture_model(fixture)
    result = model.infer(make_frame(99))

    assert len(result) == 0
    assert result.frame_id == 99


def test_version_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "old.json", _detections(0), version="0.0.1")
    with pytest.raises(FixtureFormatError):
        load_fixture_detections(fixture)


def test_contract_violating_fixture_is_rejected(tmp_path: Path) -> None:
    bad = _detections(0, provenance=Provenance.UNKNOWN)  # not an allowed provenance
    fixture = _write_fixture(tmp_path / "bad.json", bad)
    with pytest.raises(FixtureFormatError):
        load_fixture_detections(fixture)


def test_duplicate_frame_id_is_rejected(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "dup.json", _detections(0), _detections(0))
    with pytest.raises(FixtureFormatError):
        load_fixture_detections(fixture)
