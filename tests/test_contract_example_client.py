"""Tests for H4 — the reference consumer round-trips a JSONL contract log."""

from __future__ import annotations

from pathlib import Path

import pytest

from zero_ad_eyes.infrastructure.contract.example_client import WorldModelReader
from zero_ad_eyes.infrastructure.contract.sinks import JsonlFileWorldModelSink
from zero_ad_eyes.infrastructure.contract.versioning import IncompatibleSchemaError

from .test_contract_serialization import rich_world_model


def _write_log(path: Path, count: int) -> list:
    models = [rich_world_model(i) for i in range(count)]
    with JsonlFileWorldModelSink(path) as sink:
        for wm in models:
            sink.publish(wm)
    return models


def test_reader_round_trips_sink_output(tmp_path: Path) -> None:
    path = tmp_path / "wm.jsonl"
    models = _write_log(path, 3)

    reader = WorldModelReader(path)

    assert list(reader.read()) == models


def test_reader_latest_returns_last_model(tmp_path: Path) -> None:
    path = tmp_path / "wm.jsonl"
    models = _write_log(path, 3)

    assert WorldModelReader(path).latest() == models[-1]


def test_reader_latest_is_none_on_empty_log(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert WorldModelReader(path).latest() is None


def test_reader_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "wm.jsonl"
    models = _write_log(path, 2)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n   \n")

    assert list(WorldModelReader(path).read()) == models


def test_reader_rejects_incompatible_version(tmp_path: Path) -> None:
    path = tmp_path / "wm.jsonl"
    path.write_text(
        '{"schema_version": "9.9.9", "meta": '
        '{"frame_id": 0, "timestamp": 0.0, "source": "test", "width": 1, "height": 1}}\n',
        encoding="utf-8",
    )

    with pytest.raises(IncompatibleSchemaError):
        list(WorldModelReader(path).read())
