"""Tests for H2 — codec round-trips and the three ``WorldModelSink`` adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from zero_ad_eyes.application.ports import WorldModelSink
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.hud import HudState, Population
from zero_ad_eyes.domain.minimap import Blip, MinimapModel, ViewportRect
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership, Phase, ResourceType
from zero_ad_eyes.domain.world_model import WorldModel
from zero_ad_eyes.infrastructure.contract.serialization import WorldModelCodec
from zero_ad_eyes.infrastructure.contract.sinks import (
    CallbackWorldModelSink,
    InMemoryWorldModelSink,
    JsonlFileWorldModelSink,
)
from zero_ad_eyes.infrastructure.contract.versioning import IncompatibleSchemaError

from .conftest import make_frame


def rich_world_model(frame_id: int = 0) -> WorldModel:
    """A fully-populated model: exercises every nested value object on the wire."""

    return WorldModel(
        meta=make_frame(frame_id).meta,
        hud=HudState(
            stockpiles={ResourceType.FOOD: 100, ResourceType.WOOD: 50},
            population=Population(current=12, cap=20),
            phase=Phase.TOWN,
            self_player_color=(255, 0, 0),
            self_civ="athen",
            confidence=Confidence.certain(Provenance.CLASSICAL),
        ),
        minimap=MinimapModel(
            blips=(
                Blip(
                    world_pos=WorldPoint(x=1.0, y=2.0),
                    ownership=Ownership.SELF,
                    confidence=Confidence(value=0.8, provenance=Provenance.CLASSICAL),
                ),
            ),
            viewport=ViewportRect(
                top_left=WorldPoint(x=0.0, y=0.0),
                bottom_right=WorldPoint(x=10.0, y=10.0),
            ),
            confidence=Confidence.certain(),
        ),
        entities=(
            Entity(
                entity_id=7,
                kind=EntityKind.UNIT,
                ownership=Ownership.SELF,
                entity_type="worker",
                world_pos=WorldPoint(x=3.0, y=4.0),
                screen_bbox=ScreenBBox(x=2, y=2, width=5, height=5),
                health=0.75,
                selected=True,
                staleness=1,
                confidence=Confidence(value=0.9, provenance=Provenance.LEARNED),
            ),
        ),
    )


# --- codec ------------------------------------------------------------------


def test_encode_is_single_line() -> None:
    encoded = WorldModelCodec().encode(rich_world_model())
    assert "\n" not in encoded


def test_round_trip_preserves_every_field() -> None:
    codec = WorldModelCodec()
    original = rich_world_model()

    restored = codec.decode(codec.encode(original))

    assert restored == original  # frozen pydantic → structural equality


def test_round_trip_preserves_schema_version() -> None:
    codec = WorldModelCodec()
    original = rich_world_model()
    assert codec.decode(codec.encode(original)).schema_version == original.schema_version


def test_decode_rejects_incompatible_version() -> None:
    codec = WorldModelCodec()
    incompatible = WorldModel(schema_version="9.9.9", meta=make_frame().meta)
    with pytest.raises(IncompatibleSchemaError):
        codec.decode(codec.encode(incompatible))


# --- sinks ------------------------------------------------------------------


def test_sinks_satisfy_the_port(tmp_path: Path) -> None:
    with JsonlFileWorldModelSink(tmp_path / "wm.jsonl") as file_sink:
        for sink in (
            InMemoryWorldModelSink(),
            file_sink,
            CallbackWorldModelSink(lambda _wm: None),
        ):
            assert isinstance(sink, WorldModelSink)


def test_in_memory_sink_keeps_order_and_latest() -> None:
    sink = InMemoryWorldModelSink()
    models = [rich_world_model(i) for i in range(3)]

    for wm in models:
        sink.publish(wm)

    assert sink.published == tuple(models)
    assert sink.latest == models[-1]


def test_in_memory_latest_is_none_when_empty() -> None:
    assert InMemoryWorldModelSink().latest is None


def test_callback_sink_forwards_each_publish() -> None:
    seen: list[WorldModel] = []
    sink = CallbackWorldModelSink(seen.append)

    models = [rich_world_model(i) for i in range(2)]
    for wm in models:
        sink.publish(wm)

    assert seen == models


def test_jsonl_sink_writes_one_line_per_publish_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "world.jsonl"
    codec = WorldModelCodec()
    models = [rich_world_model(i) for i in range(3)]

    with JsonlFileWorldModelSink(path, codec=codec) as sink:
        for wm in models:
            sink.publish(wm)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [codec.decode(line) for line in lines] == models


def test_jsonl_sink_append_mode_preserves_prior_content(tmp_path: Path) -> None:
    path = tmp_path / "world.jsonl"
    with JsonlFileWorldModelSink(path) as sink:
        sink.publish(rich_world_model(0))
    with JsonlFileWorldModelSink(path, append=True) as sink:
        sink.publish(rich_world_model(1))

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
