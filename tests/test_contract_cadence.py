"""Tests for H3 — per-frame vs on-change publish cadence."""

from __future__ import annotations

from zero_ad_eyes.application.ports import WorldModelSink
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.domain.world_model import WorldModel
from zero_ad_eyes.infrastructure.contract.cadence import OnChangeSink, PerFrameSink
from zero_ad_eyes.infrastructure.contract.sinks import InMemoryWorldModelSink

from .conftest import make_frame


def _wm(frame_id: int, *, entities: tuple[Entity, ...] = ()) -> WorldModel:
    return WorldModel(meta=make_frame(frame_id).meta, entities=entities)


def test_cadence_wrappers_satisfy_the_port() -> None:
    inner = InMemoryWorldModelSink()
    assert isinstance(PerFrameSink(inner), WorldModelSink)
    assert isinstance(OnChangeSink(inner), WorldModelSink)


def test_per_frame_forwards_every_publish_even_when_unchanged() -> None:
    inner = InMemoryWorldModelSink()
    sink = PerFrameSink(inner)

    for i in range(3):
        sink.publish(_wm(i))  # identical payload, differing meta only

    assert len(inner.published) == 3


def test_on_change_suppresses_identical_payloads() -> None:
    inner = InMemoryWorldModelSink()
    sink = OnChangeSink(inner)

    for i in range(3):
        sink.publish(_wm(i))  # same payload, only frame meta differs

    assert len(inner.published) == 1  # first only
    assert inner.latest is not None and inner.latest.meta.frame_id == 0
    assert sink.suppressed == 2


def test_on_change_forwards_when_payload_changes() -> None:
    inner = InMemoryWorldModelSink()
    sink = OnChangeSink(inner)
    unit = Entity(entity_id=1, kind=EntityKind.UNIT)

    sink.publish(_wm(0))  # forwarded (first)
    sink.publish(_wm(1))  # suppressed (same payload)
    sink.publish(_wm(2, entities=(unit,)))  # forwarded (entities changed)
    sink.publish(_wm(3, entities=(unit,)))  # suppressed (same again)

    assert [wm.meta.frame_id for wm in inner.published] == [0, 2]
    assert sink.suppressed == 2


def test_on_change_treats_staleness_as_a_change() -> None:
    inner = InMemoryWorldModelSink()
    sink = OnChangeSink(inner)
    fresh = Entity(entity_id=1, kind=EntityKind.UNIT, staleness=0)
    decayed = Entity(entity_id=1, kind=EntityKind.UNIT, staleness=1)

    sink.publish(_wm(0, entities=(fresh,)))
    sink.publish(_wm(1, entities=(decayed,)))

    assert len(inner.published) == 2  # staleness decay is information
