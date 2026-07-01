"""Tests for H1 — schema-version parsing and the compatibility policy."""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.world_model import SCHEMA_VERSION, WorldModel
from zero_ad_eyes.infrastructure.contract.versioning import (
    CURRENT_SCHEMA_VERSION,
    IncompatibleSchemaError,
    SchemaVersion,
    check_compatibility,
)

from .conftest import make_frame


def _world_model(schema_version: str) -> WorldModel:
    return WorldModel(schema_version=schema_version, meta=make_frame().meta)


def test_current_version_matches_domain_constant() -> None:
    assert str(CURRENT_SCHEMA_VERSION) == SCHEMA_VERSION


def test_parse_round_trips_to_string() -> None:
    assert str(SchemaVersion.parse("1.2.3")) == "1.2.3"


@pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "1.x.0", "", "-1.0.0"])
def test_parse_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        SchemaVersion.parse(bad)


def test_versions_are_ordered() -> None:
    assert SchemaVersion.parse("0.1.0") < SchemaVersion.parse("0.2.0")
    assert SchemaVersion.parse("1.0.0") > SchemaVersion.parse("0.9.9")


def test_same_major_is_compatible_regardless_of_minor_patch() -> None:
    producer = SchemaVersion.parse("0.4.7")
    consumer = SchemaVersion.parse("0.1.0")
    assert producer.is_compatible_with(consumer)
    assert consumer.is_compatible_with(producer)


def test_different_major_is_incompatible() -> None:
    producer = SchemaVersion.parse("1.0.0")
    consumer = SchemaVersion.parse("0.1.0")
    assert not producer.is_compatible_with(consumer)


def test_check_compatibility_passes_for_current_build() -> None:
    check_compatibility(_world_model(SCHEMA_VERSION))  # default consumer, no raise


def test_check_compatibility_raises_on_major_mismatch() -> None:
    wm = _world_model("9.0.0")
    with pytest.raises(IncompatibleSchemaError) as exc:
        check_compatibility(wm, CURRENT_SCHEMA_VERSION)
    assert exc.value.producer == SchemaVersion.parse("9.0.0")
    assert exc.value.consumer == CURRENT_SCHEMA_VERSION
