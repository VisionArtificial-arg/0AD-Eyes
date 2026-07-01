"""Robustness of number reading (EPIC C — C6).

Digit-glyph confusions, thousands formatting, value bounds, and rejection of
transient tooltips / notifications that happen to contain numbers.
"""

from __future__ import annotations

import pytest

from zero_ad_eyes.infrastructure.hud.parsing import parse_count, parse_population


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("35O", 350),  # letter O misread for zero
        ("12O", 120),
        ("l2", 12),  # lowercase L misread for one
        ("I0", 10),  # capital I misread for one
        ("|5", 15),  # pipe misread for one
        ("1,234", 1234),
    ],
)
def test_parse_count_repairs_glyph_confusions(text: str, expected: int) -> None:
    assert parse_count(text) == expected


def test_glyph_repair_never_fabricates_from_pure_letters() -> None:
    # "food"/"Town" must NOT become digits despite containing O/o.
    assert parse_count("food") is None
    assert parse_count("Town") is None


@pytest.mark.parametrize(
    "tooltip",
    [
        "Build cost 50 wood 20 stone",
        "Enemy units spotted near your base",
        "Right-click to gather 5 food per trip",
    ],
)
def test_transient_tooltip_or_notification_is_rejected(tooltip: str) -> None:
    assert parse_count(tooltip) is None


def test_out_of_range_value_rejected_as_garbage() -> None:
    assert parse_count("99999999") is None
    assert parse_count("500", max_value=100) is None


def test_population_repairs_glyphs() -> None:
    assert parse_population("l2/2O") == (12, 20)
