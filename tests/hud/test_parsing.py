"""Unit tests for OCR number parsing (EPIC C — C1).

Pure-string tests: no tesseract, no frames. Robustness cases (glyph confusions,
bounds, tooltips) live in ``test_robustness.py`` (C6).
"""

from __future__ import annotations

import pytest

from zero_ad_eyes.domain.taxonomy import Phase
from zero_ad_eyes.infrastructure.hud.parsing import (
    normalize_civ,
    normalize_phase,
    parse_count,
    parse_population,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("350", 350),
        ("0", 0),
        ("1,234", 1234),
        ("12,345", 12345),
        ("1.234", 1234),  # dot as thousands grouping
        ("12 345", 12345),  # space grouping
        ("food 350", 350),  # icon/label noise before the number
        ("  87  ", 87),
    ],
)
def test_parse_count_reads_integers(text: str, expected: int) -> None:
    assert parse_count(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "food", "---"])
def test_parse_count_returns_none_without_digits(text: str) -> None:
    assert parse_count(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12/20", (12, 20)),
        ("12 / 20", (12, 20)),
        ("Pop 12/20", (12, 20)),
        ("0/300", (0, 300)),
    ],
)
def test_parse_population_reads_current_and_cap(text: str, expected: tuple[int, int]) -> None:
    assert parse_population(text) == expected


@pytest.mark.parametrize("text", ["", "12", "no numbers", "12-20"])
def test_parse_population_requires_both_numbers(text: str) -> None:
    assert parse_population(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Village Phase", Phase.VILLAGE),
        ("Town Phase", Phase.TOWN),
        ("City", Phase.CITY),
        ("CITY PHASE", Phase.CITY),
        ("", Phase.UNKNOWN),
        ("gibberish", Phase.UNKNOWN),
    ],
)
def test_normalize_phase(text: str, expected: Phase) -> None:
    assert normalize_phase(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Romans", "rome"),
        ("romans", "rome"),
        ("Spartans", "spart"),
        ("Ptolemies", "ptol"),
        ("", None),
        ("Klingons", None),
    ],
)
def test_normalize_civ(text: str, expected: str | None) -> None:
    assert normalize_civ(text) == expected
