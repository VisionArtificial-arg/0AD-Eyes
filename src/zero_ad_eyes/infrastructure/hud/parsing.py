"""Pure text-to-value parsers for HUD readings (EPIC C — C1).

These functions are deliberately free of OpenCV, tesseract and numpy: they take
the *string* an OCR engine produced and turn it into a domain value. That makes
the hard part — number extraction and its robustness (C6) — unit-testable with no
system dependencies at all.
"""

from __future__ import annotations

import re

from zero_ad_eyes.domain.taxonomy import Phase

# A grouping separator is a comma, dot, plain space or unicode thin/no-break space
# sitting *between two digits* — "1,234", "12 345", "1.234". Only then is it a
# thousands separator to drop; a space elsewhere just delimits words.
_GROUPING = re.compile(r"(?<=\d)[,.   ](?=\d)")
_DIGITS = re.compile(r"\d+")
# "current / cap": two integer runs separated by a slash, tolerant of spaces and
# of leading label noise ("Pop 12 / 20").
_RATIO = re.compile(r"(\d+)\s*/\s*(\d+)")

# Robustness (C6):
_ALPHA_WORD = re.compile(r"[A-Za-z]{2,}")
# Glyph confusions tesseract makes on small HUD digits. Applied ONLY to tokens
# that already contain a digit, so genuine words ("food", "Town") are never
# turned into spurious numbers.
_GLYPH_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})
_HAS_DIGIT = re.compile(r"\d")
# Counters above this are treated as OCR garbage rather than a real stockpile.
_MAX_COUNT = 1_000_000
# More alphabetic words than this ⇒ prose (a tooltip/notification), not a counter.
_MAX_LABEL_WORDS = 3


def _repair_glyphs(text: str) -> str:
    """Fix digit-glyph confusions inside numeric tokens only."""

    return " ".join(
        token.translate(_GLYPH_FIXES) if _HAS_DIGIT.search(token) else token
        for token in text.split()
    )


def parse_count(text: str, *, max_value: int = _MAX_COUNT) -> int | None:
    """Extract a single non-negative integer counter from OCR ``text``.

    Robust (C6) to: thousands grouping (``"1,234"`` / ``"12 345"`` / ``"1.234"``),
    surrounding label/icon noise (``"food 350"``), digit-glyph confusions
    (``"35O"`` → 350, ``"l2"`` → 12), and transient tooltips/notifications — a
    reading with more than a few alphabetic words is prose and yields ``None``
    rather than a number lifted out of a sentence. Values above ``max_value`` are
    rejected as garbage. Returns ``None`` when nothing numeric remains.
    """

    if not text:
        return None
    if len(_ALPHA_WORD.findall(text)) > _MAX_LABEL_WORDS:
        return None  # a tooltip / notification, not a counter
    repaired = _repair_glyphs(text)
    cleaned = _GROUPING.sub("", repaired)
    match = _DIGITS.search(cleaned)
    if match is None:
        return None
    value = int(match.group())
    if value > max_value:
        return None
    return value


def parse_population(text: str) -> tuple[int, int] | None:
    """Extract ``(current, cap)`` from a population reading like ``"12/20"``.

    Tolerates spaces around the slash and leading label noise (``"Pop 12 / 20"``).
    Returns ``None`` unless both numbers are present, so a transient single-number
    frame never fabricates a cap. Reader maps the tuple to
    :class:`~zero_ad_eyes.domain.hud.Population`.
    """

    if not text:
        return None
    cleaned = _GROUPING.sub("", _repair_glyphs(text))
    match = _RATIO.search(cleaned)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_health(text: str) -> tuple[int, int] | None:
    """Extract ``(current, maximum)`` hit points from a ``"120/150"`` reading.

    Same current/maximum-ratio grammar as population; kept as a named function so
    the selection reader reads intention-revealingly. Returns ``None`` unless both
    numbers are present.
    """

    return parse_population(text)


# Ordered longest/most-specific first so a substring match is unambiguous.
_PHASE_KEYWORDS: tuple[tuple[str, Phase], ...] = (
    ("village", Phase.VILLAGE),
    ("town", Phase.TOWN),
    ("city", Phase.CITY),
)


def normalize_phase(text: str) -> Phase:
    """Map OCR'd phase text (``"Town Phase"``, ``"City"``) to a :class:`Phase`.

    Case-insensitive keyword match; unknown/empty text yields ``Phase.UNKNOWN`` so
    the reader degrades rather than guessing (NF4). Template-icon matching is a
    valid alternative route to the same value and can be swapped in behind this
    function without changing the reader.
    """

    if not text:
        return Phase.UNKNOWN
    lowered = text.casefold()
    for keyword, phase in _PHASE_KEYWORDS:
        if keyword in lowered:
            return phase
    return Phase.UNKNOWN


# Default civ keyword → canonical 0 A.D. civ id. Config-driven (NF7): callers may
# pass their own mapping for modded/extended civ sets.
DEFAULT_CIVS: dict[str, str] = {
    "athen": "athen",
    "briton": "brit",
    "carthag": "cart",
    "gaul": "gaul",
    "iberian": "iber",
    "kushite": "kush",
    "macedon": "mace",
    "maurya": "maur",
    "persian": "pers",
    "ptolem": "ptol",
    "roman": "rome",
    "seleucid": "sele",
    "spartan": "spart",
}


def normalize_civ(text: str, civs: dict[str, str] | None = None) -> str | None:
    """Resolve OCR'd civ text to a canonical civ id, or ``None`` if unrecognised.

    Case-insensitive keyword containment against ``civs`` (defaults to
    :data:`DEFAULT_CIVS`). Keywords are stems (``"roman"`` matches "Romans") so
    minor OCR tails do not break the match.
    """

    if not text:
        return None
    lowered = text.casefold()
    for keyword, civ_id in (civs or DEFAULT_CIVS).items():
        if keyword in lowered:
            return civ_id
    return None
