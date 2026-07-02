"""OCR seam for HUD parsing (EPIC C — C1).

The HUD reader never talks to an OCR library directly; it depends on the
``OcrEngine`` protocol so the concrete engine is *injectable*. Tests pass a stub
that returns canned text, and production passes :class:`TesseractOcrEngine`.

``pytesseract`` (and the ``tesseract`` system binary it wraps) is imported
*lazily*, inside :meth:`TesseractOcrEngine.read_text`, so that importing this
module — and running the whole test suite — never requires tesseract to be
installed. Environments without the binary can still exercise every cropping,
parsing and normalisation path with a stub engine.
"""

from __future__ import annotations

import shutil
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OcrEngine(Protocol):
    """Turns an image crop into a best-effort text string.

    ``image`` is an HxWxC ``numpy.ndarray`` (BGR, OpenCV convention), matching
    :class:`~zero_ad_eyes.application.frames.Frame`. Implementations must never
    raise on unreadable input; they return ``""`` instead (NF4: degrade, never
    crash the consumer).
    """

    def read_text(self, image: Any) -> str: ...


class TesseractOcrEngine:
    """Production :class:`OcrEngine` backed by ``pytesseract`` (lazy import).

    The ``tesseract`` binary is an *external* system dependency. To keep import
    time and the test suite free of it, ``pytesseract`` is imported only when
    :meth:`read_text` is first called. If the binary is missing the engine
    degrades to ``""`` rather than raising, so callers stay alive (NF4).
    """

    def __init__(self, *, config: str) -> None:
        # Tesseract page-segmentation config (e.g. "--psm 7" = single text line,
        # the common case for a HUD counter). Supplied from the ``hud`` config.
        self._config = config

    @staticmethod
    def is_available() -> bool:
        """Whether the ``tesseract`` binary is on PATH."""

        return shutil.which("tesseract") is not None

    def read_text(self, image: Any) -> str:
        if not self.is_available():
            return ""
        try:
            import pytesseract  # lazy: only needed in production runs
        except ImportError:
            return ""
        try:
            text = pytesseract.image_to_string(image, config=self._config)
        except Exception:
            # Any tesseract-side failure (bad crop, runtime error) degrades to
            # empty text; the parser then yields None and confidence drops.
            return ""
        return str(text).strip()
