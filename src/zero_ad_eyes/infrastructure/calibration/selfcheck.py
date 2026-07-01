"""Calibration self-check (EPIC B — B4).

A stored/reused calibration silently goes stale when the live layout changes
(resolution switch, UI-scale change, theme swap, HUD mod). ``LayoutSelfCheck``
re-derives the cheap pixel anchors on a *live* frame and compares them against a
calibration, returning a bounded confidence and a boolean verdict (REQUIREMENTS.md
B4). It never mutates or recomputes the calibration — it only reports agreement, so
callers decide whether to trust the profile or trigger re-calibration.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.calibration import Calibration

from .anchors import bottom_band_fraction, top_band_fraction

# Relative deviation at which an anchor is considered a total mismatch (score 0). A
# deviation of half this tolerance still scores 0.5, so the mapping degrades linearly
# rather than cliff-edging.
_AGREEMENT_TOLERANCE = 0.6
_EPS = 1e-9
_DEFAULT_MATCH_THRESHOLD = 0.5
# Resolution matches but no anchors could be recovered to confirm the layout: we can
# neither confirm nor deny, so report a deliberately middling confidence.
_UNVERIFIABLE_CONFIDENCE = 0.5


class CalibrationCheck(BaseModel):
    """Verdict of comparing a live frame against a calibration."""

    model_config = ConfigDict(frozen=True)

    matches: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


def _agreement(observed: float, expected: float) -> float:
    rel_err = abs(observed - expected) / max(abs(expected), _EPS)
    return max(0.0, min(1.0, 1.0 - rel_err / _AGREEMENT_TOLERANCE))


class LayoutSelfCheck:
    """Flags when a live frame no longer matches a calibration (B4)."""

    def __init__(
        self,
        *,
        match_threshold: float = _DEFAULT_MATCH_THRESHOLD,
        use_anchors: bool = True,
    ) -> None:
        self._match_threshold = match_threshold
        self._use_anchors = use_anchors

    def verify(self, frame: Frame, calibration: Calibration) -> CalibrationCheck:
        width, height = self._resolution(frame)
        if (width, height) != (calibration.width, calibration.height):
            return CalibrationCheck(
                matches=False,
                confidence=0.0,
                reason=(
                    f"resolution mismatch: frame {width}x{height} "
                    f"vs calibration {calibration.width}x{calibration.height}"
                ),
            )

        scores = self._anchor_scores(frame, calibration)
        if not scores:
            return CalibrationCheck(
                matches=True,
                confidence=_UNVERIFIABLE_CONFIDENCE,
                reason="resolution matches; no anchors available to verify layout",
            )

        confidence = sum(scores) / len(scores)
        matches = confidence >= self._match_threshold
        return CalibrationCheck(
            matches=matches,
            confidence=confidence,
            reason=f"{len(scores)} anchor(s) compared; mean agreement {confidence:.3f}",
        )

    def _anchor_scores(self, frame: Frame, calibration: Calibration) -> list[float]:
        if not self._use_anchors or frame.image is None:
            return []

        scores: list[float] = []
        top = top_band_fraction(frame.image)
        if top is not None and calibration.top_bar is not None:
            expected = calibration.top_bar.height / calibration.height
            scores.append(_agreement(top, expected))

        bottom = bottom_band_fraction(frame.image)
        if bottom is not None and calibration.minimap is not None:
            expected = calibration.minimap.height / calibration.height
            scores.append(_agreement(bottom, expected))

        return scores

    def _resolution(self, frame: Frame) -> tuple[int, int]:
        image: Any = frame.image
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
        return frame.meta.width, frame.meta.height
