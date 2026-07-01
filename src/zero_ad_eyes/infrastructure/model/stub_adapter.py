"""Stub ``PerceptionModel`` adapter (MP3) — no trained model required.

Two modes (REQUIREMENTS.md §5.10.2):
- *empty* (default): returns no detections. Lets tracking/fusion/contract/overlay
  run end-to-end for wiring and smoke tests.
- *fixture*: replays a pre-supplied mapping of ``frame_id -> Detections`` (e.g.
  engine ground truth per D6, or hand-labelled frames), so downstream accuracy and
  behaviour can be exercised deterministically without the model.

Every detection this stub emits is tagged ``Provenance.CLASSICAL`` (or whatever the
fixture carries) — never ``LEARNED`` — so provenance honestly reflects the source.
"""

from __future__ import annotations

from collections.abc import Mapping

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.detections import Detections
from zero_ad_eyes.domain.geometry import ScreenBBox


class StubPerceptionModel:
    """A ``PerceptionModel`` that needs no weights (satisfies the port structurally)."""

    def __init__(self, fixtures: Mapping[int, Detections] | None = None) -> None:
        self._fixtures = dict(fixtures) if fixtures else {}

    def infer(self, frame: Frame, roi: ScreenBBox | None = None) -> Detections:
        fixed = self._fixtures.get(frame.meta.frame_id)
        if fixed is not None:
            return fixed
        return Detections(frame_id=frame.meta.frame_id, items=())
