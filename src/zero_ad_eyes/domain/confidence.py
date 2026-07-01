"""Confidence and provenance — attached to every perceived fact.

Per REQUIREMENTS.md §4 and MP2: every field the layer emits carries how sure we
are (``Confidence``) and where it came from (``Provenance``), so downstream
consumers weight a coarse classical guess differently from a confident learned
detection — identically whether it came from the stub or the real model adapter.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Provenance(StrEnum):
    """Where a perceived fact originated."""

    CLASSICAL = "classical"  # deterministic CV (template/contour/color)
    LEARNED = "learned"  # the trained model (🔌)
    ENGINE_GT = "engine_gt"  # engine ground truth — offline/eval ONLY (D6)
    FUSED = "fused"  # combined from multiple sources
    UNKNOWN = "unknown"


class Confidence(BaseModel):
    """A bounded [0, 1] confidence with its provenance."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0)
    provenance: Provenance = Provenance.UNKNOWN

    @classmethod
    def certain(cls, provenance: Provenance = Provenance.CLASSICAL) -> Confidence:
        return cls(value=1.0, provenance=provenance)

    @classmethod
    def unknown(cls) -> Confidence:
        return cls(value=0.0, provenance=Provenance.UNKNOWN)
