"""ML8 glue — score a run's predicted world models against an engine export.

This is the composition step that closes the offline accuracy loop: given the
world models a pipeline run produced over a recording, and an engine ground-truth
export for that same match (ML2/D6), it aligns the two by frame and delegates to
:func:`evaluate` (ML8) for the NF3 scorecard.

It runs neither capture nor the pipeline — the caller supplies the predicted world
models — so it is pure, offline, and unit-testable without a game or a model. Like
everything in this package it is never linked into the inference path (D1/D6).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from zero_ad_eyes.domain.world_model import WorldModel

from .evaluation import EvalConfig, EvaluationReport, evaluate
from .ground_truth import EngineStateExport, GroundTruthAligner


class AlignBy(StrEnum):
    """How to pair a predicted frame to its engine ground-truth frame."""

    FRAME_ID = "frame_id"  # exact frame-id equality (shared capture/engine clock)
    TIMESTAMP = "timestamp"  # nearest engine frame within a time tolerance (R5)


def evaluate_against_engine(
    predicted: Sequence[WorldModel],
    export: EngineStateExport,
    *,
    align_by: AlignBy = AlignBy.FRAME_ID,
    time_tolerance: float = 0.0,
    config: EvalConfig | None = None,
) -> EvaluationReport:
    """Align an engine export to the predicted frames, then score the NF3 metrics.

    ``align_by=FRAME_ID`` pairs each predicted world model to the engine frame with
    the same ``frame_id``; ``TIMESTAMP`` pairs it to the nearest engine frame within
    ``time_tolerance`` seconds (R5). Ground-truth labels are produced under each
    prediction's own ``FrameMeta``, so predicted and truth align by ``frame_id`` in
    :func:`evaluate` regardless of which alignment mode was used. Detection mAP stays
    ``pending-model``; this path scores only the classical metrics (HUD read error,
    ownership accuracy, tracking MOTA).
    """

    aligner = GroundTruthAligner(export)
    metas = [world_model.meta for world_model in predicted]
    if align_by is AlignBy.TIMESTAMP:
        aligned = aligner.align_by_timestamp(metas, time_tolerance)
    else:
        aligned = aligner.align_by_frame_id(metas)
    truth = [labelled.world_model for labelled in aligned]
    return evaluate(predicted, truth, config=config)
