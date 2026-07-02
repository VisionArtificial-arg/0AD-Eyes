"""Integration (A5 + ML2): the record sidecar is what lets a replay align to GT (#2).

Ties the acquisition reader (RecordedVideoSource) to the alignment layer
(GroundTruthAligner): a recording replayed *with* its sidecar regains the capture's
real frame ids and aligns to an engine export keyed to that clock; the same frames
replayed *without* it (the 0..N a bare video yields) do not.
"""

from __future__ import annotations

from zero_ad_eyes.infrastructure.acquisition import (
    FrameStamp,
    InMemoryFrameSource,
    RecordedVideoSource,
    RecordingManifest,
)
from zero_ad_eyes.infrastructure.data.ground_truth import (
    EngineFrameState,
    EngineStateExport,
    GroundTruthAligner,
)

from .conftest import make_frame


def _engine_export_keyed_to_real_ids() -> EngineStateExport:
    # Ground truth keyed to the capture's OWN frame ids (100, 101), not 0..N.
    return EngineStateExport(
        match_id="demo",
        self_player=1,
        frames=(
            EngineFrameState(frame_id=100, timestamp=12.5),
            EngineFrameState(frame_id=101, timestamp=12.9),
        ),
    )


def test_sidecar_replay_aligns_to_engine_where_a_bare_video_would_not() -> None:
    aligner = GroundTruthAligner(_engine_export_keyed_to_real_ids())

    # What a bare VideoFileSource yields: frames renumbered 0, 1 (PTS clock). These
    # do not match the engine's real ids 100, 101 — alignment produces nothing.
    bare = InMemoryFrameSource([make_frame(0), make_frame(1)])
    bare_metas = [frame.meta for frame in bare.frames()]
    assert aligner.align_by_frame_id(bare_metas) == ()

    # The same recording replayed through the sidecar restores 100, 101 → it aligns.
    manifest = RecordingManifest(
        video="live.mkv",
        source="live",
        fps=30.0,
        stamps=(FrameStamp(frame_id=100, timestamp=12.5), FrameStamp(frame_id=101, timestamp=12.9)),
    )
    restamped = RecordedVideoSource("live.mkv", video=bare, manifest=manifest)
    restamped_metas = [frame.meta for frame in restamped.frames()]

    aligned = aligner.align_by_frame_id(restamped_metas)
    assert [label.frame_id for label in aligned] == [100, 101]
