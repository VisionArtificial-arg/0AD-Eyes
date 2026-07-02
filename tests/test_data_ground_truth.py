"""ML2 tests — engine-state JSON schema → domain labels, aligned to frames (D6)."""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership, Phase, ResourceType
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.data.ground_truth import (
    EngineEntityState,
    EngineFrameState,
    EngineStateExport,
    GroundTruthAligner,
)


def _meta(frame_id: int, timestamp: float) -> FrameMeta:
    return FrameMeta(frame_id=frame_id, timestamp=timestamp, source="rec-1", width=64, height=48)


def _export() -> EngineStateExport:
    frame0 = EngineFrameState(
        frame_id=0,
        timestamp=0.0,
        phase=Phase.VILLAGE,
        resources={ResourceType.FOOD: 300, ResourceType.WOOD: 200},
        population_current=8,
        population_cap=20,
        entities=(
            EngineEntityState(
                entity_id=1,
                kind=EntityKind.UNIT,
                entity_type="female_citizen",
                owner=1,
                health_current=20.0,
                health_max=40.0,
                world_x=12.5,
                world_y=30.0,
                bbox=ScreenBBox(x=10, y=10, width=8, height=16),
            ),
            EngineEntityState(  # enemy, no bbox → excluded from detection labels
                entity_id=2,
                kind=EntityKind.BUILDING,
                owner=3,
                health_current=100.0,
                health_max=100.0,
                world_x=40.0,
                world_y=5.0,
            ),
            EngineEntityState(  # gaia resource node
                entity_id=3,
                kind=EntityKind.RESOURCE_NODE,
                owner=0,
                world_x=1.0,
                world_y=1.0,
                bbox=ScreenBBox(x=0, y=0, width=4, height=4),
            ),
        ),
    )
    return EngineStateExport(
        match_id="demo-001", self_player=1, ally_players=(2,), frames=(frame0,)
    )


def test_ownership_mapping() -> None:
    aligner = GroundTruthAligner(_export())
    assert aligner.ownership_of(1) == Ownership.SELF
    assert aligner.ownership_of(2) == Ownership.ALLY
    assert aligner.ownership_of(3) == Ownership.ENEMY
    assert aligner.ownership_of(0) == Ownership.GAIA


def test_labels_world_model_and_detections() -> None:
    aligner = GroundTruthAligner(_export())

    aligned = aligner.align_by_frame_id([_meta(0, 0.0)])

    assert len(aligned) == 1
    label = aligned[0]

    # World model carries all three entities with correct ownership + health.
    entities = {e.entity_id: e for e in label.world_model.entities}
    assert len(entities) == 3
    assert entities[1].ownership == Ownership.SELF
    assert entities[1].health == 0.5  # 20 / 40
    assert entities[2].ownership == Ownership.ENEMY  # owner 3 not in allies (2,)
    assert entities[2].health == 1.0
    assert entities[3].ownership == Ownership.GAIA
    assert entities[3].health is None  # no health_max → unknown fraction

    # HUD is populated from the engine export.
    assert label.world_model.hud is not None
    assert label.world_model.hud.stockpiles[ResourceType.FOOD] == 300
    assert label.world_model.hud.phase == Phase.VILLAGE
    assert label.world_model.hud.confidence.provenance == Provenance.ENGINE_GT

    # Only entities with a projected bbox become detection labels (ids 1 and 3).
    detected_kinds = {item.kind for item in label.detections.items}
    assert len(label.detections) == 2
    assert detected_kinds == {EntityKind.UNIT, EntityKind.RESOURCE_NODE}
    assert all(
        item.confidence.provenance == Provenance.ENGINE_GT for item in label.detections.items
    )


def test_align_by_frame_id_skips_unmatched() -> None:
    aligner = GroundTruthAligner(_export())

    aligned = aligner.align_by_frame_id([_meta(0, 0.0), _meta(99, 9.0)])

    assert [label.frame_id for label in aligned] == [0]


def test_align_by_timestamp_nearest_within_tolerance() -> None:
    aligner = GroundTruthAligner(_export())

    # Captured at t=0.05; engine frame at t=0.0 is within tolerance 0.1.
    within = aligner.align_by_timestamp([_meta(5, 0.05)], tolerance=0.1)
    assert len(within) == 1
    assert within[0].world_model.meta.frame_id == 5  # keeps the capture's id

    # Captured at t=5.0 is outside tolerance → no label.
    outside = aligner.align_by_timestamp([_meta(6, 5.0)], tolerance=0.1)
    assert outside == ()


def test_export_json_round_trip(tmp_path: Path) -> None:
    export = _export()
    path = tmp_path / "gt.json"

    export.save(path)
    reloaded = EngineStateExport.load(path)

    assert reloaded == export


# --- Alignment on real-capture clocks (drift hardening, R5) --------------------- #


def test_align_by_frame_id_matches_non_zero_based_capture_ids() -> None:
    # A real capture's frame ids do not start at 0; the sidecar restores them and the
    # engine export is keyed to the same ids. Exact-id alignment must still pair them.
    export = EngineStateExport(
        match_id="m",
        self_player=1,
        frames=(
            EngineFrameState(frame_id=100, timestamp=12.5),
            EngineFrameState(frame_id=101, timestamp=12.9),
        ),
    )
    aligner = GroundTruthAligner(export)

    aligned = aligner.align_by_frame_id([_meta(100, 12.5), _meta(101, 12.9)])

    assert [label.frame_id for label in aligned] == [100, 101]


def test_align_by_timestamp_maps_several_captures_to_one_engine_frame() -> None:
    # Capture fps > engine-export fps: several captured frames fall within tolerance of
    # the same engine frame. Each keeps its own id and is scored against that frame.
    export = EngineStateExport(
        match_id="m",
        self_player=1,
        frames=(
            EngineFrameState(frame_id=0, timestamp=0.0),
            EngineFrameState(frame_id=1, timestamp=0.5),
        ),
    )
    aligner = GroundTruthAligner(export)

    aligned = aligner.align_by_timestamp(
        [_meta(10, 0.00), _meta(11, 0.03), _meta(12, 0.06)], tolerance=0.1
    )

    assert [label.frame_id for label in aligned] == [10, 11, 12]


def test_align_by_timestamp_tolerance_boundary_is_inclusive() -> None:
    export = EngineStateExport(
        match_id="m", self_player=1, frames=(EngineFrameState(frame_id=0, timestamp=0.0),)
    )
    aligner = GroundTruthAligner(export)

    # A delta exactly equal to the tolerance still matches; just beyond it does not.
    assert len(aligner.align_by_timestamp([_meta(1, 0.10)], tolerance=0.1)) == 1
    assert aligner.align_by_timestamp([_meta(1, 0.11)], tolerance=0.1) == ()
