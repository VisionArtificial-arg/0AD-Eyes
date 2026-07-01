"""ML3 tests — annotation schema load/save and conversion to port detections."""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.minimap import FogState
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership, Phase, ResourceType
from zero_ad_eyes.infrastructure.data.annotation import (
    AnnotationSet,
    DetectionLabel,
    FogCellLabel,
    FrameAnnotation,
    HudLabel,
    MinimapBlipLabel,
)


def _annotation() -> FrameAnnotation:
    return FrameAnnotation(
        frame_id=0,
        timestamp=0.0,
        image_path="frames/frame_000000.npy",
        detections=(
            DetectionLabel(
                kind=EntityKind.UNIT,
                bbox=ScreenBBox(x=10, y=10, width=8, height=16),
                entity_type="female_citizen",
                ownership=Ownership.SELF,
                health=0.75,
            ),
            DetectionLabel(
                kind=EntityKind.BUILDING,
                bbox=ScreenBBox(x=30, y=5, width=20, height=20),
                ownership=Ownership.ENEMY,
            ),
        ),
        minimap_blips=(
            MinimapBlipLabel(world_pos=WorldPoint(x=1.0, y=2.0), ownership=Ownership.ALLY),
        ),
        fog_cells=(FogCellLabel(cell_x=3, cell_y=4, state=FogState.VISIBLE),),
        hud=HudLabel(
            stockpiles={ResourceType.FOOD: 100, ResourceType.WOOD: 50},
            population_current=8,
            population_cap=20,
            phase=Phase.TOWN,
        ),
    )


def test_to_detections_maps_labels_to_port_shape() -> None:
    detections = _annotation().to_detections()

    assert detections.frame_id == 0
    assert len(detections) == 2
    first = detections.items[0]
    assert first.kind == EntityKind.UNIT
    assert first.entity_type == "female_citizen"
    assert first.confidence.provenance == Provenance.ENGINE_GT
    assert first.confidence.value == 1.0


def test_annotation_set_save_load_round_trip(tmp_path: Path) -> None:
    annotation_set = AnnotationSet(annotations=(_annotation(),))
    path = tmp_path / "labels.json"

    annotation_set.save(path)
    reloaded = AnnotationSet.load(path)

    assert reloaded == annotation_set
    assert len(reloaded) == 1


def test_by_frame_id_index() -> None:
    annotation_set = AnnotationSet(
        annotations=(
            _annotation(),
            FrameAnnotation(frame_id=5),
        )
    )

    index = annotation_set.by_frame_id()

    assert set(index) == {0, 5}
    assert index[5].detections == ()
