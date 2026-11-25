"""EVAL-1 — scorecard: the eyes' output vs engine truth, unmeasured never a pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.events import Event, EventKind
from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.domain.minimap import Blip, MinimapModel
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel
from zero_ad_eyes.infrastructure.data.engine_dataset import EngineDatasetRun, EngineRunSource
from zero_ad_eyes.infrastructure.data.ground_truth import EngineStateExport, GroundTruthAligner
from zero_ad_eyes.infrastructure.data.scorecard import Scorecard, score_run
from zero_ad_eyes.interface.cli import main

from ._engine_fixtures import write_run

FINE = {101: "unit", 202: "building"}


def _export(tmp_path: Path) -> EngineStateExport:
    return EngineDatasetRun.load(write_run(tmp_path, frames=3)).to_export()


def _meta(frame_id: int) -> FrameMeta:
    return FrameMeta(frame_id=frame_id, timestamp=float(frame_id), source="t", width=60, height=40)


def _perfect(export: EngineStateExport) -> list[WorldModel]:
    """Predictions equal to the truth, as the eyes would emit them."""

    aligner = GroundTruthAligner(export)
    models = []
    for frame in export.frames:
        truth = aligner.labels_for(frame, _meta(frame.frame_id)).world_model
        entities = tuple(
            e.model_copy(
                update={
                    "entity_type": FINE.get(e.entity_id),
                    "confidence": Confidence(value=0.9, provenance=Provenance.LEARNED),
                }
            )
            for e in truth.entities
            if e.screen_bbox is not None
        )
        models.append(truth.model_copy(update={"entities": entities}))
    return models


def _score(predicted: list[WorldModel], export: EngineStateExport) -> Scorecard:
    return score_run(predicted, export, iou_threshold=0.5, blip_radius=8.0)


def test_perfect_predictions_score_perfectly(tmp_path: Path) -> None:
    export = _export(tmp_path)
    card = _score(_perfect(export), export)
    assert card.frames == 3
    assert card.detection_coarse.map50 == pytest.approx(1.0)
    assert card.detection_coarse.map50_95 == pytest.approx(1.0)
    assert card.detection_fine.ap50 == {"building": pytest.approx(1.0), "unit": pytest.approx(1.0)}
    assert card.detection_fine.truth_instances == {"building": 3, "unit": 3}
    assert card.ownership_accuracy.value == 1.0 and card.ownership_accuracy.n == 6
    assert card.tracking_mota == pytest.approx(1.0)
    assert card.hud_exact["food"].value == 1.0
    assert card.hud_exact["population_cap"].value == 1.0
    assert card.hud_mean_relative_error.value == 0.0


def test_empty_predictions_score_zero_and_leave_the_unmeasurable_unmeasured(
    tmp_path: Path,
) -> None:
    export = _export(tmp_path)
    card = _score([WorldModel(meta=_meta(f.frame_id)) for f in export.frames], export)
    assert card.detection_coarse.map50 == 0.0
    assert card.ownership_accuracy.value == 0.0  # every truth entity missed
    assert card.hud_exact["food"].value == 0.0  # HUD not read = wrong
    assert card.hud_mean_relative_error.value == 1.0
    assert card.health_mae.value is None and card.health_mae.n == 0
    assert card.minimap_blip_recall.value is None  # no minimap read → unmeasured


def test_stale_memory_is_not_scored_as_a_detection(tmp_path: Path) -> None:
    export = _export(tmp_path)
    stale = [
        m.model_copy(
            update={"entities": tuple(e.model_copy(update={"staleness": 3}) for e in m.entities)}
        )
        for m in _perfect(export)
    ]
    assert _score(stale, export).detection_coarse.map50 == 0.0


def test_minimap_blips_match_truth_within_radius(tmp_path: Path) -> None:
    root = write_run(tmp_path, frames=1)
    truth_file = root / "truth" / "000000.json"
    truth = json.loads(truth_file.read_text())
    truth["entities"][0].update(world_x=100.0, world_y=50.0)  # unit 101, visible
    truth["entities"][1].update(world_x=300.0, world_y=300.0)  # building 202, visible
    truth_file.write_text(json.dumps(truth))
    export = EngineDatasetRun.load(root).to_export()

    def blip(x: float, y: float) -> Blip:
        return Blip(
            world_pos=WorldPoint(x=x, y=y),
            ownership=Ownership.SELF,
            confidence=Confidence.certain(),
        )

    minimap = MinimapModel(blips=(blip(103.0, 52.0), blip(500.0, 500.0)))
    card = _score([WorldModel(meta=_meta(0), minimap=minimap)], export)
    assert card.minimap_blip_precision.value == 0.5  # one of two blips is real
    assert card.minimap_blip_recall.value == 0.5  # one of two visible entities found


def test_events_are_profiled_per_kind(tmp_path: Path) -> None:
    export = _export(tmp_path)
    events = (Event(kind=EventKind.COMBAT, frame_id=0), Event(kind=EventKind.COMBAT, frame_id=0))
    card = _score([WorldModel(meta=_meta(0), events=events)], export)
    assert card.events_per_kind == {"combat": 2}


def test_engine_run_source_stamps_frames_with_truth_ids(tmp_path: Path) -> None:
    run = EngineDatasetRun.load(write_run(tmp_path, frames=3))
    metas = [f.meta for f in EngineRunSource(run).frames()]
    assert [m.frame_id for m in metas] == [0, 1, 2]
    assert {m.source for m in metas} == {"a1b2c3d4"}


def test_cli_eval_engine_run_prints_and_writes_the_scorecard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_run(tmp_path, frames=2)
    out = tmp_path / "card.json"
    assert main(["eval", "--engine-run", str(root), "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "frames scored: 2" in printed and "detection (coarse)" in printed
    assert json.loads(out.read_text())["frames"] == 2
