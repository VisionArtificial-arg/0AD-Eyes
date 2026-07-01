"""Tests for the record/replay library (REQUIREMENTS.md X2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind
from zero_ad_eyes.domain.world_model import WorldModel
from zero_ad_eyes.interface.record_replay import (
    RunRecorder,
    RunReplayer,
    record_run,
    replay_run,
)

from .conftest import make_frame


def _world_for(frame_id: int) -> tuple:
    frame = make_frame(frame_id, width=32, height=24)
    # Give each frame distinct pixels so image round-trip is meaningfully checked.
    frame.image[frame_id % 24, :, :] = 255
    wm = WorldModel(
        meta=frame.meta,
        entities=(
            Entity(
                entity_id=frame_id,
                kind=EntityKind.UNIT,
                screen_bbox=ScreenBBox(x=1, y=1, width=3, height=3),
            ),
        ),
    )
    return frame, wm


def test_record_then_replay_round_trips(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    originals = [_world_for(i) for i in range(3)]

    with RunRecorder(run_dir, source="unit-test") as recorder:
        for frame, wm in originals:
            recorder.record(frame, wm)

    replayer = RunReplayer(run_dir)
    assert len(replayer) == 3
    assert replayer.source == "unit-test"

    replayed = list(replayer)
    assert len(replayed) == 3
    for (orig_frame, orig_wm), (frame, wm) in zip(originals, replayed, strict=True):
        assert wm == orig_wm
        assert np.array_equal(frame.image, orig_frame.image)
        assert frame.meta == orig_wm.meta


def test_convenience_record_and_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    originals = [_world_for(i) for i in range(2)]

    count = record_run(run_dir, originals, source="conv")

    assert count == 2
    replayed_worlds = [wm for _, wm in replay_run(run_dir)]
    assert [wm.meta.frame_id for wm in replayed_worlds] == [0, 1]


def test_world_models_iterator(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    record_run(run_dir, (_world_for(i) for i in range(4)))

    worlds = list(RunReplayer(run_dir).world_models())

    assert [wm.meta.frame_id for wm in worlds] == [0, 1, 2, 3]
