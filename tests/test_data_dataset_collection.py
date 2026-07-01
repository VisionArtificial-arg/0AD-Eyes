"""ML1 tests — dataset-collection support writes frames + a labelling manifest.

Uses a local fake ``FrameSource`` (the port is structural) rather than importing
the acquisition feature, matching this module's boundary rules.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.infrastructure.data.dataset_collection import (
    CaptureManifest,
    DatasetCollector,
)

from .conftest import make_frame


class _FakeSource:
    """A minimal ``FrameSource``: yields a fixed list of frames."""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[Frame]:
        yield from self._frames


def test_collect_writes_manifest_and_frames(tmp_path: Path) -> None:
    frames = [make_frame(i) for i in range(3)]
    collector = DatasetCollector(tmp_path, source_name="demo")

    manifest = collector.collect(_FakeSource(frames))

    assert len(manifest) == 3
    assert manifest.source_name == "demo"
    assert [entry.frame_id for entry in manifest.entries] == [0, 1, 2]
    assert collector.manifest_path.exists()
    for entry in manifest.entries:
        assert (tmp_path / entry.image_path).exists()
        assert entry.label_path is None


def test_collected_frames_round_trip_pixels(tmp_path: Path) -> None:
    frame = make_frame(0)
    frame.image[1, 2] = (10, 20, 30)  # a distinctive pixel to verify losslessness
    collector = DatasetCollector(tmp_path)

    manifest = collector.collect(_FakeSource([frame]))

    restored = np.load(tmp_path / manifest.entries[0].image_path)
    assert np.array_equal(restored, frame.image)


def test_limit_caps_number_of_frames(tmp_path: Path) -> None:
    frames = [make_frame(i) for i in range(10)]
    collector = DatasetCollector(tmp_path)

    manifest = collector.collect(_FakeSource(frames), limit=4)

    assert len(manifest) == 4
    assert [entry.frame_id for entry in manifest.entries] == [0, 1, 2, 3]


def test_manifest_save_load_round_trip(tmp_path: Path) -> None:
    collector = DatasetCollector(tmp_path)
    original = collector.collect(_FakeSource([make_frame(0), make_frame(1)]))

    reloaded = CaptureManifest.load(collector.manifest_path)

    assert reloaded == original
