"""Record / replay a perception run to disk, as a LIBRARY (REQUIREMENTS.md X2).

A *run* is a monotonic sequence of ``(Frame, WorldModel)`` pairs — exactly what the
debug overlay (X1) consumes — persisted so an offline session can be replayed
deterministically (NF5) and re-inspected without the live game (A5, EPIC A5, T2).

This module is deliberately importable-only: no CLI parsing, no ``argparse``. CLI
wiring happens later at the integration milestone (M5); keeping it a library lets
tests and other tools drive it directly.

On-disk layout (one directory per run)::

    run_dir/
        manifest.json          # {schema_version, count, source}
        000000.image.npy       # frame pixels (numpy, lossless, headless-safe)
        000000.world.json      # WorldModel as JSON (the contract, §4.7)
        000001.image.npy
        000001.world.json
        ...

Images are stored with numpy (not an image codec) so the round-trip is exact and
never opens a window — the same headless guarantee the overlay honours.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.world_model import WorldModel

MANIFEST_NAME = "manifest.json"
RUN_SCHEMA_VERSION = "1.0.0"
_INDEX_WIDTH = 6


def _image_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"{index:0{_INDEX_WIDTH}d}.image.npy"


def _world_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"{index:0{_INDEX_WIDTH}d}.world.json"


class RunRecorder:
    """Appends ``(Frame, WorldModel)`` pairs to a run directory (X2, writer side).

    Usable as a context manager so the manifest is always flushed::

        with RunRecorder(path) as rec:
            rec.record(frame, world_model)
    """

    def __init__(self, run_dir: str | Path, *, source: str | None = None) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._source = source
        self._count = 0
        self._write_manifest()

    @property
    def count(self) -> int:
        return self._count

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def record(self, frame: Frame, world_model: WorldModel) -> int:
        """Persist one pair; returns its zero-based index in the run."""

        index = self._count
        image = np.asarray(frame.image)
        np.save(_image_path(self._run_dir, index), image, allow_pickle=False)
        _world_path(self._run_dir, index).write_text(
            world_model.model_dump_json(),
            encoding="utf-8",
        )
        self._count += 1
        self._write_manifest()
        return index

    def _write_manifest(self) -> None:
        manifest: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "count": self._count,
            "source": self._source,
        }
        (self._run_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._write_manifest()


class RunReplayer:
    """Reads a recorded run back as ``(Frame, WorldModel)`` pairs (X2, reader side)."""

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        manifest_path = self._run_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no run manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._count = int(manifest.get("count", 0))
        self._source = manifest.get("source")

    def __len__(self) -> int:
        return self._count

    @property
    def source(self) -> str | None:
        return self._source

    def __iter__(self) -> Iterator[tuple[Frame, WorldModel]]:
        for index in range(self._count):
            yield self._read(index)

    def world_models(self) -> Iterator[WorldModel]:
        """Iterate world models only (e.g. to re-score a run against ground truth)."""

        for _, world_model in self:
            yield world_model

    def _read(self, index: int) -> tuple[Frame, WorldModel]:
        image = np.load(_image_path(self._run_dir, index), allow_pickle=False)
        world_model = WorldModel.model_validate_json(
            _world_path(self._run_dir, index).read_text(encoding="utf-8")
        )
        frame = Frame(image=image, meta=world_model.meta)
        return frame, world_model


def record_run(
    run_dir: str | Path,
    pairs: Iterable[tuple[Frame, WorldModel]],
    *,
    source: str | None = None,
) -> int:
    """Convenience: write an entire run at once; returns the number of pairs."""

    with RunRecorder(run_dir, source=source) as recorder:
        for frame, world_model in pairs:
            recorder.record(frame, world_model)
        return recorder.count


def replay_run(run_dir: str | Path) -> Iterator[tuple[Frame, WorldModel]]:
    """Convenience: iterate a recorded run's ``(Frame, WorldModel)`` pairs."""

    return iter(RunReplayer(run_dir))
