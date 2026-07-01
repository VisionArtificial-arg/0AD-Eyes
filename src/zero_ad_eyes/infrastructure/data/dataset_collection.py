"""ML1 — Dataset-collection support (REQUIREMENTS.md §7 ML1, EPIC A5).

A small offline pipeline that, given any ``FrameSource`` (live capture or an
offline recording — they are interchangeable behind the port, A3), persists each
frame to disk together with an **index manifest** describing what was captured.
The manifest is the hand-off artifact for labelling (ML3) and ground-truth
alignment (ML2): each entry names an image file and reserves a slot for its label.

Design notes:
- We operate on the ``FrameSource`` *port* abstractly (anything exposing
  ``frames() -> Iterator[Frame]``); we never import a concrete capture adapter.
- Frames are written as NumPy ``.npy`` arrays, not encoded images, so this module
  needs no image codec dependency and round-trips pixels losslessly and
  deterministically (NF5). The manifest is plain JSON.
- This is dataset-building machinery; it is offline-only and has no place in the
  inference path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import FrameSource

MANIFEST_VERSION = "0.1.0"
_FRAMES_SUBDIR = "frames"


class CaptureEntry(BaseModel):
    """One captured frame's provenance plus where its pixels/label live on disk."""

    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp: float
    source: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_path: str  # relative to the manifest file's directory
    label_path: str | None = None  # filled in by the labelling step (ML3), if any


class CaptureManifest(BaseModel):
    """The index of a collected dataset — the labelling hand-off artifact."""

    model_config = ConfigDict(frozen=True)

    manifest_version: str = MANIFEST_VERSION
    source_name: str
    entries: tuple[CaptureEntry, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CaptureManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class DatasetCollector:
    """Drains a ``FrameSource`` into a persisted, labellable dataset directory.

    An autonomous collaborator: hand it an output directory, then ``collect`` from
    any frame source. It owns the on-disk layout (``frames/`` for ``.npy`` images,
    ``manifest.json`` for the index) and returns the manifest it wrote.
    """

    def __init__(self, output_dir: Path, source_name: str = "capture") -> None:
        self._output_dir = output_dir
        self._source_name = source_name

    @property
    def manifest_path(self) -> Path:
        return self._output_dir / "manifest.json"

    def collect(self, source: FrameSource, limit: int | None = None) -> CaptureManifest:
        """Write frames from ``source`` and an index manifest; return the manifest.

        ``limit`` caps how many frames are drained (``None`` = exhaust the source).
        Streams frame-by-frame, so an unbounded live source is safe with a ``limit``.
        """

        frames_dir = self._output_dir / _FRAMES_SUBDIR
        frames_dir.mkdir(parents=True, exist_ok=True)

        entries: list[CaptureEntry] = []
        for index, frame in enumerate(source.frames()):
            if limit is not None and index >= limit:
                break
            entries.append(self._persist(frame, frames_dir))

        manifest = CaptureManifest(
            source_name=self._source_name,
            entries=tuple(entries),
        )
        manifest.save(self.manifest_path)
        return manifest

    def _persist(self, frame: Frame, frames_dir: Path) -> CaptureEntry:
        rel_image_path = f"{_FRAMES_SUBDIR}/frame_{frame.meta.frame_id:06d}.npy"
        np.save(self._output_dir / rel_image_path, np.asarray(frame.image))
        return CaptureEntry(
            frame_id=frame.meta.frame_id,
            timestamp=frame.meta.timestamp,
            source=frame.meta.source,
            width=frame.meta.width,
            height=frame.meta.height,
            image_path=rel_image_path,
        )
