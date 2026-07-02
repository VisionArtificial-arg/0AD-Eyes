"""Recording sidecar format (A5) — the temporal provenance a video file loses.

A video file preserves pixels but not the capture's *clock*: on replay,
``VideoFileSource`` renumbers frames 0..N by position and derives timestamps from
the container's PTS. That is fine for eyeballing footage, but the real-frame
accuracy chain (#2) has to align predicted world models against an engine
ground-truth export keyed to the *capture's own* ``frame_id`` / ``timestamp`` — and
those are exactly what the video drops.

``RecordingManifest`` is the sidecar that carries them: one ``FrameStamp`` per
recorded frame, written next to the video (same stem, ``.json``). It is deliberately
minimal — it records only what the video cannot reconstruct on its own (the true
``frame_id`` and ``timestamp``); width/height/source are recoverable from the video
and the capture config. A future replay source reads this back to restamp frames
with their real clock before alignment.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_VERSION = "0.1.0"


class FrameStamp(BaseModel):
    """One recorded frame's true clock — the pair the video file cannot preserve."""

    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp: float


class RecordingManifest(BaseModel):
    """The per-frame temporal index accompanying a recorded video file."""

    model_config = ConfigDict(frozen=True)

    manifest_version: str = MANIFEST_VERSION
    video: str  # the video's filename, relative to this manifest's directory
    source: str  # the capture source name (e.g. "live"), shared by every frame
    fps: float = Field(gt=0.0)  # the capture's target pacing
    stamps: tuple[FrameStamp, ...] = ()

    def __len__(self) -> int:
        return len(self.stamps)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RecordingManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
