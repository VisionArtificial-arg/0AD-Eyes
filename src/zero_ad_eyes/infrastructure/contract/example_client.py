"""H4 — a minimal reference consumer of the world-model contract.

This is the *decision layer's* side of the boundary, shipped here as an executable
example (and doctest-able stub) so integrators see exactly how to read the contract:
open a JSONL log, decode each line, and let the codec enforce the H1 compatibility
policy before the model is trusted. It is the symmetric counterpart to
:class:`JsonlFileWorldModelSink` — what one writes, this reads back.

It is intentionally transport-agnostic (OQ-3): it consumes *text lines* from a file,
which is the deferred-decision-safe interchange. When M5 picks a live IPC, the same
``WorldModelCodec.decode`` call consumes a line off a socket/queue instead — only
the byte source changes, not the parsing or the version check.

The prose walkthrough lives in ``docs/output-contract.md``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from zero_ad_eyes.domain.world_model import WorldModel

from .serialization import WorldModelCodec
from .versioning import CURRENT_SCHEMA_VERSION, SchemaVersion


class WorldModelReader:
    """Reads world models back from a JSONL contract log, newest-compatible-first.

    Each non-blank line is one JSON world model. Decoding runs the H1 compatibility
    check against ``supported``; an incompatible line raises rather than being
    silently misread (fail-loud is the safe default at a version boundary).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        supported: SchemaVersion = CURRENT_SCHEMA_VERSION,
    ) -> None:
        self._path = Path(path)
        self._codec = WorldModelCodec(consumer=supported)

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> Iterator[WorldModel]:
        """Yield each stored world model in file order (lazily, one line at a time)."""

        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield self._codec.decode(line)

    def latest(self) -> WorldModel | None:
        """The last world model in the log, or ``None`` if the log is empty."""

        last: WorldModel | None = None
        for world_model in self.read():
            last = world_model
        return last
