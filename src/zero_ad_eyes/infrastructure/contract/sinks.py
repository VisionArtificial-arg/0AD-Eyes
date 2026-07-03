"""H2 — ``WorldModelSink`` implementations (the decision-layer boundary).

Each sink is an autonomous collaborator satisfying the ``WorldModelSink`` port
(``publish(world_model) -> None``): the pipeline hands it a world model and does not
care where it goes. Three transport-agnostic destinations (OQ-3, §4.7) cover the
integration surface without committing to an IPC mechanism:

- :class:`InMemoryWorldModelSink` — in-process hand-off / test double. The decision
  layer sharing this process reads the latest (or full history) directly.
- :class:`JsonlFileWorldModelSink` — durable append-only log; one JSON document per
  line. The offline/decoupled consumer (and dataset replay) reads it back.
- :class:`CallbackWorldModelSink` — adapts the port to any ``callable``, so wiring a
  socket/queue later (M5) is a one-liner, not a new class here.
- :class:`CompositeWorldModelSink` — fan-out adapter for publishing to multiple
  sinks without teaching the pipeline about any destination.

None imports a socket or shared-memory primitive: the concrete IPC choice stays
deferred (OQ-3). The port is structural, so these classes need no explicit base.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import IO

from zero_ad_eyes.application.ports import WorldModelSink
from zero_ad_eyes.domain.world_model import WorldModel

from .serialization import WorldModelCodec


class InMemoryWorldModelSink:
    """Retains published models in order; the simplest in-process hand-off."""

    def __init__(self) -> None:
        self._published: list[WorldModel] = []

    def publish(self, world_model: WorldModel) -> None:
        self._published.append(world_model)

    @property
    def published(self) -> tuple[WorldModel, ...]:
        """An immutable snapshot of everything published so far."""

        return tuple(self._published)

    @property
    def latest(self) -> WorldModel | None:
        """The most recently published model, or ``None`` if none yet."""

        return self._published[-1] if self._published else None


class JsonlFileWorldModelSink:
    """Appends each published model as one JSON line to a file (JSON Lines).

    Owns its file handle for the lifetime of the sink (opened on construction,
    ``flush``-ed per publish so a crash keeps every fully-written line). Use as a
    context manager, or call :meth:`close` when done.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        append: bool = False,
        codec: WorldModelCodec | None = None,
    ) -> None:
        self._path = Path(path)
        self._codec = codec or WorldModelCodec()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = self._path.open("a" if append else "w", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def publish(self, world_model: WorldModel) -> None:
        self._file.write(self._codec.encode(world_model))
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlFileWorldModelSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class CallbackWorldModelSink:
    """Forwards each published model to a caller-supplied ``callable``.

    The adapter that keeps this package transport-agnostic: the M5 integration wires
    a socket/queue send here without a new sink class.
    """

    def __init__(self, on_publish: Callable[[WorldModel], None]) -> None:
        self._on_publish = on_publish

    def publish(self, world_model: WorldModel) -> None:
        self._on_publish(world_model)


class CompositeWorldModelSink:
    """Publishes each world model to several sinks in order."""

    def __init__(self, *sinks: WorldModelSink) -> None:
        self._sinks = tuple(sinks)

    def publish(self, world_model: WorldModel) -> None:
        for sink in self._sinks:
            sink.publish(world_model)
