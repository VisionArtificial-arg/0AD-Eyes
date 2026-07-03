"""Output contract adapters — the decision-layer boundary (REQUIREMENTS.md EPIC H).

This package turns the domain ``WorldModel`` (§4) into something a *separate*
decision layer can consume, without the schema ever learning how it is delivered.
Per OQ-3 (§4.7) the concrete IPC — in-process object, socket, shared memory — is
deliberately *deferred*: everything here is transport-agnostic. The pieces are:

- ``versioning`` (H1): a semver value object and backward-compatibility policy
  built *around* the existing ``WorldModel.schema_version`` (the domain type is
  never modified).
- ``serialization`` (H2): a codec that round-trips a ``WorldModel`` to/from JSON.
- ``sinks`` (H2): ``WorldModelSink`` implementations — in-memory, JSONL file,
  callback — each an autonomous collaborator satisfying the application port.
- ``cadence`` (H3): per-frame vs on-change publish wrappers that decorate any sink.
- ``example_client`` (H4): a minimal reference consumer that reads the contract.
"""

from __future__ import annotations

from .cadence import OnChangeSink, PerFrameSink
from .example_client import WorldModelReader
from .serialization import WorldModelCodec
from .sinks import (
    CallbackWorldModelSink,
    CompositeWorldModelSink,
    InMemoryWorldModelSink,
    JsonlFileWorldModelSink,
)
from .versioning import (
    CURRENT_SCHEMA_VERSION,
    IncompatibleSchemaError,
    SchemaVersion,
    check_compatibility,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CallbackWorldModelSink",
    "CompositeWorldModelSink",
    "IncompatibleSchemaError",
    "InMemoryWorldModelSink",
    "JsonlFileWorldModelSink",
    "OnChangeSink",
    "PerFrameSink",
    "SchemaVersion",
    "WorldModelCodec",
    "WorldModelReader",
    "check_compatibility",
]
