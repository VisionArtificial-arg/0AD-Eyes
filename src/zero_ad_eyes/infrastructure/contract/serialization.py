"""H2 — serialization codec for the world-model contract.

A ``WorldModel`` is a pydantic value object, so JSON is its natural wire form: the
codec is a thin, honest boundary over ``model_dump_json`` / ``model_validate_json``.
Kept as a first-class object (not loose functions) so a future format — protobuf,
msgpack — is a new codec implementing the same two messages, not a rewrite of every
caller. The codec is transport-agnostic (OQ-3, §4.7): it produces/consumes *text*,
never touching a socket, file, or pipe. Encoded output is one self-contained JSON
document per world model, which the JSONL sink writes one-per-line.

On :meth:`decode` the schema version is checked against the consumer's supported
version (H1 policy) so a caller cannot silently trust an incompatible model.
"""

from __future__ import annotations

from zero_ad_eyes.domain.world_model import WorldModel

from .versioning import CURRENT_SCHEMA_VERSION, SchemaVersion, check_compatibility


class WorldModelCodec:
    """Round-trips a ``WorldModel`` to/from a single-line JSON document."""

    def __init__(self, consumer: SchemaVersion = CURRENT_SCHEMA_VERSION) -> None:
        self._consumer = consumer

    def encode(self, world_model: WorldModel) -> str:
        """Serialize to a compact, single-line JSON string (JSONL-safe)."""

        return world_model.model_dump_json()

    def decode(self, text: str) -> WorldModel:
        """Parse JSON back into a ``WorldModel``, enforcing the H1 compat policy."""

        world_model = WorldModel.model_validate_json(text)
        check_compatibility(world_model, self._consumer)
        return world_model
