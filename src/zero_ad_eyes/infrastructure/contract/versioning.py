"""H1 — schema versioning & backward-compatibility policy for the world-model contract.

The domain owns the *value* (``WorldModel.schema_version``, a semver string). This
module owns the *reasoning* about that value, deliberately kept out of the domain so
the schema type stays a pure, framework-light data object (it is NOT modified here).

Compatibility policy (REQUIREMENTS.md §4.7 — semver, additive-minor):

- **Major** differs  → incompatible. A major bump signals a breaking change
  (removed/renamed/retyped fields); a consumer built for major *N* must not silently
  read major *M*.
- **Major** matches  → compatible. Minor/patch bumps are additive and
  backward-compatible: a newer producer only *adds* optional fields (a consumer
  ignores unknown ones) and an older producer omits fields the consumer treats as
  absent. Provenance/confidence on every field (§4) means "absent" is expressible.

Consumers call :func:`check_compatibility` before trusting a decoded model.
"""

from __future__ import annotations

from dataclasses import dataclass

from zero_ad_eyes.domain.world_model import SCHEMA_VERSION, WorldModel


class IncompatibleSchemaError(RuntimeError):
    """Raised when a producer's schema version is not readable by the consumer."""

    def __init__(self, producer: SchemaVersion, consumer: SchemaVersion) -> None:
        super().__init__(
            f"world-model schema {producer} is incompatible with consumer {consumer}: "
            f"major version differs ({producer.major} != {consumer.major})"
        )
        self.producer = producer
        self.consumer = consumer


@dataclass(frozen=True, order=True)
class SchemaVersion:
    """A parsed semantic version ``major.minor.patch`` (an ordered value object)."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> SchemaVersion:
        """Parse ``"1.2.3"``; raise ``ValueError`` on any malformed input."""

        parts = text.split(".")
        if len(parts) != 3:
            raise ValueError(f"not a MAJOR.MINOR.PATCH version: {text!r}")
        try:
            major, minor, patch = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"non-integer version component in {text!r}") from exc
        if major < 0 or minor < 0 or patch < 0:
            raise ValueError(f"negative version component in {text!r}")
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, consumer: SchemaVersion) -> bool:
        """True iff a consumer at ``consumer`` may read a producer at ``self``."""

        return self.major == consumer.major


CURRENT_SCHEMA_VERSION: SchemaVersion = SchemaVersion.parse(SCHEMA_VERSION)
"""The version this build produces — parsed once from the domain constant."""


def check_compatibility(
    world_model: WorldModel,
    consumer: SchemaVersion = CURRENT_SCHEMA_VERSION,
) -> None:
    """Raise :class:`IncompatibleSchemaError` if ``world_model`` is unreadable.

    ``consumer`` defaults to the version this build understands, so a caller in this
    process can validate a model it just decoded with a bare call.
    """

    producer = SchemaVersion.parse(world_model.schema_version)
    if not producer.is_compatible_with(consumer):
        raise IncompatibleSchemaError(producer, consumer)
