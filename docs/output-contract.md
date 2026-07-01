# Output Contract — Consuming the World Model (EPIC H)

This is the boundary between the **perception layer** (this project) and a
downstream **decision layer**. Perception publishes a versioned
[`WorldModel`](../src/zero_ad_eyes/domain/world_model.py); the decision layer
consumes it. This document is the consumer's reference: the schema, its versioning
policy, the serialization format, the available sinks, publish cadence, and a
minimal example client.

Everything here is **transport-agnostic**. Per open question **OQ-3**
(REQUIREMENTS.md §4.7), the concrete IPC — in-process object, socket, shared
memory — is deferred to the M5 integration milestone. The schema, codec, and sinks
are designed so that choice can be made *without reworking any of them*: a new
transport is a new `WorldModelSink` (for producing) or a new byte source feeding
`WorldModelCodec.decode` (for consuming), not a schema change.

## 1. The schema

A `WorldModel` is a frozen [pydantic](https://docs.pydantic.dev) value object with
no pixels and no framework state (REQUIREMENTS.md §4):

| Field            | Type                       | Meaning                                             |
| ---------------- | -------------------------- | --------------------------------------------------- |
| `schema_version` | `str` (semver)             | Contract version this model was produced against.   |
| `meta`           | `FrameMeta`                | Frame provenance: `frame_id`, `timestamp`, `source`, `width`, `height` (§4.1). |
| `hud`            | `HudState \| None`         | Own-POV resources, population, phase, self-id (§4.2). |
| `minimap`        | `MinimapModel \| None`     | Blips, viewport rectangle, confidence (§4.4).       |
| `entities`       | `tuple[Entity, ...]`       | Per-entity kind, ownership, health, world position, `staleness`, confidence (§4.3). |

Every perceived fact carries a `Confidence` (a bounded `[0, 1]` value plus a
`Provenance` tag: `classical` / `learned` / `engine_gt` / `fused` / `unknown`), so a
consumer weights a coarse classical guess differently from a confident learned
detection — identically whether it came from the model stub or the real adapter
(MP2). `staleness` counts frames since an entity was last actually observed (memory
of off-screen / fogged entities, §4.1 / G3).

**Own point-of-view only (A6).** Enemy resources, population, and fog-hidden state
are physically unobservable and are never present in the model. Absent knowledge is
represented as `None` / empty, never guessed.

## 2. Versioning & compatibility (H1)

`schema_version` is [semantic versioning](https://semver.org). The compatibility
policy (`versioning.py`) is **additive-minor**:

- **Major differs → incompatible.** A major bump is a breaking change (a field
  removed, renamed, or retyped). A consumer built for major *N* must refuse major
  *M*.
- **Major matches → compatible.** Minor/patch bumps only *add* optional fields; a
  newer producer's extra fields are ignored by an older consumer, and an older
  producer's missing fields are treated as absent.

Consumers should call `check_compatibility(world_model, supported)` (or let
`WorldModelCodec.decode` do it) before trusting a decoded model; an incompatible
version raises `IncompatibleSchemaError` rather than being silently misread.

```python
from zero_ad_eyes.infrastructure.contract import (
    CURRENT_SCHEMA_VERSION,
    SchemaVersion,
    check_compatibility,
)

check_compatibility(world_model, SchemaVersion.parse("0.1.0"))
```

## 3. Serialization (H2)

`WorldModelCodec` round-trips a `WorldModel` to/from a **single-line JSON document**
(JSON Lines-safe). It is the only place the wire format is decided; swapping to
protobuf/msgpack later is a new codec, not a caller change. `decode` enforces the
§2 compatibility policy.

```python
from zero_ad_eyes.infrastructure.contract import WorldModelCodec

codec = WorldModelCodec()
line = codec.encode(world_model)     # str, no embedded newline
restored = codec.decode(line)        # WorldModel, version-checked
```

## 4. Sinks — where the model goes (H2)

A **sink** implements the `WorldModelSink` port: `publish(world_model) -> None`. The
pipeline hands each world model to a sink and does not care where it goes. Three
transport-agnostic destinations ship today:

| Sink                        | Use                                                                    |
| --------------------------- | ---------------------------------------------------------------------- |
| `InMemoryWorldModelSink`    | In-process hand-off / test double. Exposes `published` and `latest`.   |
| `JsonlFileWorldModelSink`   | Durable append-only log, one JSON document per line. Context manager.  |
| `CallbackWorldModelSink`    | Adapts the port to any `callable` — the seam for a future socket/queue. |

```python
from zero_ad_eyes.infrastructure.contract import JsonlFileWorldModelSink

with JsonlFileWorldModelSink("world.jsonl") as sink:
    for world_model in pipeline.run():   # pipeline also publishes to its own sink
        sink.publish(world_model)
```

## 5. Publish cadence (H3)

Cadence is *when* a model is handed over, orthogonal to *where*. Both cadences are
decorator sinks that wrap any real sink:

- `PerFrameSink(inner)` — forward every frame. Highest freshness and bandwidth; the
  default.
- `OnChangeSink(inner)` — forward only when the **decision-relevant payload**
  (`schema_version`, `hud`, `minimap`, `entities`, including `staleness`) differs
  from the last forwarded model. Frame `meta` (id/timestamp) is ignored, since it
  changes every frame. Collapses steady scenes to zero downstream traffic; exposes a
  `suppressed` count for observability (NF6).

```python
from zero_ad_eyes.infrastructure.contract import OnChangeSink, JsonlFileWorldModelSink

with JsonlFileWorldModelSink("world.jsonl") as file_sink:
    sink = OnChangeSink(file_sink)   # only writes a line when something changed
```

**Latency budget (H3 / NF1).** `publish` runs inside the per-frame loop, so a sink
must stay well under the ~66 ms frame budget (15 FPS). On-change trades a cheap
value comparison for skipped downstream work when nothing changed.

## 6. Example client

`WorldModelReader` is the reference consumer — the symmetric counterpart to
`JsonlFileWorldModelSink`. It reads a JSONL log lazily, decodes each line, and lets
the codec enforce the compatibility check:

```python
from zero_ad_eyes.infrastructure.contract import WorldModelReader

reader = WorldModelReader("world.jsonl")          # supported=CURRENT by default
for world_model in reader.read():                 # lazy, one line at a time
    for entity in world_model.entities:
        ...                                        # decision logic reasons in world space

latest = reader.latest()                          # or just the most recent model
```

When M5 selects a live transport (OQ-3), the same `WorldModelCodec.decode` call
consumes a line off a socket/queue instead of a file — only the byte source changes,
not the parsing or the version check.
