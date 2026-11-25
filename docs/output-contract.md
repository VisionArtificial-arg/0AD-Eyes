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
| `events`         | `tuple[Event, ...]`        | Transitions since the previous frame (v0.2; payload v0.3, below). |

### Events (v0.3)

An event is something that *changed*: a unit appeared, health dropped, a resource ran
out. The policy is **events over silence**: the eyes emit every transition they can
observe, each with a confidence, and the consumer decides what to ignore.

| Field        | Type                          | Meaning |
| ------------ | ----------------------------- | ------- |
| `kind`       | enum                          | What happened, e.g. `unit_appeared`, `combat`, `resource_depleted`, `state_changed`. |
| `frame_id`   | int                           | Frame the transition was observed on. |
| `entity_id`  | int \| null                   | Tracked entity it concerns, if any. |
| `confidence` | `Confidence`                  | How sure the eyes are, plus provenance. |
| `origin`     | `hud` \| `minimap` \| `entity` \| `system` \| `unknown` | Surface it was read from; `unknown` only from v0.2 producers. |
| `details`    | map of str → scalar           | Typed payload, e.g. `health_before`/`health_after`, `type_before`/`type_after`, plus entity context (`entity_kind`, `ownership`, `entity_type`, `health`) for entity events. |
| `screen_pos` | `{x, y}` \| null              | Full-capture pixels (an entity's box centre). |
| `world_pos`  | `{x, y}` \| null              | World position; for a disappearance, where it was last seen. |

Sources run in a fixed order, so a recording always yields the same event list: system,
then HUD, then minimap, then entity. Each can be switched off in the `events` config section.

#### System events (`origin: system`)

The eyes reporting on their own health, so a consumer knows when to trust the rest of
the stream less. Failures are reported when the state changes, not on every frame.

| `kind` | `details` |
| --- | --- |
| `layout_drift` | none (the periodic self-check found the HUD layout moved) |
| `recalibrated` | `width`, `height` (layout (re-)detected) |
| `hud_read_failed` / `hud_read_restored` | `hud_present` on failure (no HUD values read) |
| `hud_low_confidence` | `confidence` (fell below `events.system_low_confidence`) |
| `minimap_read_failed` / `minimap_read_restored` | none |
| `frames_dropped` | `missed`, `step`, `usual_step` (frame ids jumped by more than the usual step) |
| `detections_stopped` / `detections_resumed` | `silent_frames` / `observed` (no observed entity for `events.system_silent_frames` frames) |

#### HUD events (`origin: hud`)

Read off the top bar and the selection panel. Confidence is the HUD read's. A value the
HUD did not read this frame is not a change: the last known value is kept until it is
read again.

| `kind` | `details` |
| --- | --- |
| `resource_gained` / `resource_spent` | `resource`, `before`, `after`, `amount` (≥ `events.hud_min_resource_delta`) |
| `population_changed` | `before`, `after`, `delta`, `cap` |
| `population_cap_reached` | `current`, `cap` (population just reached the cap) |
| `population_cap_raised` / `population_cap_lowered` | `before`, `after` |
| `phase_changed` | `before`, `after` (`village` / `town` / `city`) |
| `selection_changed` | `before`, `after`: selected entity type, `null` = nothing selected |
| `selection_health_changed` | `entity_type`, `before`, `after` (≥ `events.hud_min_selection_health_delta`) |
| `production_queue_changed` | `entity_type`, `before`, `after` (comma-joined), `length_before`, `length_after` |
| `garrison_changed` | `entity_type`, `before`, `after` |

Idle-worker counts and alert notifications are not read by any HUD reader yet, so they
have no events.

#### Minimap events (`origin: minimap`)

Positions are world coordinates from the minimap calibration. Blips have no identity,
so a blip is "the same" as the nearest same-owner blip of the previous frame within
`events.minimap_blip_match_radius`. Territory and camera compare with the last
*reported* value, so slow changes still add up to an event.

| `kind` | `details` / position |
| --- | --- |
| `blip_appeared` / `blip_vanished` | `ownership`; `world_pos` = the blip |
| `enemy_in_territory` | `ownership`; `world_pos` = the enemy blip, once per entry. A region is approximated as a disc of its coverage |
| `territory_gained` / `territory_lost` | `ownership`, `before`, `after` (map fraction; change ≥ `events.minimap_territory_min_change`) |
| `area_explored` / `area_visible` / `area_fogged` | `cells`, mean `row`/`col` of the changed cells, `explored_fraction`. One event per kind per frame |
| `camera_moved` | `dx`, `dy`, `distance`; `world_pos` = the new view centre (≥ `events.minimap_camera_move_min`) |

#### Entity events (`origin: entity`)

| `kind` | `details` (beyond `entity_kind`, `ownership`, `entity_type`, `health`) |
| --- | --- |
| `unit_appeared` / `unit_disappeared` | none (a disappearance's position is where it was last seen) |
| `combat` | `health_before`, `health_after` (drop > `tracking.combat_drop`) |
| `resource_depleted` | none (health ≤ `tracking.depletion_health`) |
| `state_changed` | `type_before`, `type_after` (e.g. scaffold → building) |
| `entity_left_view` / `entity_hidden` | `viewport_known`. Stopped being seen: outside the camera footprint, or lost inside it (fog, occlusion) |
| `entity_entered_view` / `entity_revealed` | `frames_unseen` (seen again after leaving the view / after being hidden) |
| `health_recovered` | `health_before`, `health_after` (rise ≥ `events.entity_heal_min`: repair, healing, construction) |
| `started_moving` / `stopped_moving` | `speed` (px/frame), `dx`/`dy` on start; hysteresis between `events.entity_stop_max_speed` and `events.entity_move_min_speed` |
| `entity_selected` / `entity_deselected` | none |
| `building_seen` | none (any building appearing, including on the first frame) |
| `possible_unit_trained` | `building_entity_id`, `building_type`, `distance_px`. A new own unit within `events.entity_spawn_radius` px of an own building; a guess, not emitted on the first frame |

Construction progress and garrison counts are not on the entity contract, so they have
no events; a building going up shows as `health_recovered`.

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
