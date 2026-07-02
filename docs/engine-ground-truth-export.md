# Engine ground-truth export — contract (ML2 / D6)

This document pins the JSON contract that an **external 0 A.D.-side exporter** must
emit for the accuracy loop to score classical perception against ground truth. It is
the target spec for that tool; the harness in this repo only *consumes* the format.

The authoritative schema is the Pydantic model in
`src/zero_ad_eyes/infrastructure/data/ground_truth.py`
(`EngineStateExport` / `EngineFrameState` / `EngineEntityState`). If this document
and that module ever disagree, **the module wins** — regenerate this doc.

## Purpose & label strategy (decided 2026-07-02)

The exporter is **dual-purpose**: it provides ground truth to *score* perception, and
— the larger prize — it **auto-labels a train/test/validate dataset** from engine data
+ captures, avoiding manual annotation. The label form and where it is computed follow
from that; the JSON schema below carries the per-frame simulation state regardless.

**Who computes each entity's on-screen region?**

- **Engine-side (preferred).** 0 A.D. emits the label directly — pixel-perfect, and it
  already has the renderer. The **preferred form is instance segmentation via an
  entity-ID / picking render pass** (dump the per-pixel entity-id buffer alongside the
  frame): masks are a strict superset of boxes (a box derives from a mask trivially),
  give exact silhouettes, and let the renderer handle occlusion + fog *for free*,
  auto-aligned to the RGB. Use engine-side for **both** training and scoring when
  available — it strictly dominates.
- **Harness-side (fallback only).** If the engine cannot emit labels cheaply, it emits
  a per-frame **camera pose** (view-projection matrix + viewport) + entity world
  positions/extents, and this repo projects them (reusing the F1 `ViewportCameraProjector`
  camera model). Trivial exporter, but approximate boxes and we must filter
  occlusion/fog ourselves — acceptable for bulk *training* labels, NOT tight enough for
  *validation*.

> **OPEN — verify first (first exporter task):** does 0 A.D. expose an entity-ID /
> picking render pass dumpable per frame? Very likely (it underpins mouse-selection),
> but unverified. **Yes →** build the segmentation export, drop the harness-side path.
> **No →** fall back to camera-pose + harness-side projection. This one fact decides the
> exporter's whole shape. See the `exporter-dataset-strategy` project note.

**Regardless of approach:** label only what the pixels actually show (on-screen,
unfogged, unoccluded) — the engine is omniscient, but truth for a pixel-based system
must not include fogged/hidden entities (D1). If using engine-*rendered* RGB rather than
the real screen capture, mind the domain gap (HUD, post-processing) — inference reads
the real displayed screen.

The schema below is the current, box-oriented contract (harness-side / measurement
path). It will be extended with the segmentation label form once the render-pass
question is answered.

## Boundary — offline only (D1/D6)

Decision **D6**: engine-derived state MAY be used as ground truth for building
datasets and measuring accuracy, and is **NEVER** available at inference (production
reads pixels only, D1). This export therefore lives entirely on the offline side:
nothing that produces or consumes it may be linked into the live perception pipeline.
An exporter is any 0 A.D.-side tool that can emit authoritative simulation state —
a replay parser, the AI-interface JSON, or a debug-build hook.

## Alignment contract — the shared clock (R5)

Scoring pairs each *predicted* frame (from replaying a recording through the
classical chain) to its *engine* frame. Two modes, both requiring a **shared clock**
between capture and engine:

- **`frame_id` (exact).** The engine's `frame_id` must equal the capture's real
  frame id. A recorded video on its own renumbers frames `0..N`, so replay uses the
  `--record` **sidecar** (`RecordingManifest`, the `.json` beside the `.mkv`) to
  restore the capture's real `frame_id`/`timestamp`. The exporter's `frame_id` must
  be on that same clock. Prefer this mode when capture and engine can share a frame
  counter.
- **`timestamp` (nearest within tolerance).** Each captured frame pairs to the engine
  frame nearest in time, within `--time-tolerance` seconds (inclusive). Use when only
  a common wall-clock — not a common frame counter — is available. `timestamp` is in
  **seconds**, same clock as the capture.

Consumed via `eval --recording <video> --engine-export <json> [--align-by frame_id|timestamp] [--time-tolerance S]`.

## Schema

One JSON document per match.

### Top level — `EngineStateExport`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `match_id` | string | yes | Identifier for the match/replay. |
| `self_player` | int ≥ 0 | yes | Engine player number we are playing. |
| `ally_players` | int[] | no (default `[]`) | Player numbers treated as allies. |
| `frames` | frame[] | no (default `[]`) | Per-frame authoritative state (below). |

### Per frame — `EngineFrameState`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `frame_id` | int ≥ 0 | yes | Must share the capture's clock (see Alignment). |
| `timestamp` | float (seconds) | yes | Same clock as capture; used by `timestamp` alignment. |
| `phase` | enum | no (default `"unknown"`) | One of `village`, `town`, `city`, `unknown`. |
| `resources` | map | no (default `{}`) | Keys `food`/`wood`/`stone`/`metal` → int stockpile. |
| `population_current` | int ≥ 0 \| null | no | Current population; omit or `null` if unknown. |
| `population_cap` | int ≥ 0 \| null | no | Population cap; omit or `null` if unknown. |
| `entities` | entity[] | no (default `[]`) | Simulation entities this frame (below). |

### Per entity — `EngineEntityState`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `entity_id` | int ≥ 0 | yes | Engine entity id (stable across frames; enables tracking/MOTA). |
| `kind` | enum | yes | One of `unit`, `building`, `resource_node`, `projectile`, `other`. |
| `owner` | int ≥ 0 | yes | Engine player number; `0` is Gaia (neutral). |
| `entity_type` | string \| null | no | Exact type, e.g. `"female_citizen"` (v1 taxonomy is coarse). |
| `health_current` | float ≥ 0 | no (default `0.0`) | With `health_max` yields a health fraction. |
| `health_max` | float ≥ 0 | no (default `0.0`) | If `≤ 0`, health fraction is unknown (`null`). |
| `world_x` | float \| null | no | Engine world-space X; units are engine-defined. |
| `world_y` | float \| null | no | Engine world-space Y. Both must be present for a world position. |
| `bbox` | box \| null | no | Entity's screen-space box, **if the exporter projected it**. |

`bbox` object: `{ "x": float, "y": float, "width": float ≥ 0, "height": float ≥ 0 }`
in **screen pixels** (axis-aligned).

## Derived semantics

- **Ownership** is resolved from `owner`: `0` → `gaia`; `== self_player` → `self`;
  in `ally_players` → `ally`; otherwise → `enemy`.
- **Health fraction** = `min(1, health_current / health_max)` when `health_max > 0`,
  else unknown.
- **World position** is set only when both `world_x` and `world_y` are present.
- **`bbox` is optional and load-bearing.** An entity *without* `bbox` still becomes a
  world-model label (ownership, health, world position, HUD) — so it counts toward
  ownership accuracy and MOTA — but is **excluded from detection labels** (no screen
  box to score detection mAP against). Emit `bbox` for every entity the exporter can
  project to the captured view; omit it for off-screen or unprojected entities.

## Minimal complete example

```json
{
  "match_id": "demo-001",
  "self_player": 1,
  "ally_players": [2],
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "phase": "village",
      "resources": { "food": 300, "wood": 200, "stone": 100, "metal": 50 },
      "population_current": 8,
      "population_cap": 20,
      "entities": [
        {
          "entity_id": 42,
          "kind": "unit",
          "entity_type": "female_citizen",
          "owner": 1,
          "health_current": 40.0,
          "health_max": 40.0,
          "world_x": 12.5,
          "world_y": 30.0,
          "bbox": { "x": 100, "y": 120, "width": 24, "height": 40 }
        }
      ]
    }
  ]
}
```

## What is still missing

This export **producer** is the largest remaining artifact for real-frame accuracy
(#2), F1 projection validation (#7), and now the auto-labeled training dataset (see
Purpose above). Until a 0 A.D.-side exporter emits documents matching this contract,
those metrics have nothing to score against, even with real recordings (which the
`--record` path now captures) in hand.

**First step before building it:** answer the render-pass question in *Purpose & label
strategy* — it decides whether the exporter emits segmentation masks (engine-side) or
camera pose for harness-side projection.
