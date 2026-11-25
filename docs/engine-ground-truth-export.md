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

> **RESOLVED (2026-11-25): engine-side.** The `0A.DDataset` engine fork already renders a
> segmentation pass (see `docs/engine-fork.md`). The exporter is that fork, writing the
> **Engine-fork dataset** layout at the end of this document. The camera-pose path survives
> only in the `cvdump` fallback.

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

## Engine-fork dataset (contract for ENG-1 / ENG-2 / ENG-3)

The `0A.DDataset` fork writes this layout. The loader in this repo is
`src/zero_ad_eyes/infrastructure/data/engine_dataset.py` (`EngineDatasetRun`,
`find_runs`), and its tests (`tests/test_data_engine_dataset.py`) build a synthetic run
exactly as the fork must write it. If this section and the module disagree, the module
wins.

### Layout: one directory per run (one match)

```
<dataset-root>/<run_id>/
    run.json                RunManifest (below)
    rgb/<frame>.png         the frame as the player sees it, GUI on (ENG-2)
    rgb_nogui/<frame>.png   optional: the same frame, GUI off (for the best.pt audit)
    mask/<frame>.png        instance mask (encoding below) (ENG-1)
    truth/<frame>.json      EngineFrameState for that frame (ENG-3)
```

- `<frame>` is the zero-padded capture index (`000000`, `000001`, …), and the same stem
  names the frame in every folder.
- A frame counts only when `rgb`, `mask` and `truth` all exist.
- `run_id` is the per-process id the exporter already generates; it identifies the match.
- **There are no train/val/test folders.** The split is made on ingest, by run (ENG-6).

### `run.json` — `RunManifest`

| Field | Type | Required | Notes |
|---|---|---|---|
| `run_id` | string | yes | Same as the directory name. |
| `map_name` | string | yes | e.g. `skirmishes/acropolis_bay_2p`. |
| `view_player` | int ≥ -1 | yes | Player whose view and HUD are rendered; `-1` = observer. |
| `ally_players` | int[] | no | Allies of `view_player`. |
| `width`, `height` | int > 0 | yes | Render resolution of every PNG. |
| `map_size` | float > 0 \| null | no | World units per map side (for minimap truth). |
| `mask_encoding` | string | yes | Must be `r-class_gb-instance/1`; the loader rejects anything else. |
| `gui_in_rgb` | bool | no (default `true`) | Whether `rgb/` frames include the GUI. |
| `engine_commit` | string \| null | no | Fork commit that produced the run. |

### Mask encoding `r-class_gb-instance/1`

8-bit RGB PNG, the same size as the RGB frame, **no anti-aliasing / MSAA** (every pixel
is exact).

- **R** = segmentation class id, the engine's `ESegmentationClass` value (0–16, table in
  `src/zero_ad_eyes/infrastructure/model/segmentation_classes.py`).
- **G·B** = per-frame **instance index**: `index = G * 256 + B`, 1–65535. Use 0 for
  anything that isn't an entity (terrain, water, sky, unlabeled).
- The index is a per-frame slot, not the engine entity id: entity ids can exceed 16 bits
  in long matches. Each entity's `mask_index` in the truth JSON links the two.

### `truth/<frame>.json` — `EngineFrameState`

The per-frame schema above (`frame_id`, `timestamp`, `phase`, `resources`,
`population_current`, `population_cap`, `entities`), plus:

- **Frame level.** Top-level `phase` / `resources` / population are the **viewing
  player's**, i.e. what the HUD shows. Use `timestamp` = simulation time in seconds.

  | Field | Type | Notes |
  |---|---|---|
  | `players` | player[] | Every player: `player_id`, `civ`, `color` `{r,g,b}` 0–255, `phase`, `resources`, `population_current`, `population_cap`. |
  | `camera` | object \| null | `x`, `y`, `z` (camera target, world), `rot_x` (pitch), `rot_y` (yaw) in radians, `zoom`. |

- **Entity level.** Emit every entity in or near the view, including occluded ones; the
  mask decides which get boxes. Leave `bbox` out: the loader computes it from the mask.

  | Field | Type | Notes |
  |---|---|---|
  | `mask_index` | int 1–65535 \| null | Its instance index in this frame's mask; `null` if not drawn. |
  | `template` | string \| null | Full template name, e.g. `units/athen/infantry_spearman_b`. |
  | `visibility` | `visible` \| `fogged` \| `hidden` \| null | For the viewing player. |
  | `world_z` | float \| null | Height (engine Y axis). |

### How the loader labels a frame

An entity gets a screen `bbox` (the tight box of its mask pixels) only when it covers at
least `min_pixels` pixels and is not `hidden`. Entities without a box stay in the frame,
so they still count for ownership, health and tracking, as in the rest of this contract.
`EngineDatasetRun.to_export()` returns a normal `EngineStateExport`, so everything that
consumes this contract (alignment, `evaluate`, the eval CLI) works on fork output
unchanged.
