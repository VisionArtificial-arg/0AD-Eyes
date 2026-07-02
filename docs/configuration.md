# Configuration

Every tuning value in the perception layer — thresholds, palettes, HSV windows,
kernel sizes, layout fractions, FPS, accuracy targets — is read from a single typed
config tree. Nothing needs a code change to retune; you edit a JSON file, set an
environment variable, or both.

- The **types** live in `application/settings.py` (`Config` and its sections) — pure
  `pydantic` value objects, no I/O, no OpenCV. They are **defaultless**: every field is
  required and carries only its validation range. There are no tuning values buried in
  the code.
- The **default values** all live in one place — the generator
  `interface/default_config.py::default_config()`. It is the single source of truth for
  "the defaults", surfaced to you through the `config` CLI commands (below) so you never
  read source to discover a default.
- The **loader** lives in `infrastructure/config` (`load_config` / `save_config`).
  `load_config` layers a file and env onto a supplied *base* config (the CLI injects the
  generated defaults as that base).
- The full default tree, expanded, is [`docs/config.example.json`](./config.example.json).
  It is generated from `default_config()` and kept in sync by a snapshot test, so it is
  always an accurate, copy-pasteable template — or regenerate it yourself with
  `zero-ad-eyes config init`.

## The `config` commands

The defaults are data, not code — inspect and generate them without opening a source file:

```bash
zero-ad-eyes config init [PATH]      # write the default config (default: config.json)
zero-ad-eyes config init --force     #   overwrite an existing file
zero-ad-eyes config show             # print the effective config as JSON (defaults)
zero-ad-eyes config show --config my.json   # ... defaults + your file + ZAE_* env
zero-ad-eyes config validate my.json # check a file against the schema (exit != 0 on error)
```

`config init` gives you the whole tree to edit down; `config show` prints exactly what a
run would use after layering; `config validate` fails loudly on a typo'd key or an
out-of-range value before you depend on it.

## Precedence

Lowest to highest:

```
generated defaults (base)   <   JSON file   <   environment variables
```

Anything you don't set keeps its generated default. You never write the whole tree —
only the leaves you want to change. With no `--config` and no env overrides, a command
runs on the generated defaults held in memory; nothing is written to disk unless you ask
(`config init`).

## Using a config file

Point any command at a file with `--config`:

```bash
zero-ad-eyes run   --recording path/to/frames --config my.json
zero-ad-eyes eval  --recording path/to/frames --engine-export gt.json --config my.json
zero-ad-eyes bench --recording path/to/frames --config my.json
```

A **partial** file is the norm — deep-merged onto the defaults:

```json
{
  "thresholds": { "tracking_mota_target": 0.8 },
  "minimap":    { "fog": { "visible_min": 160.0 } },
  "perception": { "ownership_min_fraction": 0.05 },
  "hud":        { "ocr_config": "--psm 6" }
}
```

Only `tracking_mota_target`, `minimap.fog.visible_min`, `perception.ownership_min_fraction`
and `hud.ocr_config` change; every other value stays at its documented default.

## Using environment variables

Any single leaf is overridable without a file. Prefix `ZAE_`, and use `__` (double
underscore) for each level of nesting — matching the JSON path:

```bash
ZAE_THRESHOLDS__TRACKING_MOTA_TARGET=0.8 \
ZAE_MINIMAP__FOG__VISIBLE_MIN=160 \
ZAE_PERCEPTION__OWNERSHIP_MIN_FRACTION=0.05 \
zero-ad-eyes eval --recording frames --engine-export gt.json
```

Env vars win over the file, so they are handy for one-off experiments and CI overrides
on top of a committed base config. Values are JSON-decoded (`0.8` → float, `true` →
bool, `"--psm 6"` → string).

## The sections

| Section | Covers |
|---|---|
| `thresholds` | NF3 accuracy targets (HUD error, mAP, ownership, MOTA) + eval IoU; the single home the ML8 harness derives from |
| `paths` | recordings / calibration directories |
| `overlay` | debug-overlay colours, fonts, sizes (X1) |
| `perception` | ownership palette (E3), `ownership_min_fraction`, `detect_resources` + `resource_cues` (E6a), `health` (E4), `state` selection/construction/garrison cues (E5) |
| `minimap` | palette (D2), `world_extent` (D6), `fog` (D4), `blips` (D2), `territory` (D3), `viewport` (D5), `disc_shape` (D1), `region_confidence` |
| `hud` | top-bar + selection sub-region fractions (C1–C5), `ocr_config` (Tesseract mode) |
| `tracking` | IoU tracker (`iou_threshold`, `min_hits`, `max_staleness`, `decay`) + event thresholds (`combat_drop`, `depletion_health`) |
| `preprocessing` | HUD / scene chain parameters (CLAHE, blur, bilateral) |
| `calibration` | HUD region `ratios`, UI-scale bounds, self-check thresholds (EPIC B) |
| `perf` | NF1/NF2 targets (`latency_target_ms`, `throughput_target_fps`) |
| `pipeline` | `recalibrate_interval` (frames between B4 self-checks) |
| `acquisition` | offline replay (`offline_fps`, `image_extensions`) + live capture (`live_monitor`, `live_fps`) |
| `geometry` | camera projection (`camera_error_tolerance`) + fusion (`fusion_agreement_scale`, `fusion_match_radius`) |

## Recipes

**Retune the enemy player colour** (e.g. a colour-blind palette). Ownership is a
nearest-HSV-window match; override the `enemy` colour's bands:

```json
{
  "perception": {
    "ownership_palette": { "colors": [
      { "name": "blue",   "ownership": "self",  "bands": [ { "h_lo": 100, "h_hi": 130, "s_lo": 70, "s_hi": 255, "v_lo": 50, "v_hi": 255 } ] },
      { "name": "orange", "ownership": "enemy", "bands": [ { "h_lo": 11,  "h_hi": 25,  "s_lo": 70, "s_hi": 255, "v_lo": 50, "v_hi": 255 } ] }
    ] }
  }
}
```

Note: a palette override replaces the whole `colors` list (lists are not merged
element-wise), so include every colour you want, not just the changed one. Each HSV
window is required in full — all six of `h_lo`, `h_hi`, `s_lo`, `s_hi`, `v_lo`, `v_hi`
(there are no per-field defaults). Copy a band from `config show` and edit it.

**Tighten the accuracy gate for a run:**

```json
{ "thresholds": { "hud_read_max_error": 0.005, "ownership_accuracy_target": 0.99 } }
```

**Slow an image-folder replay to 10 fps and change the OCR mode:**

```json
{ "acquisition": { "offline_fps": 10.0 }, "hud": { "ocr_config": "--psm 6" } }
```

## How it maps to the code

Each classical adapter is built from its config section by a `from_settings`
classmethod at the composition root (`interface/cli.py::_build_offline_pipeline`), e.g.
`ClassicalMinimapReader.from_settings(cfg.minimap)`. The pure-data config is rehydrated
into the OpenCV-backed adapter *at that boundary*, so the pipeline itself stays
config-free and the `application` ring never imports OpenCV.

To persist a starting config, dump the defaults and edit down — either via the CLI
(`zero-ad-eyes config init my.json`) or in code:

```python
from zero_ad_eyes.infrastructure.config import save_config
from zero_ad_eyes.interface.default_config import default_config

save_config(default_config(), "my.json")  # then delete everything you don't want to override
```

## Notes / limits

- **Structural constants stay in code** (epsilons, the hue-179 OpenCV ceiling,
  minimum-size guards) — they are correctness invariants, not tuning knobs.
- **Live capture** (`acquisition.live_*`) drives `zero-ad-eyes run --live`, which builds a
  `ScreenCaptureSource.from_settings(cfg.acquisition)` into the same classical chain as
  `--recording` (needs a display + the `mss` backend to actually grab; `--frames` bounds it).
- **Geometry / fusion** (`geometry.*`) drives the fusion stage (G4/G5): the pipeline folds
  minimap blips into the tracked entity set via `ClassicalEntityFuser.from_settings(cfg.geometry)`.
  Today it surfaces blips **outside the camera viewport** as their own world-space entities and
  drops in-viewport ones (which would double-count on-screen tracked entities). Full
  screen↔world merge of an on-screen blip with its tracked entity awaits the F1 camera
  projector (no screen→world correspondences exist offline yet); the stage already routes
  through `fuse_entities` with `cfg.geometry`, so that merge activates unchanged once entities
  carry world positions. The deep calibration anchor/agreement constants remain structural.
- The offline pipeline applies **no preprocessing by default**; `preprocessing`
  parameterises the HUD/scene chain factories for when a chain is enabled.
