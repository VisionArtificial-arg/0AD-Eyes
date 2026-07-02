# 0AD-Eyes — Requirements

> The **perception layer** for an autonomous 0 A.D. player. Its single responsibility
> is to convert what is *rendered on screen* into a structured, machine-readable
> model of the game world, which is then handed to a separate **decision layer**.
> This document enumerates every task required to consider *this layer* finished.

---

## 0. Implementation Status (updated 2026-07-01)

Snapshot of what is built. Checkboxes below are kept in sync with this table.

- **Output contract: `SCHEMA_VERSION` = `0.2.0`.** The world model carries entities,
  HUD (incl. **selection**), minimap (incl. **fog grid** + **territory**), per-entity
  **motion**, and an **events** list; `Frame` carries a **crop_offset**. All v0.2
  fields are populated **end-to-end** by the classical pipeline (empty/`None` only
  when honestly unobservable — never fabricated).
- **`just validate` is green:** 397 passed, 1 skipped (the skip is live screen
  capture — the X server in the build sandbox rejects `GetImage`; the code exists,
  the environment can't exercise it). `check` (ruff + mypy) and `test` are clean;
  `eval` runs the real ML8 harness and reports the 🔌 metric as `pending-model`.

| Area | State |
|------|-------|
| A Acquisition · P Preprocessing · B Calibration · C HUD · D Minimap | **Done** |
| E main-viewport **classical** (E3/E4/E5/E6a/E7/E10/E11) | **Done** |
| E main-viewport **learned** (E1/E2/E6b/E8/E9) | **🔌 blocked on model team (MP4)** |
| F Geometry · G Tracking/Fusion/Motion/Events/Spatial · H Contract | **Done** |
| Model seam MP1–MP3, MP5 (port + stub + parity) | **Done**; **MP4** 🔌 |
| Data ML1/ML2/ML3/ML8 | **Done**; ML4/ML5/ML6 🔌 (model team); ML7 blocked on MP4 |
| Tooling X1–X8 (overlay, replay CLI, config, uv+just, docs, CI, lock) | **Done** |
| Tests T1–T3, T5 | **Done**; **T4** live-smoke (needs a real display) and **T6** perf/NF1–NF2 benchmarks **open** |

**Integration decisions taken (2026-07-01), now implemented:**
- `EntityEnricher` **port** + classical adapter: a post-tracking stage folds
  ownership/health/selection onto tracked entities, *fill-if-absent* (never clobbers
  a confident learned value).
- `EventSource` **port** + `ClassicalEventDetector`: frame-over-frame transitions →
  `world_model.events`.
- **Fusion single-home (G5):** `resolve_conflict` owns entity fusion and delegates
  the two-estimate spatial blend to F3 `reconcile`; inverse-variance is the documented
  eventual target.
- **Onion fix:** config value-object *types* live in `application.settings`; the
  JSON/env loader stays in `infrastructure.config` (no interface→infra sideways dep).
- **CLI:** `zero-ad-eyes run --recording PATH` drives the full classical chain
  (`--detector classical` opts into the E6a baseline; the model seam stays stubbed).
- **Gate/DX:** `just eval` runs the real harness; git hooks (`pre-commit=check`,
  `pre-push=validate`) + `just setup`.

**Reached:** M0–M2, **M3-scaffold**, M4, M5, and the non-🔌 parts of M6. **DoD-A**
(this team's scope, stub proving the seam) is met except T4/T6. **DoD-B** remains
open on the model artifact (MP4) and the NF1/NF3 gates it unblocks.

---

## 1. Purpose & Scope

### 1.1 Mission
Given only the pixels 0 A.D. draws, produce a continuously-updated **world model**
(entities, resources, territory, fog of war, HUD state) and expose it through a
stable contract to a downstream decision layer.

### 1.2 In scope
- Frame acquisition (live capture and offline recordings).
- Interpretation of every readable visual surface: HUD, minimap, main viewport.
- Entity-level perception (position, type, ownership, health, state).
- Temporal tracking and world-model aggregation.
- A versioned output contract consumed by the decision layer.
- The dataset/annotation/training/evaluation machinery needed to build the above.

### 1.3 Explicitly out of scope
- **Any decision-making, planning, or actuation** (movement, orders, build logic).
- **Input injection** into the game (mouse/keyboard automation).
- Modifying the game, its engine, or installing mods to expose data.

---

## 2. Foundational Decisions (confirmed with stakeholder)

These four choices are treated as **facts** for this document. Changing any of them
reopens large parts of the requirements.

| # | Decision | Value | Consequence |
|---|----------|-------|-------------|
| D1 | Data-source boundary | **Pure pixel / black-box** | At inference time, *only* rendered frames may be read. No engine, log, memory, or sim-state access. |
| D2 | Runtime mode | **Both** | Must run live in near real-time *and* batch-process recordings offline. |
| D3 | Stack | **Python + OpenCV/ML** | OpenCV, NumPy; PyTorch/ONNX for learned components; Tesseract or a learned OCR. |
| D4 | Output granularity | **Full entity-level** | Per-entity position/type/owner/health/state, not just aggregates. |
| D5 | Tooling | **uv + just** | `uv` manages the environment/deps; `just` is the task runner. Every CI action is a `just` recipe (§9.1). |
| D6 | Training vs production data | **Engine data for training/eval only; production is vision-only** | Engine-derived state is authoritative ground truth offline; at inference the pipeline reads pixels exclusively. Non-negotiable boundary. |

### 2.1 Training/evaluation ground truth (resolved — D6)
D1 forbids engine access **at inference**. It says nothing about *how labels are
produced for training/evaluation*. 0 A.D. is open source and can emit authoritative
simulation state (via replays / a debug build / the AI-interface JSON).

**Decision (D6):** engine-derived state **may** be used as ground truth for building
datasets and measuring accuracy. In **production/inference the pipeline uses vision
only** — no engine, log, memory, or sim-state access. A Definition-of-Done audit
(§11) enforces that no engine dependency leaks into the inference path.

---

## 3. Assumptions

State explicitly; each is an assumption, not a fact, until confirmed.

- **A1** Target is the standard 0 A.D. RTS HUD (top resource/phase bar, bottom-left
  minimap with fog of war, bottom-center selection panel, on-map player-colored
  entities with health bars). No radical UI mods.
- **A2** The decision layer runs in the same process space or over a local IPC
  boundary; the contract can be an in-process object or a serialized message.
- **A3** Taxonomy scope (OQ-4): **multi-civilization, exact entity types**. The
  taxonomy enumerates precise unit/building/resource types across the targeted
  civilizations; it is config-driven (NF7) and extensible.
- **A6** Perspective (OQ-5): **single-player, own point-of-view only.** Only the
  playing player's screen/HUD is captured. Enemy resources, population, and any
  fog-hidden state are **physically unobservable** and must never be promised to
  the decision layer.
- **A4** Screen resolution, UI scale, and HUD theme are *knowable per session* but
  not fixed across sessions → calibration is required, not hard-coded coordinates.
- **A5** We control the capture environment (we can pick resolution, disable
  distracting overlays) for the initial milestones.

---

## 4. The Output Contract (World Model)

The deliverable of this layer is this schema. Everything else exists to populate it.
Tiered by priority: **[M]** must-have, **[S]** should-have, **[C]** could-have.

### 4.1 Frame / session metadata `[M]`
- Monotonic frame id + capture timestamp.
- Source (live | recording id), resolution, UI-scale calibration handle.
- Per-field **confidence** and **staleness** (frames since last observed).

### 4.2 Player / global HUD state `[M]`
- Resource stockpiles: **food, wood, stone, metal** (integer counts).
- Population: current / cap.
- Game phase: Village / Town / City.
- Own player color and civ emblem (self-identification).
- `[S]` Resource gather rates if visually derivable; idle-worker count; alerts.

### 4.3 Entities (main viewport) `[M]`
Per detected entity:
- Screen bounding box/centroid **and** estimated world position (§4.6).
- Class taxonomy: unit vs building vs resource-node vs projectile/other, plus
  finer type when legible (e.g. worker, cavalry, house, farm, tree, mine).
- **Ownership** by player color (self / ally / enemy / gaia-neutral).
- Health fraction (from health bar) `[M]`; rank/veterancy pips `[C]`.
- Selection state, garrison indicator, construction/production progress `[S]`.

### 4.4 Minimap model `[M]`
- Blips with world position + owner color.
- Territory / border ownership regions.
- Fog-of-war state per cell: unexplored / explored-not-visible / visible.
- Camera viewport rectangle projected onto the minimap.

### 4.5 Fog of war (main view) `[S]`
- Classification of viewport regions into the three visibility states.

### 4.6 Coordinate system `[M]`
- A documented mapping **screen-pixel ⇄ world-coordinate** (and minimap ⇄ world),
  so downstream consumers reason in world space, not pixels.

### 4.7 Contract engineering `[M]`
- Versioned schema (semver) with a serialization format (e.g. JSON/protobuf).
  **Implemented — `SCHEMA_VERSION = 0.2.0`** (v0.2 added the fog grid, territory,
  selection, per-entity motion, events, and `Frame.crop_offset`; new fields default
  empty/`None`, so 0.1 consumers stay compatible).
- Backward-compatibility policy; every field carries confidence + provenance.
- **Transport-agnostic (OQ-3 deferred):** the schema is designed independently of
  the delivery mechanism; the concrete IPC (in-process / socket / shared memory) is
  chosen at the M5 integration milestone without reworking the schema.

---

## 5. Functional Requirements — Epics & Tasks

Actionable checklist. IDs are stable references for planning/commits.

> **🔌 marker = depends on the trained model** (owned by the **model team**, not
> this one). These tasks are *stubbed behind the model port* (§5.10) and cannot be
> functionally validated until the model artifact is delivered. **Everything without
> 🔌 is buildable now** and is exactly the "plug-and-play" scope this layer targets.

### EPIC A — Frame Acquisition `[M]`
- [x] **A1** Live screen/window capture at a target FPS (region = game window).
- [x] **A2** Offline reader: video files and image-sequence folders behave
      identically to live (same downstream interface).
- [x] **A3** Unified `FrameSource` abstraction so live/offline are interchangeable.
- [x] **A4** Timestamping, frame-drop detection, backpressure handling.
- [x] **A5** Recording/dump mode: persist captured frames for dataset building.

### EPIC P — Preprocessing `[M]`
Shared frame-conditioning stage between acquisition and all perception. Every
perception epic consumes the preprocessed frame, not the raw capture.
- [x] **P1** Preprocessing pipeline scaffold (composable, per-consumer variants).
- [x] **P2** Color-space transforms (RGB↔HSV/Lab) to make UI, player colors, and
      resource nodes separable (CV-03).
- [x] **P3** Normalization: reduce brightness/contrast/render variation (CV-04).
- [x] **P4** Noise filtering: remove compression/render artifacts (CV-05).
- [x] **P5** Contrast enhancement (e.g. CLAHE) for small/low-contrast objects (CV-06).
- [x] **P6** Edge detection (Canny/Sobel) feeding classical detection & contours (CV-07).
- [x] **P7** Region-of-Interest (ROI) gating: restrict heavy processing to relevant
      areas (HUD boxes, viewport, motion regions) for latency (CV-28).

### EPIC B — Calibration & UI Layout `[M]`
- [x] **B1** Detect resolution and UI scale per session.
- [x] **B2** Locate HUD regions (top bar, minimap, selection panel) robustly to
      resolution/theme — anchor detection, not hard-coded pixels (per A4).
- [x] **B3** Persist/reuse calibration profiles keyed by resolution+theme.
- [x] **B4** Self-check: flag when the live layout no longer matches calibration.

### EPIC C — HUD Parsing `[M]`
- [x] **C1** OCR/number-reading of the four resource counters (§4.2).
- [x] **C2** Population current/cap parsing.
- [x] **C3** Phase detection (icon/text classification).
- [x] **C4** Self-identification: own player color + civ.
- [x] **C5** Selection panel reader: selected entity type, health, queues `[S]`.
- [x] **C6** Robustness to number formatting, transient tooltips, notifications.

### EPIC D — Minimap Interpretation `[M]`
- [x] **D1** Segment the minimap disc/region from calibration.
- [x] **D2** Blip detection + owner-color classification.
- [x] **D3** Territory/border region extraction.
- [x] **D4** Fog-of-war cell classification on the minimap.
- [x] **D5** Camera-viewport rectangle extraction.
- [x] **D6** Minimap-pixel → world-coordinate calibration.

### EPIC E — Main-Viewport Perception `[M]`
Primary strategy is **learned segmentation/detection**; classical methods
(template matching, contours) are baselines and fallbacks for fixed/known art.
- [ ] 🔌 **E1** Entity **detection** + **localization** in the 3D scene (CV-10/CV-11).
- [ ] 🔌 **E2** Entity **classification** into the taxonomy (§4.3), extensible (CV-14).
- [x] **E3** **Ownership** assignment via player-color segmentation, robust to
      lighting/terrain/shadow and to color-blind palettes. *(classical — not model-dep)*
- [x] **E4** Health-bar reading → health fraction. *(classical — not model-dep)*
- [x] **E5** State cues: selection ring, construction scaffold, garrison, rank `[S]`.
- [x] **E6a** On-map resource-node detection — **classical baseline** (template
      match on fixed resource art + color/contour cues for trees/mines/bushes/fauna).
      Coarse recall, works **without the model**. *(not model-dep)*
- [ ] 🔌 **E6b** Resource-node detection — **learned refinement** via the model
      (better recall/precision in cluttered 3D scenes, exact sub-type). Supersedes
      E6a where confident; falls back to E6a otherwise.
- [x] **E7** Occlusion / partial-visibility handling (CV-27). *(logic buildable on
      stub; validated only against real E1/E8 output)*
- [ ] 🔌 **E8** **Semantic segmentation** (U-Net or equivalent): per-pixel class map of
      the scene (CV-12; pixel-wise classification CV-29). Feeds E1/E2 + scene understanding.
- [ ] 🔌 **E9** **Instance segmentation**: separate individual entities of the same
      class (CV-13).
- [x] **E10** Mask post-processing: connected-component analysis, contour
      detection, morphological open/close to clean masks (CV-30/CV-31/CV-32).
- [x] **E11** **Template matching** + feature descriptors for fixed UI icons and
      known static art (CV-08/CV-09), as a deterministic complement to E1/E2.

### EPIC F — Camera & Coordinate Geometry `[M]`
- [x] **F1** Recover camera pose / ground-plane homography from visual cues so
      screen detections project to world coordinates.
- [x] **F2** Handle camera pan/zoom/rotation across frames.
- [x] **F3** Reconcile main-view world positions with minimap world positions.
- [x] **F4** Quantify projection error; expose it as confidence.

### EPIC G — Temporal Tracking & World Model `[M]`
- [x] **G1** Single- and **multi-object tracking**: assign stable ids across
      frames and maintain trajectories (CV-15/CV-16).
- [x] **G2** Track lifecycle: birth, death, enter/leave fog, enter/leave viewport.
- [x] **G3** **Memory** of entities currently off-screen or in explored-but-not-
      -visible fog, with staleness decay (per §4.1).
- [x] **G4** Fuse main-view + minimap + HUD into one coherent world model
      (scene understanding + world-state reconstruction, CV-20/CV-21).
- [x] **G5** Conflict resolution when sources disagree (confidence-weighted).
- [x] **G6** **Optical flow** + **motion estimation**: pixel motion → entity
      direction/speed; also drives ROI gating (CV-17/CV-18).
- [x] **G7** **Temporal analysis** across frames to stabilize perception (CV-19).
- [x] **G8** **Event detection**: combat, resource depletion, building completion,
      unit produced/lost (CV-34), derived from temporal state changes.
- [x] **G9** **Spatial reasoning**: proximity, visibility, occupancy relations
      between entities (CV-33).

### EPIC H — Output Contract & Integration `[M]`
- [x] **H1** Define and version the world-model schema (§4).
- [x] **H2** Serialization + in-process API for the decision layer.
- [x] **H3** Publish cadence (per-frame vs on-change) and latency budget.
- [x] **H4** Consumer-facing docs + example client stub.

### 5.9 CV Technique Catalog (method → epic → milestone)

The techniques the layer draws on, each with a stable **CV-ID**, its owning epic
task(s), and the milestone where it first appears. This is the traceability bridge
between "the method" and "the work". Ordered as the canonical pipeline flow.

| CV-ID | Technique | Purpose (what it buys us) | Owner | Milestone |
|-------|-----------|---------------------------|-------|-----------|
| CV-01 | Image acquisition (screen capture) | Get the visual input the agent perceives | A1 | M0 |
| CV-02 | Image preprocessing | Condition frames for later stages | P1 | M0 |
| CV-03 | Color-space transforms | Make UI / player colors / resources separable | P2 | M1 |
| CV-04 | Image normalization | Reduce brightness/contrast/render variation | P3 | M1 |
| CV-05 | Noise filtering | Remove artifacts that mislead detection | P4 | M1 |
| CV-06 | Contrast enhancement | Surface small / low-contrast objects | P5 | M1 |
| CV-07 | Edge detection | Object boundaries for classical vision | P6 | M2 |
| CV-08 | Feature extraction | Descriptors for recognition/tracking | E11 | M1 |
| CV-09 | Template matching | Recognize fixed UI icons / static art | C, E11 | M1 |
| 🔌 CV-10 | Object detection | Locate units/buildings/resources | E1 | M3 |
| 🔌 CV-11 | Object localization | Position of detections on screen | E1 | M3 |
| 🔌 CV-12 | Semantic segmentation (U-Net) | Per-pixel class map of the scene | E8 | M3 |
| 🔌 CV-13 | Instance segmentation | Separate same-class objects | E9 | M3 |
| 🔌 CV-14 | Object classification | Exact type of each entity | E2 | M3 |
| CV-15 | Object tracking | Follow one entity across frames | G1 | M4 |
| CV-16 | Multi-object tracking | Ids + trajectories for many entities | G1 | M4 |
| CV-17 | Optical flow | Pixel motion between frames | G6 | M4 |
| CV-18 | Motion estimation | Direction/speed of moving entities | G6 | M4 |
| CV-19 | Temporal analysis | Use past frames to stabilize/derive events | G7 | M4 |
| CV-20 | Scene understanding | Semantic interpretation of the scene | G4 | M4 |
| CV-21 | World-state reconstruction | Visual info → structured world model | G4, H1 | M4→M5 |
| CV-22 | Coordinate mapping | Image coords → game/nav space | D6, F1 | M2→M3 |
| CV-23 | Perspective transformation | Correct/simplify scene geometry | F1 | M3 |
| CV-24 | Homography (where applicable) | Align planar views (main ⇄ minimap/grid) | F1, F3 | M2→M3 |
| CV-25 | Geometric transformations | Normalize scale/rotation/orientation | F2 | M3 |
| CV-26 | Minimap-to-world correlation | Tie minimap info to the main view | D6, F3 | M2 |
| CV-27 | Occlusion handling | Keep recognizing partially hidden entities | E7 | M3 |
| CV-28 | Region-of-Interest detection | Process only relevant areas (efficiency) | P7 | M1 |
| 🔌 CV-29 | Pixel-wise classification | Class of every pixel | E8 | M3 |
| CV-30 | Connected-component analysis | Group segmented pixels into entities | E10 | M3 |
| CV-31 | Contour detection | Outlines of segmented objects | E10 | M3 |
| CV-32 | Morphological operations | Refine masks (denoise / fill gaps) | E10 | M3 |
| CV-33 | Spatial reasoning | Proximity/visibility/occupancy relations | G9 | M4 |
| CV-34 | Event detection | Combat, depletion, build-complete, losses | G8 | M4 |
| CV-35 | Visual state estimation | Game state **from vision only** (the goal) | G4, H1 | M4→M5 |

**Classical vs learned — a selection decision, not a mandate.** Several of these are
*alternative* routes to the same output, not a fixed sequence to implement wholesale:

- **Classical path** (CV-07/08/09/30/31/32): edges → features/template match →
  contours/connected-components/morphology. Cheap, deterministic, no training data,
  great for **fixed UI and known static art**. Weak on the varied 3D scene.
- **Learned path** (CV-10/11/12/13/14/29): detection + semantic/instance
  segmentation + classification. Handles the hard 3D scene and multi-civ exact
  types, at the cost of data (D6) and a latency budget (NF1).

*Engineering stance (my recommendation):* use the **classical path as the baseline
and the cross-check** — it is the fast route to M1/M2 value and a cheap validator
for the learned models — and reserve the **learned path for EPIC E** where nothing
else suffices. Pick per-element empirically against NF1/NF3; do not implement every
technique unconditionally. CV-35 (visual state estimation) is not a task — it is the
**name of the whole layer's output** and is realized by G4/H1 aggregating the rest.

### 5.10 Model Dependency & Plug-and-Play Boundary

**Goal of this section:** isolate the trained model (owned by the *model team*)
behind a single, versioned seam so this team can build, run, and test **the entire
rest of the layer now**, then swap the real model in with zero downstream changes.

#### 5.10.1 What depends on the model (🔌)
- **Tasks:** E1, E2, E6b, E8, E9.
- **Techniques:** CV-10, CV-11, CV-12, CV-13, CV-14, CV-29.
- **Milestone:** these are the *only* things that block **M3-live**; M0–M2 and the
  *scaffolding* of M4–M6 do not depend on the model.
- **Not model-dep (build now):** acquisition (A), preprocessing (P), calibration (B),
  HUD/OCR (C — OCR uses off-the-shelf Tesseract, *not* the team's model),
  minimap (D), ownership/health/state (E3/E4/E5), classical resource baseline (E6a),
  template/classical path (E7/E10/E11),
  geometry (F), tracking/motion/temporal/events/world-model (G), contract (H),
  tooling (X). All of these consume perception results **through the port**, never
  the model directly.

#### 5.10.2 The seam — `PerceptionModel` port `[M]` (build now)
A single interface (onion: a **port**; the model is an infrastructure **adapter**)
is the only thing the rest of the pipeline knows about. Nothing downstream imports
a model, a framework, or weights.

- [x] **MP1** Define `PerceptionModel` port: `infer(preprocessed_frame, roi?) ->
      Detections`, where `Detections` is a stable value object
      (per item: mask/bbox in a documented coordinate convention, class label from
      the §4.3 taxonomy, score/confidence; plus optional dense segmentation map).
- [x] **MP2** Freeze the **I/O contract**: input tensor spec (size, channel order,
      normalization from EPIC P), output schema, class-id↔taxonomy mapping,
      coordinate + scale convention. Versioned; the model team builds to this.
      - Each detection carries a **first-class confidence** *and* a **provenance
        tag** (`classical` E6a / `learned` E6b, extensible to other elements) so
        downstream (G/H) weights coarse-vs-refined signals identically whether they
        came from the stub or the real adapter. Resource-node items are the driving
        case; the field is generic, not resource-specific.
- [x] **MP3** Ship a **stub adapter** implementing the port without a real model:
      (a) *fixture mode* — replays hand-labeled/engine-ground-truth (D6) detections
      for recorded frames; (b) *classical mode* — the E11 template/contour path for
      the subset it can cover. Lets G/H/X run end-to-end today.
- [ ] **MP4** Ship the **real adapter** (loads the delivered artifact, e.g. ONNX
      via onnxruntime) — the *only* new code needed at plug-in time.
- [x] **MP5** Adapter-parity tests: same port contract satisfied by stub and real
      adapter (T2 golden fixtures run against both).

#### 5.10.3 What the model team must deliver (the plug)
Acceptance checklist for the artifact hand-off — until met, MP4 stays open:
- Model artifact in the agreed export format (ONNX target) + version.
- Exact I/O tensor spec matching **MP2** (or a documented adapter shim).
- Class-id → taxonomy (§4.3) mapping for the targeted civilizations (OQ-4).
- Expected preprocessing (must be expressible by EPIC P) and input resolution.
- Measured performance envelope (latency/throughput) to check against NF1/NF2.
- Held-out accuracy figures to check against NF3.

#### 5.10.4 Consequence for validation
- `just check` / `just test` (T1–T3, T5) run fully on the **stub** — CI is green
  without the model.
- The **NF3 detection/classification/segmentation thresholds** (mAP, per-type
  accuracy) and the full **NF1 live-latency** figure are *unmeasurable until MP4*;
  the eval harness (ML8) reports them as `pending-model` rather than failing.
- **Definition of Done** for *this team's* scope = everything except the 🔌 tasks,
  with the stub proving the seam. Full DoD (§11) closes when MP4 lands.

---

## 6. Non-Functional Requirements

- **NF1 Latency (live) — target set (OQ-2, Aggressive):** end-to-end
  frame→world-model **≤ ~66 ms** (one frame at 15 FPS); measured, not assumed.
- **NF2 Throughput — target set:** sustain **15–30 FPS** without unbounded queue
  growth.
- **NF3 Accuracy targets — thresholds set (OQ-2, Aggressive):** HUD/resource read
  error **< 1%**, detection **mAP ≥ 0.80**, ownership accuracy **≥ 98%**, tracking
  **MOTA ≥ 0.70**, on the held-out test set. These are the `just validate` gate.
- **NF4 Robustness:** degrade gracefully (emit low-confidence, never crash the
  consumer) under occlusion, HUD popups, unusual maps/biomes, night maps.
- **NF5 Determinism (offline):** same recording → same world model (reproducible).
- **NF6 Observability:** structured logs, per-stage timing, health metrics.
- **NF7 Configurability:** all thresholds/paths/taxonomy in config, not code.
- **NF8 Portability:** runs on the target OS/GPU; documented dependencies.

---

## 7. Data & ML Requirements

> **Ownership split.** Training the model is the **model team's** scope (🔌). This
> team owns only the pieces needed to *feed* that team and to *plug in* the result:
> data capture, ground-truth export, the integration seam, and the eval harness.
> Marked **[model team]** vs **[this team]** below.

- [x] **ML1** Dataset collection pipeline (uses EPIC A recording mode). **[this team]**
- [x] **ML2** Ground-truth pipeline (per **D6**): extract authoritative state from
      the open-source engine (replays / debug build / AI-interface JSON) and align
      it to captured frames as labels. Offline-only; never linked into inference.
      **[this team]** (deliverable *to* the model team).
- [x] **ML3** Annotation tooling/format for detection, classification, ownership,
      health, fog states, minimap blips. **[shared]**
- [ ] 🔌 **ML4** Dataset coverage matrix: civilizations, biomes/maps, day/night,
      resolutions, zoom levels, crowded vs sparse scenes. **[model team]**
- [ ] 🔌 **ML5** Train/val/test split policy (no leakage across matches). **[model team]**
- [ ] 🔌 **ML6** Model selection & training pipeline for learned components
      (E1/E2/E6b/E8/E9). **[model team]**
- [ ] **ML7** Model versioning + export consumed via the **real adapter (MP4)**;
      this team owns the inference integration behind the port, not the training.
      **[this team]** — blocked on the MP2 I/O contract, not on the weights.
- [x] **ML8** Continuous evaluation harness with the NF3 metrics; reports
      `pending-model` for 🔌 metrics until MP4. **[this team]**

---

## 8. Testing & Validation

- [x] **T1** Unit tests for deterministic CV stages (calibration, segmentation).
- [x] **T2** Golden-frame regression tests (frozen inputs → expected outputs).
- [x] **T3** Integration test: recording → full world model, scored vs ground truth.
- [ ] **T4** Live smoke test on a real match with the debug overlay.
- [x] **T5** Adversarial/edge scenes: heavy fog, mass battles, rare units, mods off.
- [ ] **T6** Performance benchmarks enforcing NF1/NF2 in CI.

---

## 9. Tooling & Developer Experience

- [x] **X1** Debug **overlay**: render the world model back onto the frame
      (boxes, ids, owner colors, health, fog) — primary sanity tool.
- [x] **X2** Recording/replay CLI for offline runs.
- [x] **X3** Config system + calibration-profile store.
- [x] **X4** Repo scaffolding managed by **uv** (env + locked deps via
      `pyproject.toml` / `uv.lock`) and **just** as the task runner (§9.1);
      package layout, CI, linting, formatting (8-space indentation per convention).
- [x] **X5** README + architecture doc (onion boundaries: acquisition → perception
      → tracking → world model → contract).

### 9.1 `just` command surface (required)

The `justfile` is the single entry point for humans and CI; CI runs the same
recipes a developer runs locally. Recipes wrap `uv run …` so the environment is
always the locked one.

| Recipe | Purpose | Roughly wraps |
|--------|---------|---------------|
| `just check` | Static gates: lint, format-check, type-check. Fast, no game needed. | `ruff check` / `ruff format --check` / `mypy` |
| `just test` | Automated tests: T1–T3, T5 (deterministic + integration). | `pytest` |
| `just eval` | ML8 accuracy harness; 🔌 metrics report `pending-model`, else score a `--dataset`. | `uv run zero-ad-eyes eval` |
| `just validate` | **The full CI action** = `check` + `test` + `eval`. Green ⇒ CI green. | composition of the above |
| `just run` | Run the pipeline; `--recording PATH` drives the real classical chain (X1 overlay / live source once a display is available). | `uv run zero-ad-eyes run …` |
| `just fmt` / `just setup` | Auto-format + apply safe fixes / activate committed git hooks + `uv sync`. | `ruff` / `git config core.hooksPath` |

> **Git hooks (committed):** `pre-commit` runs `just check`, `pre-push` runs
> `just validate`; enable with `just setup`. **NF1/NF2 perf benchmarks (T6) are not
> yet wired into `just test`.**

- [x] **X6** Author the `justfile` with at least `check`, `test`, `validate`, `run`.
- [x] **X7** Wire CI to invoke **`just validate`** and nothing else (parity with local).
- [x] **X8** `uv`-managed, reproducible env; committed `uv.lock`; documented Python
      version pin.

---

## 10. Suggested Milestones (build order, not commitments)

> **Status (§0):** M0–M2, **M3-scaffold**, M4, M5 reached; M6 non-🔌 parts done.
> Only **M3-live** (real detection/segmentation via MP4) is outstanding, plus the
> NF1/NF2 perf gate.

1. **M0 Skeleton:** repo, `FrameSource` (live+offline), preprocessing scaffold,
   debug overlay, empty schema (EPIC A + P1).
2. **M1 HUD:** preprocessing (EPIC P) + resources/pop/phase/self-id read reliably
   (EPIC B+C), classical/template path (CV-07/08/09).
3. **M2 Minimap:** blips, fog, territory, viewport rect (EPIC D).
4. **M3 Entities:** split by the model boundary —
   - **M3-scaffold `[this team, now]`:** model port + stub adapter (MP1–MP3),
     classical/ownership/health path (E3/E4/E5/E6a/E7/E10/E11), camera geometry (F).
     Runs end-to-end on the stub.
   - **M3-live `[🔌 blocked on model]`:** real detection/classification/segmentation
     (E1/E2/E6b/E8/E9) via the real adapter (MP4). *Only* part gated on the model team.
5. **M4 Temporal:** tracking + memory + fusion into the world model (G). Built and
   tested against the stub; no model dependency.
6. **M5 Contract:** versioned output + decision-layer integration (H). No model dep.
7. **M6 Hardening:** NF targets, dataset/ML maturation, CI benchmarks. 🔌 accuracy
   gates (mAP/per-type) close only after MP4.

---

## 11. Definition of Done (for *this layer*)

> Two-tier DoD because of the model boundary (§5.10.4). **DoD-A (this team, now)**
> = everything except the 🔌 tasks, with the stub adapter proving the port. **DoD-B
> (full)** = the below, reached once the real adapter (MP4) lands. The list below is
> DoD-B; strike the 🔌-gated lines to read DoD-A.
>
> **Status:** DoD-A met **except two lines** — T4 live-smoke (needs a real display)
> and the NF1/NF2 perf benchmark (T6). Everything else below is satisfied on the
> stub/classical path.

- All **[M]** epics implemented and passing T1–T4 (🔌-gated items on the stub for DoD-A).
- **`just validate` is green** (check + test + accuracy eval) in CI (X7).
- NF3 accuracy thresholds met on the held-out test set (numbers agreed upfront).
- NF1/NF2 latency & throughput met live, enforced in CI (T6).
- Versioned contract (H1) documented; a decision-layer stub consumes it (H4).
- Debug overlay (X1) visually confirms the world model on live play.
- **Pixel-only inference audit passes (D1/D6):** the production/inference path has
  zero dependency on engine, logs, memory, or sim state — only frames.

---

## 12. Risks

- **R1** Pure-pixel entity classification in a dense, 3D, occluded RTS scene is
  hard; mAP may be the binding constraint. *Mitigation:* start with HUD/minimap
  (high-signal, 2D) which alone yields a usable coarse world model.
- **R1b (compounded target risk)** The chosen bar combines the two hardest
  choices: **Aggressive real-time (OQ-2)** *and* **multi-civ exact types (OQ-4)**.
  Exact-type classification across civilizations at mAP ≥0.80 within a ~66 ms
  budget is the single most demanding combination in this document. *Mitigation:*
  stage it — hit broad-class detection at speed first, then refine to exact types;
  treat the full NF3 bar as an M6 acceptance gate, not an M3 one. *Recommendation:*
  consider relaxing either FPS or type-granularity for the first release if data
  shows the combined target is not reachable.
- **R2** Screen→world projection error compounds through tracking. *Mitigation:*
  treat minimap as an independent position source and fuse (G4/G5).
- **R3** Live latency vs accuracy tension. *Mitigation:* tiered pipeline; cheap
  per-frame updates + heavier periodic passes.
- **R4** Player-color ambiguity (similar colors, terrain tint, color-blind modes).
  *Mitigation:* calibrate the active palette per session (B3).
- **R5** Ground-truth alignment: matching engine state (D6) to the exact captured
  frame (timing/coordinate skew) can be error-prone. *Mitigation:* timestamp both
  sources and validate alignment on known scenes.
- **R6 (plug-and-play's real risk)** Contract drift: if the **MP2 I/O contract**
  (tensor spec, taxonomy mapping, coordinate convention, expected preprocessing) is
  not frozen and agreed with the model team *up front*, the stub and real adapters
  diverge and "plug-and-play" becomes a rewrite. *Mitigation:* MP2 is the very first
  cross-team artifact; the model team builds *to* it; MP5 parity tests enforce it in
  CI. This is the single coordination point that makes or breaks the split.

---

## 13. Open Questions (blocking where noted)

- **OQ-1 — RESOLVED (D6):** engine-derived state may be used as training/eval
  ground truth; production/inference is vision-only.
- **OQ-2 — RESOLVED (Aggressive):** live 15–30 FPS, ≤ ~66 ms latency; HUD error
  <1%, mAP ≥0.80, ownership ≥98%, MOTA ≥0.70. Encoded in NF1–NF3.
- **OQ-3 — OPEN (non-blocking until M5):** exact IPC/contract mechanism
  (in-process object vs local socket vs shared memory). Schema stays
  transport-agnostic (§4.7) until then.
- **OQ-4 — RESOLVED:** multi-civilization, exact entity types (A3).
- **OQ-5 — RESOLVED:** single-player, own-POV only (A6); enemy/fog-hidden state is
  unobservable by construction.
