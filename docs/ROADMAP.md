# Roadmap to the final (2026-12-11)

**Waves** are time-boxed milestones, each closed by a gate you can check. **Lanes** are
parallel streams of work. Story IDs use new prefixes so they don't collide with the
requirement IDs in `REQUIREMENTS.md`. The "Refs" column points back to those IDs.

Owners: **You** = the student, **Agent** = Claude Code running on the PC.

## Lanes

| Lane | Scope |
|---|---|
| **L1 Engine & Data** | The `0A.DDataset` engine fork as the label factory (see `docs/engine-fork.md`). Fallback: the `cvdump` mod. |
| **L2 Model** | Audit `best.pt`, train our own YOLO11-seg, adapter behind `PerceptionModel`. |
| **L3 Events** | Schema 0.3 and every event source. Recall over precision. |
| **L4 Real play & Eval** | PC setup, OBS real-play recordings, live capture, metrics. |
| **L5 Delivery** | Freeze, docs, demo video, slides, rehearsal. |

## Waves

```
      W0 Spike      W1 Data factory  W2 Model v1     W3 Measured     W4 Fix & v2    W5 Frozen
      Nov 25–26     Nov 27–29        Nov 30–Dec 2    Dec 3–5         Dec 6–7        Dec 8–10
L1    ENG-0         ENG-1,2,3,4      ENG-5,6         ──              ENG-5 more
L2    ──            ──               MOD-1,2,3,4     (MOD-5)         MOD-6
L3    ──            EVT-1,2,3        EVT-4,5,6       EVT-7           ──
L4    CAP-1         CAP-2,3          EVAL-1,2        EVAL-3          EVAL-4, CAP-4
L5    ──            ──               ──              ──              ──             PRES-1..4
```

| Wave | Gate |
|---|---|
| **W0 Spike** | The fork, built on the PC, writes one RGB + mask pair from an AI-vs-AI match. **Go/no-go on Nov 26**: if no-go, L1 switches to the `cvdump` fallback (ENG-F) and W1 is spent on it. |
| **W1 Data factory** | ≥2,000 frames from ≥10 matches, each with RGB (HUD on) + instance mask + truth JSON. Masks overlaid on frames look right by eye. |
| **W2 Model v1** | YOLO11-seg trained on the match-split data. Scores at least as well as `best.pt` on held-out matches, or the gap is explained. |
| **W3 Measured** | `docs/results.md` holds the full table: engine-generated vs real play, `best.pt` vs ours vs classical. |
| **W4 Fix & v2** | Before/after table for the top fixes. **Feature freeze at the end of Dec 7.** |
| **W5 Frozen** | `just validate` green on the default config, docs match reality, demo video, slides rehearsed twice. |

## Stories

### L1 Engine & Data

| ID | Story | Owner | Wave | Done when | Refs |
|---|---|---|---|---|---|
| ENG-0 | Build the fork on the PC (distrobox) and run one autostart AI-vs-AI match with `-generate-dataset`. | Agent + You | W0 | one `rgb/` + `mask/` PNG pair on disk | ML2 |
| ENG-1 | Instance masks: encode class + entity id in the mask colour so each entity has its own silhouette. | Agent | W1 | decoding a mask recovers per-entity masks whose ids match the game | E9, ML2 |
| ENG-2 | Save the RGB frame with the GUI/HUD rendered, matching real play. Keep a flag for GUI-off frames (used by MOD-1). | Agent | W1 | RGB shows HUD, minimap, health bars; the flag gives GUI-off frames | ML2 |
| ENG-3 | Per-frame truth JSON next to each pair: entity id → template, owner, hp, position, visibility; player resources, population, phase; camera pose. | Agent | W1 | JSON validates against `infrastructure/data/ground_truth.py` (extend it if needed) | ML2, D6 |
| ENG-4 | Linux match driver replacing `generate_dataset.ps1`: random map/civ/AI difficulty, fixed resolution, N matches unattended. | Agent | W1 | a run of 3 matches completes with no hand-holding | ML1, ML4 |
| ENG-5 | Bulk generation overnight on the PC. | You | W2, W4 | ≥10k frames, ≥30 matches | ML4 |
| ENG-6 | Ingest into 0AD-Eyes: re-split **by match** (run id in the filename), drop near-duplicates, convert masks → YOLO-seg labels. | Agent | W2 | train/val/test have disjoint matches; `data.yaml` builds | ML3, ML5 |
| ENG-F | *Fallback only:* the `cvdump` JS mod from `TpFinalCaptura` on stock 0 A.D. plus homography boxes. | Agent | W1 | replaces ENG-0..4 if W0 is no-go | ML2 |

### L2 Model

| ID | Story | Owner | Wave | Done when | Refs |
|---|---|---|---|---|---|
| MOD-1 | Audit `best.pt` on **our** held-out matches, which it never saw: mIoU per class plus box precision/recall/mAP via connected components. Score it on GUI-on and GUI-off frames, because it was trained without the HUD. | Agent | W2 | baseline row saved | — |
| MOD-2 | Training script for Kaggle/Colab: YOLO11-seg at imgsz 1280, exports weights + a sidecar with input size and class map. | Agent | W2 | runs end-to-end on a toy subset | ML6, ML7 |
| MOD-3 | Train v1 on Kaggle/Colab (the PC's RX 5500 has no ROCm support). | You | W2 | weights + curves in the repo or a release | ML6 |
| MOD-4 | `YoloSegPerceptionModel` adapter behind `PerceptionModel` with config + `--detector yolo`, plus parity tests. | Agent | W2 | `just validate` green | MP4, E1, E2, E8, E9 |
| MOD-5 | *Optional:* retrain the colleague's U-Net on the match split, for a like-for-like comparison. | Agent | W3 | extra row in results | — |
| MOD-6 | v2: retrain with more data, tune imgsz/conf/NMS based on W3 failures. | Agent + You | W4 | before/after numbers | NF3 |
| MOD-7 | *Optional:* ncnn export with Vulkan inference so the PC GPU runs the live demo. | Agent | W4 | adapter runs on Vulkan | NF1 |

### L3 Events (schema 0.2 → 0.3, additive)

Policy: **events over silence.** Emit every observable transition with a confidence.
Thresholds are recall-biased and the consumer filters.

| ID | Story | Wave | Done when | Refs |
|---|---|---|---|---|
| EVT-1 | Event payload: `details` (typed key/values), optional `position` (screen/world), `source` (hud/minimap/entity/system); bump to 0.3.0, keep 0.2 consumers working. | W1 | contract tests + codec round-trip | H1, G8 |
| EVT-2 | One event pipeline: an `EventSource` per surface, merged into `world_model.events`, thresholds in config. | W1 | config snapshot updated | G8, NF7 |
| EVT-3 | HUD events: resource gained/spent (with amount), population changed, cap reached, cap raised, phase changed, idle workers changed, selection changed, alert shown. | W1 | one test per kind | C1–C6 |
| EVT-4 | Minimap events: blip appeared/vanished, enemy in own territory, territory gained/lost, area explored, area back in fog, camera moved. | W2 | one test per kind | D2–D5 |
| EVT-5 | Entity events: entered/left the viewport, entered/left fog, health dropped/recovered, started/stopped moving, construction progress, new building seen, possible unit trained. | W2 | one test per kind | G2, G6 |
| EVT-6 | System events: calibration drift, OCR failed or low confidence, frame dropped, model degraded or stalled. | W2 | one test per kind | B4, A4, NF4 |
| EVT-7 | Event catalogue in `docs/output-contract.md` plus a scrolling event log in the overlay. | W3 | doc + overlay screenshot | H4, X1 |

### L4 Real play & Eval

| ID | Story | Owner | Wave | Done when | Refs |
|---|---|---|---|---|---|
| CAP-1 | PC setup (`docs/pc-setup.md`). | You + Agent | W0 | every check in that doc passes | A1, X8 |
| CAP-2 | Record 4–6 real matches with OBS (fixed resolution, MKV, CQP 15–18). These are the real-play test set and demo material. | You | W1 | files + a recording log (map/civ/notes) | A5, T4 |
| CAP-3 | Replay OBS videos without the frame-clock sidecar; add `docs/config.linux-portal.json`; Bazzite section in the README. | Agent | W1 | test for a sidecar-less replay | A2, X5 |
| CAP-4 | Live capture with the `portal` backend on KDE. Fallback: the demo runs from video. | Agent + You | W4 | `just smoke-live` with the portal config shows a correct overlay | A1, T4 |
| EVAL-1 | Metrics vs truth JSON: box/mask mAP per class, HUD accuracy per field, ownership and health accuracy, minimap blip precision/recall, MOTA, event counts per kind. | Agent | W2 | `zero-ad-eyes eval --dataset` prints a table + JSON | ML8, NF3 |
| EVAL-2 | Hand-check ~50 real-play frames (boxes + HUD values), because auto-labels need an independent check. | You | W2 | labels saved | T3 |
| EVAL-3 | Full run on held-out engine matches + the real-play set, plus `just bench`. | Agent | W3 | `docs/results.md` | NF1–NF3, T6 |
| EVAL-4 | Fix the top 3–5 failures and re-measure. | Agent | W4 | before/after table | T5, NF4 |

### L5 Delivery (W5)

| ID | Story | Owner |
|---|---|---|
| PRES-1 | Make our model the default, `just validate` green, update `REQUIREMENTS.md` §0 + README to reality. | Agent |
| PRES-2 | Demo video: raw + overlay side by side, event log visible. | You |
| PRES-3 | Slides: pipeline, CV technique map (§5.9), label factory, `best.pt` audit, metrics, failures, events, limits. | You + Agent |
| PRES-4 | Rehearse twice; prepare answers on the pixel-only boundary (D1/D6) and the classical vs learned choices. | You |
| PRES-5 | *Optional:* a separate demo consumer that reads the JSONL and asks an LLM for a suggestion. Lives outside the eyes. | Agent |

## Critical path and cut order

Critical path: **ENG-0 → ENG-1/3 → ENG-5/6 → MOD-3 → EVAL-3 → PRES-3.**

If late, cut in this order: PRES-5 → MOD-7 → CAP-4 (demo from video) → MOD-6 → MOD-5 →
ENG-1 (fall back to class masks with connected-component boxes). EVAL-3 is never cut.
