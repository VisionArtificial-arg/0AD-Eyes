# 0AD-Eyes

The **eyes** of an autonomous 0 A.D. player, for a Computer Vision course final exam on
**2026-12-11**. The pipeline turns rendered frames into a versioned JSON world model
plus events. A separate decision model consumes that JSON and lives outside this repo.
The student works solo, and the CV quality is what gets graded.

## Rules

- **The eyes only report what they see.** Decisions and game input belong to the
  consumer.
- **Pixel-only inference (D1/D6).** At runtime `zero_ad_eyes` reads frames only. Engine
  data (the fork, mods, replays) is for generating training and evaluation labels.
- **Events over silence.** Every observable transition becomes an event with a
  confidence and a `details` payload. Thresholds are recall-biased and the consumer
  filters. When unsure whether something deserves an event, emit it.
- **`best.pt` is untrusted.** A colleague trained it on a leaky per-frame split. It
  serves as a baseline to audit and as a pre-labeller.
- **Evidence over infrastructure.** With the exam this close, prefer work that yields
  real-game numbers, overlay footage or a working demo over new abstractions.
- Every tuning value lives in config (NF7); read `docs/configuration.md` before adding
  one.

## Where things stand

The plan lives in GitHub issues on `VisionArtificial-arg/0AD-Eyes`:
- **Milestones = waves** (W0–W5). Each one's gate is in its description.
- **Epics** are labelled `epic` (`[ENG]`, `[MOD]`, `[EVT]`, `[CAP]`, `[EVAL]`, `[PRES]`), and their stories are sub-issues titled `[ENG-1] …`.
- **Labels** mark the lane, `owner: you` / `owner: agent`, `optional` and `fallback`.
- **The pinned issue #42** `Roadmap to the final` (label `roadmap`) holds the critical path and the cut order.

At the start of every session, find the current wave with
`gh issue list --milestone "<wave title>"` and read the pinned roadmap issue. Close each
story when its "Done when" holds, referencing it from the commit (`Closes #N`).

`REQUIREMENTS.md` §0 is dated July and lags reality; trust the roadmap and the code. It
gets updated in PRES-1.

## Machines

- **PC (Bazzite, RX 5500):** runs 0 A.D., OBS, the engine fork build, dataset generation
  and the demo. Setup is in `docs/pc-setup.md`. The GPU has no ROCm support, so training
  happens on Kaggle/Colab.
- **Laptop (NixOS):** coding only. Run tests there through a nix shell with `uv` and
  python312, setting `LD_LIBRARY_PATH` for libstdc++, libGL, glib, zlib and libxcb.

## Sibling repos (`~/dev/va-final/` on the PC, org `VisionArtificial-arg`)

- `0A.DDataset`: the engine fork that renders segmentation masks. It is the label factory
  (L1). Read `docs/engine-fork.md` before touching it or the dataset pipeline.
- `TpFinalCaptura`: the `cvdump` JS mod (fallback labeller), plus a split-by-match
  splitter and dedup code worth reusing. Its `docs/api_verificada.md` checks the
  Release 28 script APIs against the source.

## Commands

`just --list` shows the recipes; `just validate` (check + test + eval) is the CI gate and
must be green before a commit. Learned detectors need a torch extra: `uv sync --extra cpu`.
