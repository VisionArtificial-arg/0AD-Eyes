# The engine fork as a label factory

The training and evaluation labels come from a modified 0 A.D. engine:
**`VisionArtificial-arg/0A.DDataset`**. This is allowed by **D6** in `REQUIREMENTS.md`:
engine data may be used for training and evaluation, while the inference path stays
pixel-only. The fork runs only while generating data and is never imported by
`zero_ad_eyes`.

## What the fork already does (3 commits on 2026-07-03, on top of upstream `main`)

- **Segmentation render pass.** Every model is drawn a second time in a flat class colour
  into an offscreen framebuffer, then read back. This gives pixel-exact masks aligned
  with the RGB, and the renderer resolves occlusion and fog itself.
  - Classes and palette: `source/graphics/SegmentationClasses.{h,cpp}`. There are 17
    classes, identical to `CLASS_NAMES` in
    `src/zero_ad_eyes/infrastructure/model/segmentation_adapter.py`.
  - Capture: `CRenderer::RenderSegmentationDatasetCapture` in
    `source/renderer/Renderer.cpp`.
  - The colour is threaded per model as `SegmentationColor` through
    `CModelAbstract`/`CModel` and `source/renderer/RenderModifiers.cpp`.
- **`CDatasetExporter`** (`source/ps/DatasetExporter.*`), configured by command-line
  flags:
  - `-generate-dataset` turns it on.
  - `-dataset-output-dir=PATH` is resolved under the user data dir's `dataset/`.
  - `-dataset-interval=N` captures every N frames.
  - `-dataset-no-random-camera` keeps the camera still. By default the camera jumps to
    a random spot, angle and zoom before each capture.
  - Output: `<split>/{rgb,mask}/frame_<runid>_<index>.png`. `runid` is random and
    unique per process, so it identifies one match.
- **`source/tools/dataset/generate_dataset.ps1`**: a Windows-only loop over random
  skirmish maps, civilisations and Petra AI difficulty.
- **`source/tools/dataset/unet/`**: the U-Net training code that produced `best.pt`
  (384×288 input, inverse-frequency class weights, mIoU on val).

**Output contract:** the fork must write the layout, mask encoding and truth JSON pinned in
`docs/engine-ground-truth-export.md` ("Engine-fork dataset"). Read that before ENG-1/2/3.

## Known flaws (why `best.pt` is untrusted) and the planned fixes

| Flaw | Where | Fix (story) |
|---|---|---|
| Train/val/test is assigned **per frame** at random (70/15/15), so frames from one match leak across splits and the reported mIoU is inflated. | `CDatasetExporter::PickSplit` | Ignore the engine's split and re-split by `runid` on ingest (ENG-6). |
| The RGB is re-rendered with the GUI off (`RenderFrameImpl(*swapChain, false, false)`), so training frames lack the HUD that real play has. | `RenderSegmentationDatasetCapture` | Render the GUI into the RGB; the mask pass stays scene-only (ENG-2). |
| The colour is per class, so same-class entities that touch merge into one blob. | `SegmentationColor` | Encode class + per-frame instance index in the colour (ENG-1). |
| No per-frame metadata: owner, hp, template, resources and camera pose are missing. | — | Write a truth JSON per pair from the simulation (ENG-3). |
| The 384×288 input loses small units. | `train.py` | Our model trains at imgsz 1280 (MOD-2). |

## Building on Linux (the PC runs Bazzite, so use a distrobox)

- Upstream build guide: <https://gitea.wildfiregames.com/0ad/0ad/wiki/BuildInstructions>.
  The fork is upstream `main` (newer than Release 28) plus about 1,000 lines, so follow
  the guide for `main`. The usual flow is `libraries/build-source-libs.sh`, then
  `build/workspaces/update-workspaces.sh`, then `make -C build/workspaces/gcc -j$(nproc)`.
- **Game art is in Git LFS** (`.gitattributes`: `*.dds *.dae *.png *.ogg …`), and
  `.lfsconfig` points LFS at **upstream** Wildfire Games, not GitHub. Clone with
  `GIT_LFS_SKIP_SMUDGE=1`, then run `git lfs pull`. This is a multi-GB download, and
  without it the game starts with no art.
- Run the built binary from `binaries/system/pyrogenesis` inside the distrobox. GPU
  access comes through `/dev/dri`, which distrobox shares by default.
- If the build fails, read the error before changing the fork: most failures are a
  missing `-devel` package in the container.

## Alternative if the build fails: `cvdump` (ENG-F)

`VisionArtificial-arg/TpFinalCaptura` has a JS GUI mod (`mod/gui/session/cvdump.js`) that
runs on stock 0 A.D. It dumps the on-screen entities (template, owner, 3D position) and a
screen↔world calibration grid via `Engine.GetTerrainAtScreenPoint`, and Python fits a
homography to turn those into YOLO boxes.

Its `docs/api_verificada.md` checks the Release 28 script APIs against the source:
- `Engine.WriteJSONFile` exists.
- There is no screenshot function for scripts, so frames must be captured externally
  and matched by timestamp.
- The `-autostart*` flags are verified.

The repo also has a split-by-match splitter and dedup code worth reusing in ENG-6. Its
weaknesses: it has never run against the real game, box heights are placeholders, and it
assumes flat maps.

## From runs to a training set (ENG-6)

```bash
uv run zero-ad-eyes dataset build --input <dataset-root> --output data/yolo
```

- **Input:** every `<run_id>/` with a `run.json` under `<dataset-root>`.
- **Output:** an Ultralytics YOLO-seg dataset: `images/` and `labels/` per split,
  `data.yaml`, and `report.json` with runs, frames and instances per split and per
  class.
- **Splitting is by run (match), never by frame.** The assignment is deterministic
  (`dataset.split_seed`), and val and test each get at least one run once there are
  three or more.
- **Near-duplicate frames inside a run are dropped,** using `dataset.dedup_threshold`
  on 32 px grayscale thumbnails.
- **Labels come from the instance masks:** one polygon per visible entity, over 7
  classes (wood, stone, metal, food, fauna, building, unit).
- **Settings:** all knobs are in the `dataset` config section.
  `dataset.image_variant: rgb_nogui` trains on the GUI-off frames instead.

## Scoring the eyes on held-out runs (EVAL-1, MOD-1, EVAL-3)

```bash
# Full scorecard for the default detector on the held-out (test) runs:
uv run zero-ad-eyes eval --engine-run <dataset-root> --split test --output card.json

# The best.pt audit (MOD-1): boxes and tracking via the learned adapter, plus U-Net mIoU,
# once on GUI-on frames and once on GUI-off frames (it was trained without the HUD):
uv run --extra cpu zero-ad-eyes eval --engine-run <dataset-root> --split test \
    --detector learned --seg-audit --output audit-gui.json
uv run --extra cpu zero-ad-eyes eval --engine-run <dataset-root> --split test \
    --detector learned --seg-audit --image-variant rgb_nogui --output audit-nogui.json
```

- **Same split everywhere.** `--split` uses the dataset build's assignment (same
  `dataset.split_seed`), so every model is scored on the same matches and none of them
  trained on those.
- **The scorecard measures what a consumer receives:**
  - detection AP50 / mAP50 / mAP50-95, both coarse (unit / building / resource node)
    and per class (7 classes);
  - HUD exact-match per field and mean relative error;
  - ownership accuracy and health error;
  - MOTA;
  - minimap blip precision/recall;
  - event counts per kind.
- **`--seg-audit`** adds IoU per class, plain mIoU and **entity mIoU** (entity classes
  only, since terrain inflates the plain number).
- **Unmeasured metrics** print as `unmeasured`, never as a pass.
