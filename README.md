# 0AD-Eyes

Pure-pixel **computer-vision perception layer** for an autonomous
[0 A.D.](https://play0ad.com/) player. It turns rendered game frames into a
structured **world model** and hands it to a separate decision layer.

See [`REQUIREMENTS.md`](./REQUIREMENTS.md) for the full requirements, the CV
technique catalog, the model plug-and-play boundary, and the deliverable pipeline.
See [`docs/vision-boundary.md`](./docs/vision-boundary.md) for the current split:
traditional CV owns HUD/minimap/calibration/enrichment; the learned model owns
main-viewport detection/classification/segmentation.

## Architecture (onion)

```
interface/        CLI, debug overlay
application/       ports (FrameSource, PerceptionModel) + pipeline orchestration
domain/            world-model value objects — the integration seam (no I/O, no frameworks)
infrastructure/    adapters: capture, preprocessing, perception, model stub/real
```

The **domain** core and the **ports** are committed first; every adapter is built
and tested against them. The trained model plugs in behind the `PerceptionModel`
port (see `REQUIREMENTS.md` §5.10). By default `run` uses the **stub** detector
(HUD/minimap classical, main-viewport empty); pass `--detector learned` to run the
trained U-Net segmentation adapter (needs the optional torch extra — see below).

## Getting started (fresh clone)

Run these in order.

```bash
# 1. Install uv (the package/venv manager), if you don't have it.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter the repo.
git clone git@github.com:VisionArtificial-arg/0AD-Eyes.git
cd 0AD-Eyes

# 3. Create the locked base environment (numpy + opencv; no ML frameworks).
uv sync

# 4. Sanity check — no display or GPU needed.
uv run zero-ad-eyes --help
just validate                       # lint + type-check + tests + eval

# 5. Install torch for the learned detector — pick EXACTLY ONE:
uv sync --extra gpu                 # CUDA build (needs an NVIDIA driver); auto-uses the GPU
uv sync --extra cpu                 # CPU-only wheels (smaller; no GPU required)
#   GPU note: the index is cu124 (pyproject [[tool.uv.index]] "pytorch-cuda").
#   If your driver needs a different CUDA, change it to e.g. cu121/cu126 and re-lock.
uv run --extra gpu python -c "import torch; print('cuda:', torch.cuda.is_available())"

# 6. Run the learned detector. Keep the SAME --extra you installed, so `uv run`
#    doesn't drop torch from the environment.
uv run --extra gpu zero-ad-eyes run --recording <video.mp4> --detector learned --output out.jsonl
uv run --extra gpu zero-ad-eyes run --live --detector learned      # streams until Ctrl-C
```

Notes:
- `--detector learned` auto-selects CUDA when a GPU is visible, else CPU — no flag needed.
  Force it with the adapter's `device` arg (`"cpu"`, `"cuda:1"`) if you wire a custom call.
- Swap `--extra gpu` for `--extra cpu` throughout if you installed the CPU build.
- Live capture / HUD-OCR additionally need a display and the system `tesseract` binary
  (on Wayland, capture uses `grim`); offline `--recording` runs need neither.

### Windows: capture just the game window

The default backend (`mss`) grabs a whole monitor. To capture **only the 0 A.D.
window** on Windows, use the `window` backend: it locates the window by a title
substring every frame (so it follows moves/resizes) and scrapes its client area —
which works on 0 A.D.'s OpenGL surface where `PrintWindow`/BitBlt return black frames.

```bash
uv sync --extra gpu                 # torch (CUDA); auto-uses the GPU
uv pip install pywin32              # only the window backend needs this (Windows-only)

# run live, capturing just the 0 A.D. window, with the learned detector:
uv run --extra gpu zero-ad-eyes run --live --detector learned \
    --config docs/config.windows.json --record
```

`docs/config.windows.json` just flips two keys (everything else keeps its default):

```json
{ "acquisition": { "capture_backend": "window", "window_title": "0 A.D." } }
```

`window_title` is a **substring** of the window title — edit it if your build titles
the window differently. The window must be **visible / not occluded** (it is a screen
scrape). Equivalent env override, no file needed:
`set ZAE_ACQUISITION__CAPTURE_BACKEND=window` and `set ZAE_ACQUISITION__WINDOW_TITLE=0 A.D.`.

## Developing

```bash
uv sync            # create the locked environment
just check         # lint + format-check + type-check
just test          # automated tests
just validate      # the full CI action (check + test + eval)
just run           # live capture: write raw video, overlay video, and JSONL
just run-stdout    # write JSONL and mirror world models to stdout
just smoke-live    # one-frame live capture + overlay smoke
just record-live   # live raw recording + sibling overlay recording
just replay PATH   # replay a recording and write world models to JSONL
```

Live capture, live calibration, and live OCR need a real display plus the system
`tesseract` binary. Offline (recordings/fixtures) runs without either.
`run` writes live raw video, a sibling overlay video, and world models under
`recordings/` by default; pass `--stdout` when you also want JSON lines in the terminal.
A live `run` with no `--frames` captures until you press `Ctrl-C`, which flushes the
JSONL and finalizes the recording cleanly; pass `--frames N` to cap it instead.

## Configuration

Every tuning value (thresholds, palettes, HSV windows, layout fractions, FPS,
accuracy targets) is config-driven — no code change needed to retune. Pass
`--config my.json` to `run`/`eval`/`bench`, or override single leaves via `ZAE_*`
environment variables (defaults < file < env). See
[`docs/configuration.md`](./docs/configuration.md) for the guide and
[`docs/config.example.json`](./docs/config.example.json) for the full default tree.
