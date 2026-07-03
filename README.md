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
port (see `REQUIREMENTS.md` §5.10). Until that adapter lands, `run` defaults to the
stub detector: HUD/minimap are classical, while main-viewport detections are empty.

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
