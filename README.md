# 0AD-Eyes

Pure-pixel **computer-vision perception layer** for an autonomous
[0 A.D.](https://play0ad.com/) player. It turns rendered game frames into a
structured **world model** and hands it to a separate decision layer.

See [`REQUIREMENTS.md`](./REQUIREMENTS.md) for the full requirements, the CV
technique catalog, the model plug-and-play boundary, and the deliverable pipeline.

## Architecture (onion)

```
interface/        CLI, debug overlay
application/       ports (FrameSource, PerceptionModel) + pipeline orchestration
domain/            world-model value objects — the integration seam (no I/O, no frameworks)
infrastructure/    adapters: capture, preprocessing, perception, model stub/real
```

The **domain** core and the **ports** are committed first; every adapter is built
and tested against them. The trained model plugs in behind the `PerceptionModel`
port (see `REQUIREMENTS.md` §5.10) — production is **vision-only**.

## Developing

```bash
uv sync            # create the locked environment
just check         # lint + format-check + type-check
just test          # automated tests
just validate      # the full CI action (check + test + eval)
just run           # launch the perception layer + overlay
```

Live screen capture needs an X display; live OCR needs the system `tesseract`
binary. Offline (recordings/fixtures) runs without either.
