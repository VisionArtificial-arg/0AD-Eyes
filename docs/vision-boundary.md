# Vision Boundary

The runtime is split between deterministic computer vision and the learned model
adapter. The split is intentional: CV handles stable UI/symbolic surfaces; the
model owns cluttered 3D scene understanding.

## Traditional CV

These stages stay classical and run without model weights:

- HUD calibration and profile reuse.
- HUD resources and population OCR.
- Optional HUD fields when calibrated: player color sampling, selection-panel OCR,
  and simple UI state cues.
- Minimap crop, blips, viewport, fog grid, and territory/player-color regions.
- Post-detection enrichment: ownership color, visible health bars, selection rings,
  construction/garrison cues.
- Tracking, minimap fusion, event derivation, recording, overlay, config, and
  evaluation plumbing.

## Model Adapter

These stages belong behind the `PerceptionModel` port and should come from the
model team adapter:

- Main-viewport entity detection and localization.
- Unit/building/resource classification.
- Reliable resource-node detection in the 3D scene.
- Semantic segmentation.
- Instance segmentation and crowded-object separation.

## Runtime Default

`zero-ad-eyes run` defaults to `--detector stub`. That means no traditional-CV
main-viewport detections are emitted by default; HUD and minimap still run through
the classical readers. `--detector classical` remains available only as a debug
baseline for the old template/resource-cue detector.
