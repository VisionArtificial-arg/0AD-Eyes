"""MOD-1 — score a semantic segmenter's class map against engine-fork masks (mIoU).

``audit_segmentation`` runs a segmenter (anything mapping a ``Frame`` to an H×W class
map, e.g. ``SegmentationPerceptionModel.segment``) over runs and accumulates one
confusion matrix over the 17 engine classes. It reports:

- **IoU per class** for every class that appears in the truth or the predictions;
- **mIoU** over classes present in the truth;
- **entity mIoU** over the entity classes only (wood, stone, metal, food, fauna,
  building, unit), the ones the eyes must actually find; terrain dominates pixel
  counts and would flatter the plain mIoU;
- **pixel accuracy**.

Pass held-out runs only: scoring on training matches measures memory, not vision.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, Field

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.model.segmentation_classes import (
    CLASS_NAMES,
    ENTITY_CLASS_IDS,
    NUM_CLASSES,
)

from .engine_dataset import EngineDatasetRun, decode_mask, read_mask

Segmenter = Callable[[Frame], np.ndarray]


class SegmentationAudit(BaseModel):
    frames: int
    pixel_accuracy: float | None = None
    iou: dict[str, float] = Field(default_factory=dict)
    miou: float | None = None
    miou_entities: float | None = None


class ConfusionMatrix:
    """Pixel confusion counts, ``counts[truth, predicted]``."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.counts = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.frames = 0

    def add(self, predicted: np.ndarray, truth: np.ndarray) -> None:
        if predicted.shape != truth.shape:
            raise ValueError(f"shape mismatch: predicted {predicted.shape} vs truth {truth.shape}")
        n = self.counts.shape[0]
        index = truth.astype(np.int64).ravel() * n + predicted.astype(np.int64).ravel()
        self.counts += np.bincount(index, minlength=n * n).reshape(n, n)
        self.frames += 1

    def result(self) -> SegmentationAudit:
        counts = self.counts
        total = int(counts.sum())
        if total == 0:
            return SegmentationAudit(frames=self.frames)
        hits = np.diag(counts).astype(np.float64)
        truth_pixels = counts.sum(axis=1)
        union = truth_pixels + counts.sum(axis=0) - hits
        iou = {CLASS_NAMES[c]: float(hits[c] / union[c]) for c in range(len(hits)) if union[c] > 0}
        present = [c for c in range(len(hits)) if truth_pixels[c] > 0]
        entities = [c for c in present if c in ENTITY_CLASS_IDS]

        def mean(classes: list[int]) -> float | None:
            return float(np.mean([hits[c] / union[c] for c in classes])) if classes else None

        return SegmentationAudit(
            frames=self.frames,
            pixel_accuracy=float(hits.sum() / total),
            iou=iou,
            miou=mean(present),
            miou_entities=mean(entities),
        )


def audit_segmentation(
    segment: Segmenter, runs: Sequence[EngineDatasetRun], *, variant: str = "rgb"
) -> SegmentationAudit:
    """Accumulate ``segment``'s class maps against every frame's mask in ``runs``."""

    matrix = ConfusionMatrix()
    for run in runs:
        for paths in run.frames():
            image_path = paths.rgb_nogui if variant == "rgb_nogui" else paths.rgb
            if image_path is None:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            frame = Frame(
                image=image,
                meta=FrameMeta(
                    frame_id=0,
                    timestamp=0.0,
                    source=run.manifest.run_id,
                    width=width,
                    height=height,
                ),
            )
            truth, _ = decode_mask(read_mask(paths.mask))
            matrix.add(segment(frame), truth)
    return matrix.result()
