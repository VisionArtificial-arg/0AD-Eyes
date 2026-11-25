"""Engine-fork runs → an Ultralytics YOLO-seg dataset (ENG-6, ML3/ML5).

Three decisions keep the dataset honest:

- **Split by run, never by frame.** One run is one match; frames of a match are
  strongly correlated, so a per-frame split leaks test scenes into training (the flaw
  behind ``best.pt``). Runs are ordered by a seeded hash of their id and dealt into
  test, val, then train, so the split is deterministic and stable as runs are added.
- **Near-duplicates are dropped within a run.** A frame whose small grayscale
  thumbnail differs from the last kept frame by less than a threshold adds nothing.
- **Labels come from the instance mask.** One polygon per visible entity instance
  (the renderer's own occlusion), its class the mask's class there. Only the 7 entity
  classes are labelled; terrain, water and vegetation are background.

Output (Ultralytics layout)::

    <out>/images/{train,val,test}/<run_id>_<frame>.png
    <out>/labels/{train,val,test}/<run_id>_<frame>.txt   class x1 y1 x2 y2 ... (normalised)
    <out>/data.yaml
    <out>/report.json                                     BuildReport

WARNING — offline-only, like the rest of this package.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from zero_ad_eyes.application.settings import DatasetSettings
from zero_ad_eyes.infrastructure.model.segmentation_classes import CLASS_NAMES, ENTITY_CLASS_IDS

from .engine_dataset import EngineDatasetRun, decode_mask, read_mask

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")

#: YOLO class index → engine class id (the 7 entity classes, in engine order).
YOLO_CLASS_IDS: tuple[int, ...] = tuple(sorted(ENTITY_CLASS_IDS))
YOLO_INDEX_BY_CLASS_ID: dict[int, int] = {cid: i for i, cid in enumerate(YOLO_CLASS_IDS)}
YOLO_NAMES: tuple[str, ...] = tuple(CLASS_NAMES[cid] for cid in YOLO_CLASS_IDS)


def split_runs(
    run_ids: Sequence[str], *, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, Split]:
    """Assign whole runs to splits. Val and test get ≥ 1 run each once there are ≥ 3."""

    ordered = sorted(run_ids, key=lambda r: hashlib.sha256(f"{seed}:{r}".encode()).hexdigest())
    n = len(ordered)

    def share(fraction: float) -> int:
        if fraction <= 0 or n < 3:
            return 0
        return max(1, round(n * fraction))

    n_test, n_val = share(test_fraction), share(val_fraction)
    n_test = min(n_test, max(0, n - 1))
    n_val = min(n_val, max(0, n - 1 - n_test))
    assigned: dict[str, Split] = {}
    for i, run_id in enumerate(ordered):
        assigned[run_id] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assigned


class Deduplicator:
    """Flags frames nearly identical to the last kept one (per run; call ``reset``)."""

    def __init__(self, *, threshold: float, thumbnail: int) -> None:
        self._threshold = threshold
        self._size = (thumbnail, thumbnail)
        self._last: np.ndarray | None = None

    def reset(self) -> None:
        self._last = None

    def is_duplicate(self, image_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, self._size, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._last is not None and float(np.abs(thumb - self._last).mean()) < self._threshold:
            return True
        self._last = thumb
        return False


def instance_polygons(
    class_map: np.ndarray,
    instance_map: np.ndarray,
    *,
    min_pixels: int,
    epsilon: float,
    skip: frozenset[int] = frozenset(),
) -> list[tuple[int, np.ndarray]]:
    """``(yolo_class, polygon N×2 in pixels)`` for each labelled entity instance.

    An instance split by occlusion into several parts keeps its largest outline.
    Instances in ``skip`` (e.g. hidden for the viewer), with fewer than ``min_pixels``
    pixels, or whose class is not an entity class are left out.
    """

    found: list[tuple[int, np.ndarray]] = []
    for index in np.unique(instance_map):
        if index == 0 or int(index) in skip:
            continue
        mask = instance_map == index
        if int(mask.sum()) < min_pixels:
            continue
        classes, counts = np.unique(class_map[mask], return_counts=True)
        class_id = int(classes[np.argmax(counts)])
        if class_id not in YOLO_INDEX_BY_CLASS_ID:
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outline = max(contours, key=cv2.contourArea)
        if epsilon > 0:
            outline = cv2.approxPolyDP(outline, epsilon, closed=True)
        points = outline.reshape(-1, 2)
        if len(points) < 3:
            continue
        found.append((YOLO_INDEX_BY_CLASS_ID[class_id], points))
    return found


def yolo_label_lines(polygons: list[tuple[int, np.ndarray]], width: int, height: int) -> list[str]:
    """YOLO-seg label lines: ``class x1 y1 x2 y2 ...`` normalised to [0, 1]."""

    lines = []
    for yolo_class, points in polygons:
        coords = points.astype(np.float64) / np.array([width, height])
        flat = " ".join(f"{v:.6f}" for v in np.clip(coords, 0.0, 1.0).ravel())
        lines.append(f"{yolo_class} {flat}")
    return lines


class SplitReport(BaseModel):
    runs: list[str] = Field(default_factory=list)
    frames: int = 0
    instances: int = 0


class BuildReport(BaseModel):
    """What ``build_yolo_dataset`` wrote (also saved as ``report.json``)."""

    splits: dict[str, SplitReport]
    duplicates_dropped: int = 0
    instances_per_class: dict[str, int]


def build_yolo_dataset(
    runs: Sequence[EngineDatasetRun], out_dir: Path, settings: DatasetSettings
) -> BuildReport:
    """Write the YOLO-seg dataset for ``runs`` into ``out_dir`` (see module docstring)."""

    assignment = split_runs(
        [r.manifest.run_id for r in runs],
        val_fraction=settings.val_fraction,
        test_fraction=settings.test_fraction,
        seed=settings.split_seed,
    )
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    report = BuildReport(
        splits={split: SplitReport() for split in SPLITS},
        instances_per_class={name: 0 for name in YOLO_NAMES},
    )
    dedup = Deduplicator(threshold=settings.dedup_threshold, thumbnail=settings.dedup_thumbnail)
    for run in sorted(runs, key=lambda r: r.manifest.run_id):
        split = assignment[run.manifest.run_id]
        split_report = report.splits[split]
        split_report.runs.append(run.manifest.run_id)
        dedup.reset()
        for frame in run.frames():
            image_path = frame.rgb_nogui if settings.image_variant == "rgb_nogui" else frame.rgb
            if image_path is None:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if dedup.is_duplicate(image):
                report.duplicates_dropped += 1
                continue
            hidden = frozenset(
                e.mask_index
                for e in run.truth(frame).entities
                if e.visibility == "hidden" and e.mask_index is not None
            )
            class_map, instance_map = decode_mask(read_mask(frame.mask))
            polygons = instance_polygons(
                class_map,
                instance_map,
                min_pixels=settings.min_instance_pixels,
                epsilon=settings.polygon_epsilon,
                skip=hidden,
            )
            name = f"{run.manifest.run_id}_{frame.stem}"
            height, width = image.shape[:2]
            shutil.copyfile(image_path, out_dir / "images" / split / f"{name}.png")
            lines = yolo_label_lines(polygons, width, height)
            (out_dir / "labels" / split / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            split_report.frames += 1
            split_report.instances += len(polygons)
            for yolo_class, _ in polygons:
                report.instances_per_class[YOLO_NAMES[yolo_class]] += 1

    (out_dir / "data.yaml").write_text(_data_yaml(out_dir), encoding="utf-8")
    (out_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def _data_yaml(out_dir: Path) -> str:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(YOLO_NAMES))
    return (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n"
    )
