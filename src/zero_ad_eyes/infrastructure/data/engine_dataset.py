"""Engine-fork dataset: run layout, instance-mask codec and loader (ML2, D6).

The ``0A.DDataset`` engine fork renders, for each captured frame, the normal RGB, a
flat-colour instance mask, and a truth JSON. It writes one directory per run (one
match)::

    <dataset-root>/<run_id>/
        run.json                RunManifest
        rgb/<frame>.png         the frame as the player sees it (GUI on)
        rgb_nogui/<frame>.png   optional: the same frame with the GUI off
        mask/<frame>.png        instance mask, encoded as MASK_ENCODING (below)
        truth/<frame>.json      EngineFrameState for that frame

``<frame>`` is the zero-padded capture index (``000042``); the same stem names the
files of one frame in every folder. The full contract is in
``docs/engine-ground-truth-export.md`` ("Engine-fork dataset").

Mask encoding ``r-class_gb-instance/1`` (PNG, 8-bit RGB, no anti-aliasing):

- **R** = segmentation class id (``segmentation_classes.CLASS_NAMES``, 0–16).
- **G·B** = per-frame instance index, ``G * 256 + B`` (1–65535); 0 = no entity
  (terrain, water, sky). Each entity's ``mask_index`` in the truth JSON names its
  index, so the index is a per-frame slot, not the engine entity id.

WARNING — offline-only, like ``ground_truth``: never import this on the inference path.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.world_model import FrameMeta

from .ground_truth import EngineFrameState, EngineStateExport

MASK_ENCODING = "r-class_gb-instance/1"
OBSERVER = -1


class RunManifest(BaseModel):
    """``run.json``: what one engine-fork run (one match) contains."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    map_name: str
    view_player: int = Field(ge=OBSERVER)  # player whose view/HUD is rendered; -1 = observer
    ally_players: tuple[int, ...] = ()
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    map_size: float | None = Field(default=None, gt=0)  # world units per side
    mask_encoding: str = MASK_ENCODING
    gui_in_rgb: bool = True
    engine_commit: str | None = None


class InstanceStats(NamedTuple):
    """Where one mask instance sits on screen and how many pixels it covers."""

    bbox: ScreenBBox
    pixels: int
    class_id: int | None = None  # the mask's most common class there (if a class map was given)


def encode_mask(class_map: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    """Pack a class map (H×W, ≤255) and instance map (H×W, ≤65535) into an RGB mask."""

    if class_map.shape != instance_map.shape:
        raise ValueError(f"shape mismatch: {class_map.shape} vs {instance_map.shape}")
    instances = instance_map.astype(np.uint16)
    return np.dstack(
        (
            class_map.astype(np.uint8),
            (instances >> 8).astype(np.uint8),
            (instances & 0xFF).astype(np.uint8),
        )
    )


def decode_mask(mask_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an RGB mask into ``(class_map uint8, instance_map uint16)``."""

    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"expected an H×W×3 mask, got shape {mask_rgb.shape}")
    class_map = mask_rgb[..., 0].astype(np.uint8)
    instance_map = (mask_rgb[..., 1].astype(np.uint16) << 8) | mask_rgb[..., 2].astype(np.uint16)
    return class_map, instance_map


def read_mask(path: Path) -> np.ndarray:
    """Load a mask PNG as RGB (OpenCV reads BGR; the encoding is defined on RGB)."""

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read mask {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def instance_stats(
    instance_map: np.ndarray, class_map: np.ndarray | None = None
) -> dict[int, InstanceStats]:
    """Tight screen box + pixel count (+ majority class) for every non-zero instance."""

    height, width = instance_map.shape
    flat = instance_map.ravel()
    positions = np.flatnonzero(flat)
    if positions.size == 0:
        return {}
    ids = flat[positions]
    order = np.argsort(ids, kind="stable")
    ids, positions = ids[order], positions[order]
    unique, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    ys, xs = positions // width, positions % width
    classes: list[int | None] = [None] * len(unique)
    if class_map is not None:
        flat_classes = class_map.ravel()[positions]
        ends = np.append(starts[1:], len(positions))
        classes = [
            int(np.bincount(flat_classes[s:t]).argmax()) for s, t in zip(starts, ends, strict=True)
        ]
    x0, x1 = np.minimum.reduceat(xs, starts), np.maximum.reduceat(xs, starts)
    y0, y1 = np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts)
    return {
        int(index): InstanceStats(
            bbox=ScreenBBox(
                x=float(x0[i]),
                y=float(y0[i]),
                width=float(x1[i] - x0[i] + 1),
                height=float(y1[i] - y0[i] + 1),
            ),
            pixels=int(counts[i]),
            class_id=classes[i],
        )
        for i, index in enumerate(unique)
    }


@dataclass(frozen=True)
class EngineFramePaths:
    """The files of one captured frame. ``rgb_nogui`` is ``None`` when not rendered."""

    stem: str
    rgb: Path
    mask: Path
    truth: Path
    rgb_nogui: Path | None


class EngineDatasetRun:
    """One run directory of the engine-fork dataset."""

    def __init__(self, root: Path, manifest: RunManifest) -> None:
        self.root = root
        self.manifest = manifest

    @classmethod
    def load(cls, root: Path) -> EngineDatasetRun:
        manifest = RunManifest.model_validate_json((root / "run.json").read_text(encoding="utf-8"))
        if manifest.mask_encoding != MASK_ENCODING:
            raise ValueError(
                f"{root}: mask encoding {manifest.mask_encoding!r} is not {MASK_ENCODING!r}"
            )
        return cls(root, manifest)

    def frames(self) -> tuple[EngineFramePaths, ...]:
        """Frames that have all of rgb, mask and truth, in capture order."""

        found: list[EngineFramePaths] = []
        for rgb in sorted((self.root / "rgb").glob("*.png")):
            mask = self.root / "mask" / rgb.name
            truth = (self.root / "truth" / rgb.name).with_suffix(".json")
            if not (mask.is_file() and truth.is_file()):
                continue
            nogui = self.root / "rgb_nogui" / rgb.name
            found.append(
                EngineFramePaths(rgb.stem, rgb, mask, truth, nogui if nogui.is_file() else None)
            )
        return tuple(found)

    @staticmethod
    def truth(frame: EngineFramePaths) -> EngineFrameState:
        return EngineFrameState.model_validate_json(frame.truth.read_text(encoding="utf-8"))

    def labelled_truth(self, frame: EngineFramePaths, *, min_pixels: int = 1) -> EngineFrameState:
        """The frame's truth with every entity's ``bbox`` set from its mask pixels.

        An entity gets a box only when it covers ≥ ``min_pixels`` mask pixels and is not
        ``hidden``. The mask is the renderer's own occlusion, so an entity hidden behind
        a building has no pixels and no box. Every entity stays in the frame for
        ownership/health/tracking scoring.
        """

        truth = self.truth(frame)
        class_map, instance_map = decode_mask(read_mask(frame.mask))
        stats = instance_stats(instance_map, class_map)
        entities = []
        for entity in truth.entities:
            stat = stats.get(entity.mask_index) if entity.mask_index is not None else None
            visible = (
                stat is not None and stat.pixels >= min_pixels and entity.visibility != "hidden"
            )
            entities.append(
                entity.model_copy(
                    update={
                        "bbox": stat.bbox if visible and stat else None,
                        "seg_class": stat.class_id if stat else entity.seg_class,
                    }
                )
            )
        return truth.model_copy(update={"entities": tuple(entities)})

    def to_export(
        self, *, self_player: int | None = None, min_pixels: int = 1
    ) -> EngineStateExport:
        """The run as an ``EngineStateExport`` (boxes from masks) for the eval harness.

        Ownership needs a viewing player. Observer runs (``view_player == -1``) must pass
        ``self_player`` explicitly.
        """

        player = self.manifest.view_player if self_player is None else self_player
        if player < 0:
            raise ValueError(
                f"run {self.manifest.run_id} is an observer run; "
                "pass self_player to score ownership"
            )
        return EngineStateExport(
            match_id=self.manifest.run_id,
            self_player=player,
            ally_players=self.manifest.ally_players,
            frames=tuple(self.labelled_truth(f, min_pixels=min_pixels) for f in self.frames()),
        )


def find_runs(dataset_root: Path) -> tuple[EngineDatasetRun, ...]:
    """Every run directory (one containing ``run.json``) under ``dataset_root``."""

    return tuple(
        EngineDatasetRun.load(manifest.parent)
        for manifest in sorted(dataset_root.glob("*/run.json"))
    )


class EngineRunSource:
    """``FrameSource`` over a run's frames, stamped with the truth's ids and clock.

    Replaying a run through the pipeline this way makes every predicted world model
    carry the same ``frame_id`` as its truth file, so scoring pairs them exactly.
    ``variant="rgb_nogui"`` replays the GUI-off frames (where the run has them).
    """

    def __init__(self, run: EngineDatasetRun, *, variant: str = "rgb") -> None:
        self._run = run
        self._variant = variant

    def frames(self) -> Iterator[Frame]:
        for paths in self._run.frames():
            image_path = paths.rgb_nogui if self._variant == "rgb_nogui" else paths.rgb
            if image_path is None:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            truth = self._run.truth(paths)
            height, width = image.shape[:2]
            yield Frame(
                image=image,
                meta=FrameMeta(
                    frame_id=truth.frame_id,
                    timestamp=truth.timestamp,
                    source=self._run.manifest.run_id,
                    width=width,
                    height=height,
                ),
            )
