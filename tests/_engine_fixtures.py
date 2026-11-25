"""Synthetic engine-fork runs, written exactly as the fork must write them (ENG-1/2/3).

Shared by the loader tests and the YOLO dataset build tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from zero_ad_eyes.infrastructure.data.engine_dataset import MASK_ENCODING, encode_mask

H, W = 40, 60
UNIT, BUILDING, GRASS = 16, 15, 1


def maps() -> tuple[np.ndarray, np.ndarray]:
    classes = np.full((H, W), GRASS, np.uint8)
    instances = np.zeros((H, W), np.uint16)
    classes[5:15, 10:14] = UNIT  # instance 1: a unit, 4×10 box at (10, 5)
    instances[5:15, 10:14] = 1
    classes[20:35, 30:50] = BUILDING  # instance 300 (> 255, exercises the G byte)
    instances[20:35, 30:50] = 300
    return classes, instances


def write_run(
    root: Path,
    run_id: str = "a1b2c3d4",
    *,
    view_player: int = 1,
    frames: int = 2,
    distinct: bool = False,
) -> Path:
    """A run dir. ``distinct`` makes every frame's RGB different (so none dedup away)."""
    run = root / run_id
    for sub in ("rgb", "mask", "truth"):
        (run / sub).mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "map_name": "skirmishes/acropolis_bay_2p",
                "view_player": view_player,
                "width": W,
                "height": H,
                "map_size": 1024.0,
                "mask_encoding": MASK_ENCODING,
            }
        ),
        encoding="utf-8",
    )
    mask_bgr = cv2.cvtColor(encode_mask(*maps()), cv2.COLOR_RGB2BGR)
    for i in range(frames):
        stem = f"{i:06d}"
        shade = (i * 60) % 256 if distinct else 0
        cv2.imwrite(str(run / "rgb" / f"{stem}.png"), np.full((H, W, 3), shade, np.uint8))
        cv2.imwrite(str(run / "mask" / f"{stem}.png"), mask_bgr)
        truth = {
            "frame_id": i,
            "timestamp": float(i),
            "resources": {"food": 300},
            "population_current": 5,
            "population_cap": 20,
            "entities": [
                {
                    "entity_id": 101,
                    "kind": "unit",
                    "owner": 1,
                    "mask_index": 1,
                    "visibility": "visible",
                    "template": "units/athen/infantry_spearman_b",
                },
                {
                    "entity_id": 202,
                    "kind": "building",
                    "owner": 2,
                    "mask_index": 300,
                    "visibility": "visible",
                },
                {
                    "entity_id": 303,
                    "kind": "unit",
                    "owner": 2,
                    "mask_index": 7,
                    "visibility": "visible",
                },  # occluded: index 7 has no pixels
                {"entity_id": 404, "kind": "unit", "owner": 1, "visibility": "hidden"},
            ],
            "players": [{"player_id": 1, "civ": "athen", "color": {"r": 46, "g": 46, "b": 200}}],
            "camera": {"x": 1.0, "y": 2.0, "z": 3.0, "rot_x": 0.8, "rot_y": 0.0, "zoom": 120.0},
        }
        (run / "truth" / f"{stem}.json").write_text(json.dumps(truth), encoding="utf-8")
    return run
