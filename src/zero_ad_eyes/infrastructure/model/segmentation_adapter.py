"""Learned ``PerceptionModel`` adapter (MP4) — U-Net semantic segmentation.

Bridges a per-pixel segmentation model to the entity-level ``Detections`` the port
promises: run the U-Net, argmax to a class map, then per foreground class run
connected-components to recover one ``Detection`` (bbox + coarse ``EntityKind``)
per blob. Every detection is tagged :class:`Provenance.LEARNED`.

Torch is a heavy, *optional* dependency (`uv sync --extra gpu` or `--extra cpu`); it
and the :mod:`unet` architecture are imported lazily inside ``from_weights`` so the
core (numpy + opencv) stays framework-free and headless-importable. Inference runs on
CUDA automatically when a GPU is visible (see ``from_weights``).

Deliberate v1 simplifications (config-externalise later, per NF7):
- input size / normalization / thresholds are named constants here, not yet in the
  typed config — the training spec was not shipped with the checkpoint;
- semantic (not instance) segmentation, so touching same-class entities merge into
  one blob → one detection; matches ``best.pt``'s nature, revisit with an entity-id
  render pass (see the exporter/dataset strategy).
- ``roi`` is accepted for port-compatibility but not yet used to crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind

if TYPE_CHECKING:  # torch only present with the `learned` extra
    import torch

    from zero_ad_eyes.application.settings import SegmentationModelSettings

# --- Model-intrinsic constants (tied to best.pt; not user tunables) --------- #

DEFAULT_WEIGHTS = Path(__file__).with_name("best.pt")
NUM_CLASSES = 17
BASE_CHANNELS = 32

#: class-id → human label (the training taxonomy, supplied at hand-off).
CLASS_NAMES: dict[int, str] = {
    0: "unlabeled",
    1: "terrain_grass",
    2: "terrain_sand",
    3: "terrain_dirt",
    4: "terrain_rock",
    5: "terrain_snow",
    6: "terrain_road",
    7: "terrain_special",
    8: "water",
    9: "resource_wood",
    10: "resource_stone",
    11: "resource_metal",
    12: "resource_food",
    13: "vegetation",
    14: "fauna",
    15: "building",
    16: "unit",
}

#: class-id → coarse ``EntityKind`` (MP2 class map). Terrain/water/backdrop resolve
#: to OTHER but are dropped before emission (see ``_EMIT_CLASSES``).
KIND_BY_CLASS_ID: dict[int, EntityKind] = {
    0: EntityKind.OTHER,
    1: EntityKind.OTHER,
    2: EntityKind.OTHER,
    3: EntityKind.OTHER,
    4: EntityKind.OTHER,
    5: EntityKind.OTHER,
    6: EntityKind.OTHER,
    7: EntityKind.OTHER,
    8: EntityKind.OTHER,
    9: EntityKind.RESOURCE_NODE,
    10: EntityKind.RESOURCE_NODE,
    11: EntityKind.RESOURCE_NODE,
    12: EntityKind.RESOURCE_NODE,
    13: EntityKind.OTHER,
    14: EntityKind.UNIT,
    15: EntityKind.BUILDING,
    16: EntityKind.UNIT,
}

#: The classes worth emitting as entities. Terrain (0-8), decorative vegetation (13)
#: and the unlabeled backdrop are scene, not entities, so they are not detected.
_EMIT_CLASSES: frozenset[int] = frozenset({9, 10, 11, 12, 14, 15, 16})

# --- Preprocessing / post-processing defaults (v1; pre-config) -------------- #

INPUT_HEIGHT = 288  # training input is 384×288 (W×H) → tensor (1, 3, 288, 384)
INPUT_WIDTH = 384
PIXEL_SCALE = 1.0 / 255.0  # [0,1] scaling; no ImageNet mean/std (matches training)
SCORE_THRESHOLD = 0.0  # the model has no confidence gate — argmax always wins
MIN_REGION_AREA = 16  # drop pixel-speckle components (seg→entity denoise, adapter concern)


class SegmentationPerceptionModel:
    """A ``PerceptionModel`` backed by the U-Net weights in ``best.pt``."""

    def __init__(
        self,
        model: Any,
        *,
        device: Any = None,
        input_size: tuple[int, int] = (INPUT_HEIGHT, INPUT_WIDTH),
        score_threshold: float = SCORE_THRESHOLD,
        min_region_area: int = MIN_REGION_AREA,
    ) -> None:
        self._model = model  # a torch.nn.Module in eval mode, on ``device``
        self._device = device  # torch.device the model + inputs live on (None until loaded)
        self._input_h, self._input_w = input_size
        self._score_threshold = score_threshold
        self._min_region_area = min_region_area

    @classmethod
    def from_weights(
        cls, weights: Path | str = DEFAULT_WEIGHTS, *, device: str | None = None, **kwargs: Any
    ) -> Any:
        """Load the U-Net weights and return the adapter, on GPU when one is visible.

        Needs the optional torch extra (``uv sync --extra gpu`` or ``--extra cpu``).
        ``device`` defaults to CUDA when ``torch.cuda.is_available()``, else CPU; pass
        e.g. ``"cpu"`` or ``"cuda:1"`` to pin it.
        """

        import torch

        from .unet import UNet

        resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(Path(weights), map_location=resolved, weights_only=True)
        state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        model = UNet(n_classes=NUM_CLASSES, base_channels=BASE_CHANNELS)
        model.load_state_dict(state)
        model.to(resolved)
        model.eval()
        return cls(model, device=resolved, **kwargs)

    @classmethod
    def from_settings(
        cls, settings: SegmentationModelSettings, *, device: str | None = None
    ) -> Any:
        """Build the adapter from typed config (NF7 composition-root wiring)."""

        return cls.from_weights(
            settings.weights_path,
            device=device,
            input_size=(settings.input_height, settings.input_width),
            score_threshold=settings.score_threshold,
            min_region_area=settings.min_region_area,
        )

    def infer(self, frame: Frame, roi: ScreenBBox | None = None) -> Detections:
        import torch

        source_h, source_w = frame.image.shape[:2]
        tensor = self._to_input_tensor(frame.image)
        with torch.no_grad():
            logits = self._model(tensor)  # (1, C, h, w)
            probs = torch.softmax(logits, dim=1)[0]  # (C, h, w)
        class_map = probs.argmax(dim=0).cpu().numpy().astype(np.int32)
        prob_map = probs.max(dim=0).values.cpu().numpy()

        scale_x = source_w / self._input_w
        scale_y = source_h / self._input_h
        items = self._detections_from_mask(class_map, prob_map, scale_x, scale_y)
        return Detections(frame_id=frame.meta.frame_id, items=tuple(items))

    def _to_input_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        import torch

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(resized.astype(np.float32) * PIXEL_SCALE, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        return tensor.to(self._device) if self._device is not None else tensor

    def _detections_from_mask(
        self, class_map: np.ndarray, prob_map: np.ndarray, scale_x: float, scale_y: float
    ) -> list[Detection]:
        items: list[Detection] = []
        for class_id in sorted(_EMIT_CLASSES):
            mask = (class_map == class_id).astype(np.uint8)
            if not mask.any():
                continue
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            for label in range(1, count):  # 0 is background
                x, y, w, h, area = stats[label]
                if area < self._min_region_area:
                    continue
                region_conf = float(prob_map[labels == label].mean())
                if region_conf < self._score_threshold:
                    continue
                items.append(
                    Detection(
                        kind=KIND_BY_CLASS_ID[class_id],
                        bbox=ScreenBBox(
                            x=x * scale_x, y=y * scale_y, width=w * scale_x, height=h * scale_y
                        ),
                        confidence=Confidence(value=region_conf, provenance=Provenance.LEARNED),
                        entity_type=CLASS_NAMES[class_id],
                    )
                )
        return items
