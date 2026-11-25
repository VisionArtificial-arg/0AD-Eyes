"""Learned ``PerceptionModel`` adapter over our own YOLO11-seg weights (MOD-4).

Loads the weights named by a training ``sidecar.json`` (``training/train_yolo_seg.py``)
and turns each predicted instance into a ``Detection``:

- ``kind``: the coarse ``EntityKind`` of its class;
- ``entity_type``: the fine class name, e.g. ``resource_wood`` or ``unit``;
- ``bbox``: the instance box in full-frame pixels;
- ``contour``: its mask outline, simplified to a few dozen points;
- ``confidence``: the model score, tagged ``learned``.

Unlike the U-Net adapter it yields true instances: units standing together stay
separate.

Ultralytics (``--extra yolo``, with ``--extra cpu`` or ``--extra gpu``) is imported
lazily in ``from_sidecar``, so this module imports on the core stack. The confidence
cutoff is recall-biased by default; downstream stages and the consumer filter. ``roi``
is accepted for port compatibility but not used to crop: YOLO needs the whole frame
for context.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox, ScreenPoint
from zero_ad_eyes.domain.taxonomy import EntityKind

from .segmentation_classes import CLASS_NAMES, KIND_BY_CLASS_ID

if TYPE_CHECKING:
    from zero_ad_eyes.application.settings import YoloModelSettings

SIDECAR_FORMAT = "zero-ad-eyes/yolo-seg/1"
_KIND_BY_NAME: dict[str, EntityKind] = {
    name: KIND_BY_CLASS_ID[class_id] for class_id, name in CLASS_NAMES.items()
}

#: ``(image_bgr) -> one Ultralytics-style result`` (has ``.boxes`` and ``.masks``).
Predictor = Callable[[np.ndarray], Any]


class YoloSegPerceptionModel:
    """A ``PerceptionModel`` backed by YOLO11-seg weights + their training sidecar."""

    def __init__(
        self,
        predictor: Predictor,
        names: dict[int, str],
        *,
        max_contour_points: int = 48,
    ) -> None:
        self._predict = predictor
        self._names = names
        self._max_points = max_contour_points

    @classmethod
    def from_sidecar(
        cls,
        sidecar_path: Path | str,
        *,
        conf_threshold: float,
        iou_threshold: float,
        max_detections: int,
        device: str | None = None,
        imgsz: int | None = None,
    ) -> YoloSegPerceptionModel:
        """Load the weights the sidecar names (relative to the sidecar's folder)."""

        from ultralytics import YOLO

        path = Path(sidecar_path)
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if sidecar.get("format") != SIDECAR_FORMAT:
            raise ValueError(f"{path}: not a {SIDECAR_FORMAT} sidecar")
        names = {int(k): str(v) for k, v in sidecar["names"].items()}
        model = YOLO(str(path.parent / sidecar["weights"]))
        model_names = {int(k): str(v) for k, v in model.names.items()}
        if model_names != names:
            raise ValueError(
                f"{path}: weights' class names {model_names} differ from the sidecar's"
            )
        size = imgsz or int(sidecar["imgsz"])

        def predict(image: np.ndarray) -> Any:
            results = model.predict(
                image,
                imgsz=size,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                device=device,
                verbose=False,
            )
            return list(results)[0]

        return cls(predict, names)

    @classmethod
    def from_settings(cls, settings: YoloModelSettings) -> YoloSegPerceptionModel:
        """Build from typed config (NF7 composition-root wiring)."""

        return cls.from_sidecar(
            settings.sidecar_path,
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
            max_detections=settings.max_detections,
            imgsz=settings.imgsz or None,
        )

    def infer(self, frame: Frame, roi: ScreenBBox | None = None) -> Detections:
        result = self._predict(frame.image)
        boxes = result.boxes
        polygons: Sequence[np.ndarray] = result.masks.xy if result.masks is not None else ()
        xyxy = _to_numpy(boxes.xyxy)
        scores = _to_numpy(boxes.conf)
        classes = _to_numpy(boxes.cls).astype(int)
        items = []
        for i in range(len(scores)):
            name = self._names.get(int(classes[i]), str(classes[i]))
            x0, y0, x1, y1 = (float(v) for v in xyxy[i])
            items.append(
                Detection(
                    kind=_KIND_BY_NAME.get(name, EntityKind.OTHER),
                    bbox=ScreenBBox(x=x0, y=y0, width=max(0.0, x1 - x0), height=max(0.0, y1 - y0)),
                    confidence=Confidence(
                        value=min(1.0, max(0.0, float(scores[i]))), provenance=Provenance.LEARNED
                    ),
                    entity_type=name,
                    contour=self._contour(polygons[i]) if i < len(polygons) else (),
                )
            )
        return Detections(frame_id=frame.meta.frame_id, items=tuple(items))

    def _contour(self, polygon: np.ndarray) -> tuple[ScreenPoint, ...]:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            return ()
        epsilon = 0.5
        simplified = points
        while len(simplified) > self._max_points:
            simplified = cv2.approxPolyDP(points.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
            epsilon *= 2
        return tuple(ScreenPoint(x=float(x), y=float(y)) for x, y in simplified)


def _to_numpy(value: Any) -> np.ndarray:
    """Torch tensors (Ultralytics) or plain arrays (tests) → numpy."""

    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)
