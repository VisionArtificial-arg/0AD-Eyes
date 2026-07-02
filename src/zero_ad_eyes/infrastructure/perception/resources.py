"""E6a — On-map resource-node detection, classical baseline (not model-dep).

A deliberately coarse detector that works *without* the learned model: colour +
contour cues for natural resources (green foliage → trees, grey rock → mines,
red berries → bushes, brown → fauna), plus optional template matching on fixed
resource art (E11). Recall is prioritised over precision — this is the fallback
that E6b's learned refinement supersedes where it is confident.

Every emitted ``Detection`` is ``kind == RESOURCE_NODE`` and carries
``Provenance.CLASSICAL``; the fine ``entity_type`` is the cue's coarse label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import PerceptionSettings, ResourceCueSetting
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection
from zero_ad_eyes.domain.geometry import ScreenBBox, ScreenPoint
from zero_ad_eyes.domain.taxonomy import EntityKind

from .masks import clean_mask
from .palette import HsvBand
from .templates import Template, match_template


@dataclass(frozen=True)
class ResourceCue:
    """A coarse colour signature for one class of natural resource node."""

    entity_type: str
    bands: tuple[HsvBand, ...]
    min_area: int = 20

    @classmethod
    def from_settings(cls, cue: ResourceCueSetting) -> ResourceCue:
        """Rehydrate the cv2-capable cue from its pure-data config (Approach B)."""

        return cls(
            entity_type=cue.entity_type,
            bands=tuple(HsvBand(**band.model_dump()) for band in cue.bands),
            min_area=cue.min_area,
        )


def resource_cues_from_settings(
    cues: Sequence[ResourceCueSetting],
) -> tuple[ResourceCue, ...]:
    """Map the pure-data cue list into cv2-capable cues at the boundary."""

    return tuple(ResourceCue.from_settings(cue) for cue in cues)


# Coarse, overlap-tolerant signatures (recall over precision), derived from the
# config default so there is a single source of truth.
DEFAULT_RESOURCE_CUES: tuple[ResourceCue, ...] = resource_cues_from_settings(
    PerceptionSettings().resource_cues
)


def _cue_mask(hsv: np.ndarray, cue: ResourceCue) -> np.ndarray:
    combined: np.ndarray = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for band in cue.bands:
        combined = cv2.bitwise_or(combined, band.mask(hsv))
    return clean_mask(combined, open_ksize=3, close_ksize=3)


def _detection_from_contour(contour: np.ndarray, cue: ResourceCue, ox: int, oy: int) -> Detection:
    x, y, w, h = cv2.boundingRect(contour)
    box_area = float(w * h)
    solidity = float(cv2.contourArea(contour)) / box_area if box_area > 0 else 0.0
    value = max(0.2, min(0.9, 0.3 + 0.5 * solidity))
    polygon = tuple(ScreenPoint(x=float(ox + p[0][0]), y=float(oy + p[0][1])) for p in contour)
    return Detection(
        kind=EntityKind.RESOURCE_NODE,
        bbox=ScreenBBox(x=float(ox + x), y=float(oy + y), width=float(w), height=float(h)),
        confidence=Confidence(value=value, provenance=Provenance.CLASSICAL),
        entity_type=cue.entity_type,
        contour=polygon,
    )


def detect_resource_nodes(
    frame: Frame,
    cues: Sequence[ResourceCue] = DEFAULT_RESOURCE_CUES,
    templates: Sequence[Template] = (),
    roi: ScreenBBox | None = None,
) -> tuple[Detection, ...]:
    """Detect resource nodes by colour/contour cues plus optional fixed-art templates.

    ``roi`` restricts the search; all returned boxes/contours are in full-frame
    coordinates.
    """

    image = frame.image
    h, w = image.shape[:2]
    ox, oy = 0, 0
    region = image
    if roi is not None:
        ox, oy = max(0, int(roi.x)), max(0, int(roi.y))
        region = image[oy : min(h, oy + int(roi.height)), ox : min(w, ox + int(roi.width))]
    if region.size == 0:
        return ()

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    detections: list[Detection] = []
    for cue in cues:
        mask = _cue_mask(hsv, cue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < cue.min_area:
                continue
            detections.append(_detection_from_contour(contour, cue, ox, oy))

    for template in templates:
        for hit in match_template(image, template, roi=roi):
            detections.append(
                Detection(
                    kind=EntityKind.RESOURCE_NODE,
                    bbox=hit.bbox,
                    confidence=Confidence(
                        value=min(1.0, max(0.0, hit.score)), provenance=Provenance.CLASSICAL
                    ),
                    entity_type=template.entity_type,
                )
            )

    return tuple(detections)
