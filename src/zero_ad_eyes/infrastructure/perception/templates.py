"""E11 — Template matching + feature descriptors (CV-08 / CV-09).

Deterministic recognition of *fixed, known* art: UI icons and static entity
sprites whose appearance does not vary. Two complementary tools:

- ``match_template`` — normalised cross-correlation (``cv2.matchTemplate``) with
  non-maximum suppression, for pixel-exact known art.
- ``describe_features`` / feature matching — ORB keypoint descriptors (CV-08),
  for recognition that must tolerate small scale/rotation changes.

A ``TemplateBank`` bundles a set of templates and turns their hits into domain
``Detection`` items. This is the deterministic complement to the learned E1/E2
detector and the engine backing the ``ClassicalPerceptionModel``. Everything
here is ``Provenance.CLASSICAL``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.detections import Detection
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind


@dataclass(frozen=True)
class Match:
    """A single template hit: where it landed and how strong the correlation was."""

    bbox: ScreenBBox
    score: float


@dataclass(frozen=True)
class Template:
    """A fixed reference patch to look for, tagged with what it means."""

    name: str
    image: np.ndarray  # BGR reference patch
    kind: EntityKind = EntityKind.OTHER
    entity_type: str | None = None
    threshold: float = 0.8


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _iou(a: ScreenBBox, b: ScreenBBox) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    inter_w = max(0.0, min(ax2, bx2) - max(a.x, b.x))
    inter_h = max(0.0, min(ay2, by2) - max(a.y, b.y))
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _suppress(matches: list[Match], iou_threshold: float) -> list[Match]:
    """Greedy non-maximum suppression: keep strong hits, drop overlapping weaker ones."""

    kept: list[Match] = []
    for match in sorted(matches, key=lambda m: m.score, reverse=True):
        if all(_iou(match.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(match)
    return kept


def match_template(
    frame_image: np.ndarray,
    template: Template,
    roi: ScreenBBox | None = None,
    max_matches: int = 32,
    nms_iou: float = 0.3,
) -> tuple[Match, ...]:
    """Locate ``template`` in ``frame_image`` above its threshold, NMS-deduplicated.

    ``roi`` restricts the search to a sub-window; returned boxes are always in
    full-frame coordinates. Matches are returned strongest-first.
    """

    ox, oy = 0, 0
    search = frame_image
    if roi is not None:
        ox, oy = int(roi.x), int(roi.y)
        search = frame_image[oy : oy + int(roi.height), ox : ox + int(roi.width)]

    gray_search = _to_gray(search)
    gray_template = _to_gray(template.image)
    th, tw = gray_template.shape[:2]
    if gray_search.shape[0] < th or gray_search.shape[1] < tw:
        return ()

    result = cv2.matchTemplate(gray_search, gray_template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= template.threshold)
    candidates = [
        Match(
            bbox=ScreenBBox(x=float(ox + x), y=float(oy + y), width=float(tw), height=float(th)),
            score=float(result[y, x]),
        )
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    ]
    return tuple(_suppress(candidates, nms_iou)[:max_matches])


def describe_features(
    image: np.ndarray, max_features: int = 500
) -> tuple[tuple, np.ndarray | None]:
    """ORB keypoints + descriptors (CV-08) for scale/rotation-tolerant matching."""

    orb = cv2.ORB_create(nfeatures=max_features)  # type: ignore[attr-defined]
    keypoints, descriptors = orb.detectAndCompute(_to_gray(image), None)
    return tuple(keypoints), descriptors


def count_feature_matches(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
    max_distance: int = 64,
) -> int:
    """Number of good Hamming (ORB) descriptor matches between two images."""

    if descriptors_a is None or descriptors_b is None:
        return 0
    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)
    return sum(1 for m in matches if m.distance <= max_distance)


@dataclass(frozen=True)
class TemplateBank:
    """A collection of fixed templates that yields domain detections for a frame."""

    templates: tuple[Template, ...] = field(default_factory=tuple)

    def detect(
        self, frame_image: np.ndarray, roi: ScreenBBox | None = None
    ) -> tuple[Detection, ...]:
        """Match every template and emit one ``Detection`` per surviving hit."""

        detections: list[Detection] = []
        for template in self.templates:
            for hit in match_template(frame_image, template, roi=roi):
                detections.append(
                    Detection(
                        kind=template.kind,
                        bbox=hit.bbox,
                        confidence=Confidence(
                            value=min(1.0, max(0.0, hit.score)),
                            provenance=Provenance.CLASSICAL,
                        ),
                        entity_type=template.entity_type,
                    )
                )
        return tuple(detections)
