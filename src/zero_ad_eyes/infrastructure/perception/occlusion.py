"""E7 — Occlusion / partial-visibility handling (CV-27).

Pure geometric logic over ``Detection`` boxes — no pixels — so it is buildable
and testable today against stub detections, then runs unchanged on real E1/E8
output later. It answers: how much of each detection is hidden behind others,
and which detections are the occluders.

Depth heuristic for 0 A.D.'s angled top-down camera: an entity whose box reaches
*lower* on the screen (larger bottom edge y) sits nearer the camera, so it
occludes overlapping entities that sit behind it. Occluded detections get a
``visible_fraction`` below 1 and a confidence scaled down accordingly — provenance
is carried through untouched (this stage never invents a source tag).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from zero_ad_eyes.domain.confidence import Confidence
from zero_ad_eyes.domain.detections import Detection
from zero_ad_eyes.domain.geometry import ScreenBBox


@dataclass(frozen=True)
class VisibilityInfo:
    """Per-detection occlusion verdict, indexed back into the source order."""

    index: int
    detection: Detection
    visible_fraction: float
    occluder_indices: tuple[int, ...]
    adjusted_confidence: Confidence

    @property
    def is_occluded(self) -> bool:
        return self.visible_fraction < 1.0


def _intersection_area(a: ScreenBBox, b: ScreenBBox) -> float:
    inter_w = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    inter_h = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    return inter_w * inter_h


def visible_fraction(target: ScreenBBox, occluders: Sequence[ScreenBBox]) -> float:
    """Fraction of ``target`` not covered, treating every ``occluder`` as in front.

    Overlap between occluders is not resolved exactly; covered area is summed and
    capped at the target's area — a conservative (lower-bound) visibility estimate.
    """

    if target.area <= 0:
        return 1.0
    covered = sum(_intersection_area(target, occ) for occ in occluders)
    covered = min(target.area, covered)
    return max(0.0, 1.0 - covered / target.area)


def _bottom_edge(detection: Detection) -> float:
    return detection.bbox.y + detection.bbox.height


def resolve_occlusions(
    detections: Sequence[Detection],
    depth: Callable[[Detection], float] = _bottom_edge,
) -> tuple[VisibilityInfo, ...]:
    """Compute visibility for each detection given who sits in front of it.

    ``depth`` maps a detection to a nearness score (larger = nearer the camera);
    the default uses the box's bottom edge. A detection is occluded by any other
    strictly nearer detection whose box overlaps it.
    """

    infos: list[VisibilityInfo] = []
    for i, target in enumerate(detections):
        occluder_boxes: list[ScreenBBox] = []
        occluder_indices: list[int] = []
        for j, other in enumerate(detections):
            if i == j:
                continue
            if depth(other) > depth(target) and _intersection_area(target.bbox, other.bbox) > 0.0:
                occluder_boxes.append(other.bbox)
                occluder_indices.append(j)

        vf = visible_fraction(target.bbox, occluder_boxes)
        adjusted = Confidence(
            value=max(0.0, min(1.0, target.confidence.value * vf)),
            provenance=target.confidence.provenance,
        )
        infos.append(
            VisibilityInfo(
                index=i,
                detection=target,
                visible_fraction=vf,
                occluder_indices=tuple(occluder_indices),
                adjusted_confidence=adjusted,
            )
        )
    return tuple(infos)
