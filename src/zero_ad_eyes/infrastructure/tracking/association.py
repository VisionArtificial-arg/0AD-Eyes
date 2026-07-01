"""Data association for multi-object tracking (REQUIREMENTS.md G1, CV-15/CV-16).

Greedy IoU / nearest-neighbour matching between the tracks carried from the previous
frame and the detections of the current frame. Deliberately dependency-light: a
small numpy IoU matrix plus a greedy assignment, so no scipy/Hungarian is needed
(the greedy solver is deterministic and adequate for the RTS entity densities we
target; it can be swapped for a full assignment later without touching callers).

All geometry is expressed in screen pixels via the domain ``ScreenBBox``; the module
knows nothing about tracks or entities, only boxes — keeping association reusable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from zero_ad_eyes.domain.geometry import ScreenBBox


def iou(a: ScreenBBox, b: ScreenBBox) -> float:
    """Intersection-over-union of two axis-aligned screen boxes, in [0, 1]."""

    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height

    inter_w = max(0.0, min(ax2, bx2) - max(a.x, b.x))
    inter_h = max(0.0, min(ay2, by2) - max(a.y, b.y))
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def iou_matrix(tracks: Sequence[ScreenBBox], detections: Sequence[ScreenBBox]) -> np.ndarray:
    """Row = track, column = detection; entry = IoU. Shape ``(len(t), len(d))``."""

    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for i, track_box in enumerate(tracks):
        for j, det_box in enumerate(detections):
            matrix[i, j] = iou(track_box, det_box)
    return matrix


@dataclass(frozen=True)
class Assignment:
    """Result of matching tracks to detections for one frame.

    Indices reference the input sequences positionally: ``matches`` pairs a track
    index with a detection index; the two ``unmatched_*`` tuples list the leftovers.
    """

    matches: tuple[tuple[int, int], ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]


def greedy_match(
    tracks: Sequence[ScreenBBox],
    detections: Sequence[ScreenBBox],
    *,
    iou_threshold: float = 0.3,
) -> Assignment:
    """Greedily pair the highest-IoU track/detection boxes above ``iou_threshold``.

    Pairs are considered in descending IoU order; each track and each detection can
    be claimed at most once. Ties break deterministically on (track, detection)
    index so the same inputs always yield the same assignment (NF5).
    """

    n_tracks, n_dets = len(tracks), len(detections)
    if n_tracks == 0 or n_dets == 0:
        return Assignment(
            matches=(),
            unmatched_tracks=tuple(range(n_tracks)),
            unmatched_detections=tuple(range(n_dets)),
        )

    matrix = iou_matrix(tracks, detections)

    # Candidate pairs above threshold, ordered by -IoU then indices (determinism).
    candidates = sorted(
        (-float(matrix[i, j]), i, j)
        for i in range(n_tracks)
        for j in range(n_dets)
        if matrix[i, j] >= iou_threshold
    )

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_tracks or j in used_dets:
            continue
        matches.append((i, j))
        used_tracks.add(i)
        used_dets.add(j)

    unmatched_tracks = tuple(i for i in range(n_tracks) if i not in used_tracks)
    unmatched_detections = tuple(j for j in range(n_dets) if j not in used_dets)
    return Assignment(
        matches=tuple(matches),
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_detections,
    )
