"""Temporal tracking & world-model infrastructure (REQUIREMENTS.md EPIC G, M4).

This package turns the per-frame ``Detections`` value coming out of the model seam
into temporally-stable domain ``Entity`` objects with persistent ids, trajectories,
lifecycle, memory/staleness, motion, events, fusion and spatial reasoning.

Onion placement: this is an *infrastructure adapter* for the ``Tracker`` port
(``zero_ad_eyes.application.ports``). It depends only on the domain and application
layers and on numpy/opencv; it never imports other perception features — cross
-wiring happens later at integration (§5.10). Everything here operates on plain
domain types so the same code serves stub and real perception equally.
"""

from __future__ import annotations

from .events import EventDetector, EventKind, TrackingEvent
from .fusion import fuse_entities, resolve_conflict
from .motion import FarnebackMotionEstimator, Motion, motion_from_trajectory
from .spatial import OccupancyGrid, distance, neighbours, proximity_pairs
from .temporal import TemporalStabilizer, majority
from .tracker import IouTracker
from .tracks import TrackStatus

__all__ = [
    "EventDetector",
    "EventKind",
    "FarnebackMotionEstimator",
    "IouTracker",
    "Motion",
    "OccupancyGrid",
    "TemporalStabilizer",
    "TrackStatus",
    "TrackingEvent",
    "distance",
    "fuse_entities",
    "majority",
    "motion_from_trajectory",
    "neighbours",
    "proximity_pairs",
    "resolve_conflict",
]
