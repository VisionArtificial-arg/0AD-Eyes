"""Per-entity motion estimate (REQUIREMENTS.md EPIC G, CV-17/CV-18).

Added in schema v0.2: the tracking layer estimates velocity per entity (optical
flow / trajectory), which the decision layer needs for intercepts and threat
direction. Kept a domain value object so it rides in the world model, not a
side-channel. Units are per-frame in the entity's own coordinate frame.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from .confidence import Confidence


class Motion(BaseModel):
    """Velocity of an entity between consecutive observations."""

    model_config = ConfigDict(frozen=True)

    dx: float
    dy: float
    confidence: Confidence = Confidence.unknown()

    @property
    def speed(self) -> float:
        return math.hypot(self.dx, self.dy)

    @property
    def heading(self) -> float:
        """Direction of travel in radians, ``atan2(dy, dx)``."""
        return math.atan2(self.dy, self.dx)
